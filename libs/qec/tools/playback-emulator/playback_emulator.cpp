/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "playback_emulator.h"
#include "rpc_call.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#define EMULATOR_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define EMULATOR_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define EMULATOR_CPU_RELAX()                                                   \
  do {                                                                         \
  } while (0)
#endif

namespace cudaq::qec::emulator {

namespace {

constexpr std::uint64_t kNsPerSec = 1'000'000'000ull;
/// Sleep duration used by measure_wakeup_overshoot().  wait_until() caps each
/// nap to this value so overshoot stays within the calibrated budget.
constexpr std::uint64_t kCalibrationSleepNs = 50'000;

/// CLOCK_MONOTONIC via the vDSO: ~25 ns, no syscall.  Deliberately not
/// CLOCK_MONOTONIC_RAW -- RAW skips NTP's slew but is also uncorrected for
/// crystal frequency error, which is the larger of the two over a run.
inline std::uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * kNsPerSec +
         static_cast<std::uint64_t>(ts.tv_nsec);
}

inline struct timespec to_timespec(std::uint64_t ns) {
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(ns / kNsPerSec);
  ts.tv_nsec = static_cast<long>(ns % kNsPerSec);
  return ts;
}

/// Absolute sleep, restarted on signal so a stray SIGCHLD cannot cut it short.
inline void sleep_until_ns(std::uint64_t deadline) {
  const struct timespec ts = to_timespec(deadline);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR)
    ;
}

inline void spin_until_ns(std::uint64_t deadline) {
  // Deliberately NO pause hint here.  PAUSE exists to avoid memory-order
  // machine clears and to yield to an SMT sibling while polling a cacheline
  // another core is writing.  This loop polls a CLOCK -- a vDSO read of a
  // thread-local page -- so there is no contended line and nothing to clear.
  // All the hint buys is latency: measured at 16 ns/iteration on an Arrow Lake
  // Core Ultra 7, which lands directly in the worst-case overshoot past the
  // deadline.  A compiler fence emits no instruction and only stops the empty
  // loop being optimised away.
  //
  // The flag-polling spins in ring_buffer_injector_sink.cpp DO
  // keep the hint -- see reclaim() there.
  while (now_ns() < deadline)
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

/// The one piece of control flow in the hot loop: sleep off the bulk of the
/// gap, then watch the clock for the last `slack` nanoseconds.
///
/// Sleeping alone cannot hit a deadline -- the kernel's timer slack and the
/// scheduler put wake-up overshoot in the tens of microseconds, worse under a
/// hypervisor.  Spinning alone hits it to the clock-read floor (~25 ns) but
/// burns a core for the entire run.  Doing both costs one core only for the
/// tail, which is why `slack` wants to exceed worst-case overshoot; below that
/// the sleep lands past the deadline and the spin has nothing left to correct.
inline void wait_until(std::uint64_t deadline, std::uint64_t slack) {
  // Approach the spin window in steps no longer than kCalibrationSleepNs so
  // per-step overshoot stays within the calibrated budget.  For short gaps
  // (inter-event spacing < slack) the loop is never entered: spin only.
  if (deadline > slack) {
    const std::uint64_t spin_start = deadline - slack;
    std::uint64_t now = now_ns();
    while (now < spin_start) {
      sleep_until_ns(now + std::min(spin_start - now, kCalibrationSleepNs));
      now = now_ns();
    }
  }
  spin_until_ns(deadline);
}

/// Strip a trailing `#` comment and surrounding whitespace.
std::string strip(const std::string &line) {
  const auto hash = line.find('#');
  std::string out = hash == std::string::npos ? line : line.substr(0, hash);
  const auto first = out.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = out.find_last_not_of(" \t\r\n");
  return out.substr(first, last - first + 1);
}

/// Strict unsigned parse: rejects signs, junk suffixes and overflow, all of
/// which `std::stoull` would quietly accept or throw opaquely on.
bool parse_u64(const std::string &text, std::uint64_t &out) {
  if (text.empty() ||
      !std::all_of(text.begin(), text.end(),
                   [](unsigned char c) { return std::isdigit(c); }))
    return false;
  try {
    std::size_t consumed = 0;
    out = std::stoull(text, &consumed);
    return consumed == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

/// Held open for the process lifetime: the C-state floor lasts only as long as
/// the fd, so closing it would silently give the deep idle states back.
int g_cpu_dma_latency_fd = -1;

/// Bound on the NOT_READY retry loop.
constexpr int kNotReadyTimeoutMs = 5000;

} // namespace

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Playback file
//===----------------------------------------------------------------------===//

const char *to_string(operation op) {
  switch (op) {
  case operation::enqueue:
    return "enqueue";
  case operation::get_corrections:
    return "get_corrections";
  case operation::reset:
    return "reset";
  case operation::stream_until:
    return "stream_until";
  }
  return "?";
}

bool parse_operation(const std::string &text, operation &out) {
  if (text == "enqueue")
    out = operation::enqueue;
  else if (text == "get_corrections")
    out = operation::get_corrections;
  else if (text == "reset")
    out = operation::reset;
  else if (text == "stream_until")
    out = operation::stream_until;
  else
    return false;
  return true;
}

bool takes_syndromes(operation op) { return op == operation::enqueue; }

bool returns_corrections(operation op) {
  // stream_until absorbs its read: the poll that ends the streaming loop is a
  // consuming get_corrections, so the bits land in sink::corrections().
  return op == operation::get_corrections || op == operation::stream_until;
}


bool parse_duration_ns(const std::string &text, std::uint64_t &out) {
  static const struct {
    const char *suffix;
    double scale;   // nanoseconds per unit
  } kUnits[] = {{"ns", 1.0},   {"us", 1e3},  {"\u00b5s", 1e3},
                {"ms", 1e6},   {"s", 1e9}};

  for (const auto &unit : kUnits) {
    const std::string suffix = unit.suffix;
    if (text.size() <= suffix.size() ||
        text.compare(text.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;
    // "s" also matches the tail of "ns"/"us"/"ms"; those are handled by the
    // earlier entries, so reaching here with a longer unit means a real match.
    const std::string mantissa = text.substr(0, text.size() - suffix.size());
    if (mantissa.empty())
      return false;
    char *end = nullptr;
    const double value = std::strtod(mantissa.c_str(), &end);
    if (end != mantissa.c_str() + mantissa.size() || value <= 0.0)
      return false;
    const double ns = value * unit.scale;
    if (ns < 1.0 || ns > 9.0e18)
      return false;
    out = static_cast<std::uint64_t>(ns);
    return true;
  }
  return false;   // no unit: refuse rather than assume nanoseconds
}

playback_file load_playback(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("load_playback: cannot open " + path);

  playback_file file;
  std::string line;
  std::size_t line_no = 0;

  // Every diagnostic names the file and line: a playback file is usually
  // machine-generated, and "bad record" without a location is useless.
  const auto fail = [&](const std::string &why) {
    throw std::runtime_error("load_playback: " + path + ":" +
                             std::to_string(line_no) + ": " + why);
  };

  while (std::getline(in, line)) {
    ++line_no;

    // Parse structured metadata from header comments before stripping.
    // Format: # PLAYBACK_META decoder_id=N syndrome_size=N [...]
    if (line.rfind("# PLAYBACK_META", 0) == 0) {
      std::istringstream meta_stream(line.substr(2)); // skip "# "
      std::string token;
      meta_stream >> token; // consume "PLAYBACK_META"
      std::uint64_t did = 0, ss = 0;
      bool have_did = false, have_ss = false;
      while (meta_stream >> token) {
        if (token.rfind("decoder_id=", 0) == 0)
          have_did = parse_u64(token.substr(11), did);
        else if (token.rfind("syndrome_size=", 0) == 0)
          have_ss = parse_u64(token.substr(14), ss);
      }
      if (have_did) {
        file.meta[did]; // ensure entry exists
        if (have_ss)
          file.meta[did].syndrome_size = ss;
      }
      continue;
    }

    const std::string text = strip(line);
    if (text.empty())
      continue;

    std::istringstream fields(text);
    playback_record rec{};

    std::string tick_text, op_text, decoder_text;
    if (!(fields >> tick_text >> op_text >> decoder_text))
      fail("expected `<tick> <operation> <decoder_id> [operands...]`, "
           "got: " + text);

    if (!parse_u64(tick_text, rec.tick))
      fail("tick is not an unsigned integer: " + tick_text);
    if (!parse_operation(op_text, rec.op))
      fail("unknown operation `" + op_text +
           "`; expected enqueue, get_corrections or reset");
    if (!parse_u64(decoder_text, rec.decoder_id))
      fail("decoder_id is not an unsigned integer: " + decoder_text);

    if (takes_syndromes(rec.op)) {
      std::string token;
      if (!(fields >> token))
        fail(std::string(to_string(rec.op)) +
             " needs a syndrome bit string or source_id=N");
      if (token.rfind("source_id=", 0) == 0) {
        std::uint64_t sid = 0;
        if (!parse_u64(token.substr(10), sid))
          fail("source_id= expects a non-negative integer: " + token);
        rec.source_id = static_cast<std::int64_t>(sid);
      } else {
        rec.syndrome_offset = file.syndromes.size();
        for (const char c : token) {
          if (c != '0' && c != '1')
            fail(std::string("syndrome data must be `0`/`1` characters, got `") +
                 c + "` in: " + token);
          file.syndromes.push_back(static_cast<std::uint8_t>(c - '0'));
        }
        rec.syndrome_count = token.size();
        if (rec.syndrome_count > wire::kMaxSyndromeBits)
          fail("syndrome is " + std::to_string(rec.syndrome_count) +
               " bits, over the " + std::to_string(wire::kMaxSyndromeBits) +
               "-bit wire cap");
      }
    }

    if (rec.op == operation::stream_until) {
      std::string token;
      if (!(fields >> token))
        fail("stream_until needs source_id=N");
      if (token.rfind("source_id=", 0) != 0)
        fail("stream_until requires source_id=N, got: " + token);
      std::uint64_t sid = 0;
      if (!parse_u64(token.substr(10), sid))
        fail("source_id= expects a non-negative integer: " + token);
      rec.source_id = static_cast<std::int64_t>(sid);
    }

    // Both reading ops allow an optional expected-correction bit string so the
    // emulator can verify the decoder's output in-line, without an out-of-band
    // diff.  For stream_until this follows the source_id=N token consumed
    // above.  Any other trailing operand is still an error.
    std::string extra;
    if (fields >> extra) {
      if (!returns_corrections(rec.op))
        fail("unexpected trailing operand `" + extra + "` for " +
             to_string(rec.op));
      rec.corrections_offset = file.corrections.size();
      for (const char c : extra) {
        if (c != '0' && c != '1')
          fail(std::string("expected correction bits must be `0`/`1`, got `") +
               c + "` in: " + extra);
        file.corrections.push_back(static_cast<std::uint8_t>(c - '0'));
      }
      rec.corrections_count = extra.size();
    }

    file.records.push_back(rec);
  }

  if (file.records.empty())
    throw std::runtime_error("load_playback: " + path + " has no records");
  return file;
}

std::vector<event> build_events(const playback_file &file,
                                std::uint64_t decoder_id,
                                std::uint64_t tick_ns, bool deltas,
                                std::size_t *out_skipped,
                                const source_registry &sources) {
  if (tick_ns == 0)
    throw std::runtime_error("build_events: tick duration must be non-zero");
  std::vector<event> events;
  events.reserve(file.records.size());

  std::size_t skipped = 0;
  std::uint64_t running = 0;   // delta mode: prefix sum
  std::uint64_t previous = 0;  // absolute mode: monotonicity check
  bool seen = false;

  for (const auto &rec : file.records) {
    if (rec.decoder_id != decoder_id) {
      ++skipped;
      continue;
    }

    // Timestamps resolve to offsets from t0 HERE, once.  The run loop never
    // derives one deadline from another, so a late wake-up cannot accumulate.
    std::uint64_t offset;
    if (deltas) {
      running += rec.tick;
      offset = running;
    } else {
      if (seen && rec.tick < previous)
        throw std::runtime_error(
            "build_events: absolute ticks must be non-decreasing, but tick " +
            std::to_string(rec.tick) + " follows tick " +
            std::to_string(previous) +
            " (pass --delta-ticks to read the column as gaps)");
      offset = rec.tick;
      previous = rec.tick;
    }
    seen = true;

    const std::uint8_t *syndrome_data =
        (takes_syndromes(rec.op) && rec.source_id < 0)
            ? file.syndromes.data() + rec.syndrome_offset
            : nullptr;
    const std::uint8_t *corrections_data =
        rec.corrections_count > 0
            ? file.corrections.data() + rec.corrections_offset
            : nullptr;
    // Ticks become nanoseconds HERE, once, alongside the prefix sum -- the run
    // loop never multiplies, and a late wake-up cannot accumulate.
    if (tick_ns != 0 && offset > UINT64_MAX / tick_ns)
      throw std::runtime_error(
          "build_events: tick " + std::to_string(offset) + " x " +
          std::to_string(tick_ns) + " ns overflows the schedule");
    event ev{};
    ev.op = rec.op;
    ev.syndrome_data = syndrome_data;
    ev.num_syndromes = rec.syndrome_count;
    if (rec.source_id >= 0) {
      const auto it = sources.find(rec.source_id);
      if (it == sources.end())
        throw std::runtime_error(
            "build_events: no source registered for source_id=" +
            std::to_string(rec.source_id));
      ev.source = it->second;
    }
    ev.offset_ns = offset * tick_ns;
    ev.corrections_data = corrections_data;
    ev.corrections_count = rec.corrections_count;
    events.push_back(ev);
  }

  if (events.empty())
    throw std::runtime_error(
        "build_events: no records for decoder_id " +
        std::to_string(decoder_id) + " (" + std::to_string(skipped) +
        " records belong to other decoders)");
  if (out_skipped)
    *out_skipped = skipped;
  return events;
}

//===----------------------------------------------------------------------===//
// Sinks
//===----------------------------------------------------------------------===//

sink::sink(std::uint64_t decoder_id, std::uint64_t num_observables)
    : decoder_id_(decoder_id), num_observables_(num_observables) {
  // Sized once, here, so transport() never allocates on the timing thread.
  result_scratch_.assign(
      std::max<std::size_t>(wire::bit_packed_bytes(num_observables_), 1), 0);
}

void sink::set_num_observables(std::uint64_t n) {
  num_observables_ = n;
  result_scratch_.assign(std::max<std::size_t>(wire::bit_packed_bytes(n), 1),
                         0);
}

void sink::send(const event &e, std::uint64_t tag) {
  if (e.op == operation::stream_until)
    return; // no wire RPC of its own; run() drives it via enqueue + poll
  last_not_ready_retries_ = 0;
  const rpc_call call =
      build_rpc_call(e.op, e, decoder_id_, num_observables_, tag,
                     /*consuming=*/false, wait_for_ready_);
  const std::int32_t status = dispatch_with_retry(call);
  if (status != 0)
    throw std::runtime_error(std::string(name()) + ": RPC returned status " +
                             std::to_string(status));
}

bool sink::try_get_corrections() {
  const event dummy{};
  const rpc_call call = build_rpc_call(
      operation::get_corrections, dummy, decoder_id_, num_observables_,
      next_request_id_++, /*consuming=*/true, /*retry_on_not_ready=*/false);
  const std::int32_t status = dispatch_once(call);
  if (status == static_cast<std::int32_t>(wire::RpcStatus::NOT_READY))
    return false;
  if (status != 0)
    throw std::runtime_error(std::string(name()) +
                             "::try_get_corrections: status " +
                             std::to_string(status));
  return true;
}

std::int32_t sink::dispatch_once(const rpc_call &call) {
  const std::int32_t status = transport(
      call, call.expected_result_bits > 0 ? result_scratch_.data() : nullptr);
  if (status == 0 && call.expected_result_bits > 0)
    unpack_and_log_corrections(result_scratch_.data(),
                               call.expected_result_bits);
  return status;
}

std::int32_t sink::dispatch_with_retry(const rpc_call &call) {
  if (!call.blocking)
    return transport(call, nullptr); // fire-and-forget: status ignored

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kNotReadyTimeoutMs);
  while (true) {
    const std::int32_t status = dispatch_once(call);
    if (status == 0)
      return 0;
    if (status != static_cast<std::int32_t>(wire::RpcStatus::NOT_READY))
      return status;
    ++last_not_ready_retries_;
    if (!call.retry_on_not_ready)
      throw std::runtime_error(
          std::string(name()) +
          ": RPC returned NOT_READY (the schedule may be too tight; use "
          "--wait-for-ready to retry until the decoder is done)");
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error(std::string(name()) +
                               ": wait_for_ready timed out after " +
                               std::to_string(kNotReadyTimeoutMs) + " ms");
    EMULATOR_CPU_RELAX();
  }
}

void sink::unpack_and_log_corrections(const std::uint8_t *packed,
                                      std::size_t n_bits) {
  for (std::size_t i = 0; i < n_bits; ++i)
    corrections_log_.push_back((packed[i / 8] >> (i % 8)) & 0x1u);
}

std::int32_t null_sink::transport(const rpc_call &call,
                                  std::uint8_t *result_buf) {
  // Still pays the framing/bit-packing cost a real sink would; see the class
  // comment.  The frame is built and checksummed, then discarded.
  const std::size_t frame_len =
      sizeof(cudaq::realtime::RPCHeader) + call.payload_len +
      (call.bits ? wire::bit_packed_bytes(call.num_bits) : 0);
  if (frame_scratch_.size() < frame_len)
    frame_scratch_.resize(frame_len);
  const std::size_t written =
      serialize_rpc_frame(call, static_cast<std::uint32_t>(call.tag),
                          frame_scratch_.data(), frame_scratch_.size());
  for (std::size_t i = 0; i < written; ++i)
    checksum_ += frame_scratch_[i];

  // No decoder behind this sink, so neither shape of get_corrections call
  // returns real bits (num_observables_ is 0). They still answer
  // differently: the stream_until poll (consuming, reset=1) must return
  // NOT_READY so the loop always exhausts its source; an explicit
  // get_corrections (reset=0) must succeed silently, since playback files
  // run it against `null` as a baseline. Reuse the `reset` byte already on
  // the wire to tell the two apart, same as every other sink does.
  (void)result_buf;
  if (call.function_id == wire::kGetCorrectionsFunctionId) {
    const auto *p = reinterpret_cast<const wire::GetCorrectionsRequestPayload *>(
        call.payload.data());
    if (p->reset)
      return static_cast<std::int32_t>(wire::RpcStatus::NOT_READY);
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Real-time posture
//===----------------------------------------------------------------------===//

std::vector<std::string> apply_rt_config(const run_config &cfg) {
  std::vector<std::string> warnings;

  if (cfg.pin_cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cfg.pin_cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
      warnings.push_back("could not pin to CPU " + std::to_string(cfg.pin_cpu) +
                         ": " + std::strerror(errno));
  }

  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
    warnings.push_back(std::string("mlockall failed: ") + std::strerror(errno));

  return warnings;
}

std::uint64_t measure_wakeup_overshoot(int samples) {
  std::uint64_t worst = 0;
  for (int i = 0; i < samples; ++i) {
    const std::uint64_t target = now_ns() + kCalibrationSleepNs;
    sleep_until_ns(target);
    const std::uint64_t woke = now_ns();
    if (woke > target)
      worst = std::max(worst, woke - target);
  }
  return worst;
}

//===----------------------------------------------------------------------===//
// Runner
//===----------------------------------------------------------------------===//

std::vector<record> run(const std::vector<event> &events, sink &dst,
                        const run_config &cfg) {
  // Allocated and zero-initialised before t0 so no page faults mid-run.
  std::vector<record> records(events.size());

  dst.set_wait_for_ready(cfg.wait_for_ready);
  const std::uint64_t t0 = now_ns() + cfg.lead_in_ns;
  dst.warm_up();
  // Align precisely to t0. Without this, the first event's wait_until sleeps
  // for ~lead_in_ns, and long sleeps overshoot by more than spin_slack
  // (calibrated on short samples), making every event late by a fixed offset.
  wait_until(t0, cfg.spin_slack_ns);

  std::size_t read_count = 0;
  const std::size_t width = dst.correction_width();

  // Compare the correction just produced against the record's expected bits.
  // Shared by get_corrections and stream_until: both leave one correction row
  // in dst.corrections(), indexed by read_count.  A no-op when the record
  // carried no expectation.
  const auto verify = [&](record &r, const event &e) {
    if (e.corrections_data == nullptr)
      return;
    if (width == 0 || e.corrections_count != width) {
      r.correction_mismatch = true;
      return;
    }
    const auto &corr = dst.corrections();
    const std::size_t base = read_count * width;
    if (base + width > corr.size()) {
      r.correction_mismatch = true;
      return;
    }
    for (std::size_t j = 0; j < width; ++j) {
      if ((corr[base + j] & 1u) != (e.corrections_data[j] & 1u)) {
        r.correction_mismatch = true;
        return;
      }
    }
  };

  for (std::size_t i = 0; i < events.size(); ++i) {
    const event &e = events[i];
    const std::uint64_t deadline = t0 + e.offset_ns;

    // Overrun case: if the previous send ran past this deadline, wait_until
    // falls straight through and we fire late. The schedule is never rewritten,
    // so the shift is visible in the records.
    wait_until(deadline, cfg.spin_slack_ns);

    const std::uint64_t call = now_ns();

    if (e.op == operation::stream_until) {
      record &r = records[i];
      r.deadline_ns = e.offset_ns;
      r.call_ns     = call - t0;
      event enq{}; enq.op = operation::enqueue;
      std::uint64_t synthetic_tag = 0;
      bool done = false;
      while (!done) {
        auto syndrome = e.source->next_round();
        if (syndrome.empty()) break;   // source exhausted before decoder finished
        enq.syndrome_data = syndrome.data();
        enq.num_syndromes = syndrome.size();
        dst.send(enq, synthetic_tag++);
        ++r.syndromes_streamed;
        // The poll that returns true is a CONSUMING read: the correction bits
        // are now in dst.corrections(), exactly as a get_corrections would
        // have left them.  That is what lets stream_until stand alone.
        done = dst.try_get_corrections();
      }
      r.return_ns      = now_ns() - t0;
      r.read_completed = done;
      if (done) {
        verify(r, e);
        ++read_count;
      }
      continue;
    }

    if (e.op == operation::enqueue && e.source != nullptr) {
      auto bytes = e.source->next_round();
      event dyn = e;
      dyn.syndrome_data = bytes.data();
      dyn.num_syndromes = bytes.size();
      dst.send(dyn, i);
    } else {
      dst.send(e, i);
    }
    const std::uint64_t returned = now_ns();

    record &r = records[i];
    r.deadline_ns = e.offset_ns;
    r.call_ns     = call - t0;
    r.return_ns   = returned - t0;

    if (e.op == operation::get_corrections) {
      r.not_ready_retries = dst.last_not_ready_retries();
      r.read_completed    = true;
      verify(r, e);
      ++read_count;
    }
  }
  return records;
}

//===----------------------------------------------------------------------===//
// Reporting
//===----------------------------------------------------------------------===//

std::vector<std::uint64_t> lateness_ns(const std::vector<record> &records) {
  std::vector<std::uint64_t> out;
  out.reserve(records.size());
  for (const auto &r : records)
    out.push_back(r.call_ns - r.deadline_ns);
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::uint64_t> latency_ns(const std::vector<record> &records) {
  std::vector<std::uint64_t> out;
  out.reserve(records.size());
  for (const auto &r : records)
    out.push_back(r.return_ns - r.call_ns);
  std::sort(out.begin(), out.end());
  return out;
}

std::uint64_t quantile(const std::vector<std::uint64_t> &sorted, double q) {
  if (sorted.empty())
    return 0;
  if (q <= 0.0)
    return sorted.front();
  const auto index = static_cast<std::size_t>(q * (sorted.size() - 1) + 0.5);
  return sorted[std::min(index, sorted.size() - 1)];
}

std::size_t count_overruns(const std::vector<record> &records) {
  std::size_t n = 0;
  for (std::size_t i = 1; i < records.size(); ++i)
    if (records[i - 1].return_ns > records[i].deadline_ns)
      ++n;
  return n;
}

void print_stats(const std::vector<record> &records,
                 const std::vector<double> &quantiles) {
  const std::size_t count = records.size();
  const std::size_t overruns = count_overruns(records);
  std::printf("events            %zu\n", count);
  std::printf("overruns          %zu (%.2f%%)\n", overruns,
              count ? 100.0 * overruns / count : 0.0);

  // Both rows use the same quantile set so the columns line up under each
  // other; the label is the only thing that differs.
  const std::pair<const char *, std::vector<std::uint64_t>> rows[] = {
      {"lateness (ns)    ", lateness_ns(records)},
      {"sink latency (ns) ", latency_ns(records)}};
  for (const auto &[label, dist] : rows) {
    std::printf("%s", label);
    for (double q : quantiles) {
      char name[16];
      if (q >= 1.0)
        std::snprintf(name, sizeof(name), "max");
      else
        std::snprintf(name, sizeof(name), "p%g", 100.0 * q);
      std::printf(" %s=%llu", name,
                  static_cast<unsigned long long>(quantile(dist, q)));
    }
    std::printf("\n");
  }
}

void write_csv(const std::string &path,
               const std::vector<record> &records,
               const std::vector<event> &events,
               const std::vector<std::uint8_t> &corrections,
               std::size_t correction_width) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("write_csv: cannot open " + path);
  out << "event,op,deadline_ns,call_ns,return_ns,lateness_ns,latency_ns,"
         "correction,syndromes_streamed\n";
  std::size_t read_idx = 0;
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto &r = records[i];
    const auto  op = events[i].op;
    const std::uint64_t late =
        r.call_ns > r.deadline_ns ? r.call_ns - r.deadline_ns : 0;
    out << i << ',' << to_string(op) << ','
        << r.deadline_ns << ',' << r.call_ns << ','
        << r.return_ns   << ',' << late      << ',' << (r.return_ns - r.call_ns)
        << ',';
    // Key off read_completed, not the op: a stream_until whose source ran dry
    // produced no correction, so advancing read_idx would shift every later
    // row's bits by one.
    if (r.read_completed && correction_width > 0) {
      const std::size_t base = read_idx * correction_width;
      if (base + correction_width <= corrections.size()) {
        for (std::size_t j = 0; j < correction_width; ++j)
          out << static_cast<int>(corrections[base + j] & 1u);
      } else {
        out << '-';
      }
      ++read_idx;
    } else {
      out << '-';
    }
    out << ',' << r.syndromes_streamed << '\n';
  }
}

} // namespace cudaq::qec::emulator

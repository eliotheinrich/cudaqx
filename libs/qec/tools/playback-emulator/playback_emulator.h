/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file playback_emulator.h
/// @brief Timing core of the syndrome playback emulator: stands in for a quantum
/// controller by replaying pre-recorded syndrome bits into the realtime
/// decoding API on a user-supplied schedule.
///
/// The design goal is that the ONLY work on the timing thread between two
/// deadlines is (a) waiting and (b) one sink call.  Everything else -- parsing,
/// slicing, allocation, formatting -- happens before the run anchor `t0`.
///
/// Deadlines are precomputed as absolute offsets from `t0` by prefix-summing
/// the user's relative deltas, so a late wake-up never propagates into later
/// events.  On overrun the runner fires as soon as it can and does NOT rewrite
/// the table: the schedule shifts naturally, no event is dropped, and the
/// lateness shows up in the stats rather than being silently absorbed.
///
/// This header declares `sink`'s dispatch surface but not its RPC-object
/// plumbing: `sink::transport()` takes a forward-declared `rpc_call`, whose
/// definition (and the CUDA-Q wire-format headers it needs) lives in
/// `rpc_call.h`, included only by `playback_emulator.cpp` and the concrete
/// sinks.

#include "syndrome_source.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec::emulator {

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

/// Real-time pacing knobs.
struct run_config {
  /// How long before each deadline to stop sleeping and start watching the
  /// clock.  The runner sleeps off the bulk of the gap and spins only the last
  /// stretch.
  std::uint64_t spin_slack_ns = 100'000;
  /// Gap between `run()` being called and `t0`.  Gives the caches and the sink
  /// a moment to settle.
  std::uint64_t lead_in_ns = 50'000'000;
  /// Core to pin the timing thread to, or -1 for no affinity change.
  int pin_cpu = -1;
  /// Retry policy for `get_corrections` when the decoder answers NOT_READY
  /// (fired before the decoder has finished collecting/decoding).
  ///
  /// Default (false): treated as an error -- throws, surfacing a schedule
  /// that is too tight rather than returning stale corrections.
  ///
  /// True: retries until the decoder is ready or a fixed timeout elapses.
  /// The timing thread blocks past its deadline, so latency/overrun stats
  /// reflect the wait.  `record::not_ready_retries` counts the retries.
  bool wait_for_ready = false;
};

//===----------------------------------------------------------------------===//
// Playback file
//===----------------------------------------------------------------------===//
//
// One record per line, whitespace-separated:
//
//     <tick> <operation> <decoder_id> [operands...]
//
//   tick        Tick index from the start of the run (t0); a tick's wall-clock
//               duration is supplied separately (`--tick=1us`), so one file
//               replays at any cadence. Absolute and non-decreasing by
//               default; pass `deltas = true` to `build_events` to read it as
//               a gap from the previous kept record instead.
//   operation   enqueue | get_corrections | reset | stream_until.  Unknown is
//               a parse error, never a silent skip.
//   decoder_id  Which syndrome source this record belongs to.  ONE file can
//               describe a multi-source experiment; each emulator process
//               keeps only its own decoder's records (see `build_events`).
//
// Operations and their operands:
//
//   enqueue <bits>
//       Push one round of syndrome bits (a run of `0`/`1` chars). Fire-and-
//       forget: the dispatcher ACKs but returns no body.
//
//   get_corrections [expected_bits]
//       Read back the accumulated corrections, at the decoder's declared
//       observable width. Never resets -- follow with an explicit `reset`
//       when a shot boundary needs the accumulator cleared.
//
//       Optional `expected_bits` (one `0`/`1` char per observable): `run()`
//       compares it against the actual read and sets `record::
//       correction_mismatch` on a mismatch.
//
//   reset
//       Clear the decoder's queued syndromes and zero its corrections.
//
//   stream_until source_id=N [expected_bits]
//       Feed rounds from a registered syndrome_source one at a time, polling
//       try_get_corrections() after each, until the decoder is ready or the
//       source is exhausted (empty round).
//
//       ABSORBS the read: the successful poll is a consuming get_corrections,
//       so its bits land in `sink::corrections()` same as an explicit read
//       would. Do NOT follow with a get_corrections -- the result is already
//       taken and the second read would block forever.
//
//       `expected_bits` behaves exactly as on get_corrections, e.g.:
//           0  stream_until  0  source_id=2  10   # expected: obs0=1 obs1=0
//
// Blank lines are ignored and `#` starts a comment that runs to end of line.
//
// Example (with --tick=1us):
//     0  reset            0
//     1  enqueue          0  010
//     2  get_corrections  0  10   # expected: obs0=1 obs1=0
//     3  reset            0
//
//===----------------------------------------------------------------------===//

/// Operations a record can carry.
enum class operation {
  enqueue,
  get_corrections,
  reset,
  /// Drive the decoder from a registered syndrome_source, polling after each
  /// round until the decoder signals ready or the source is exhausted.  The
  /// successful poll is a consuming read, so this op also RETURNS corrections
  /// (see `returns_corrections`).  Requires a sink implementing
  /// try_get_corrections().
  stream_until,
};

const char *to_string(operation op);
bool parse_operation(const std::string &text, operation &out);

/// True when `op` carries syndrome bits.
bool takes_syndromes(operation op);

/// True when `op` reads a result back, and so must wait for its response.
/// Both `get_corrections` and `stream_until` do: the latter absorbs the read
/// into the poll that ends its streaming loop.
bool returns_corrections(operation op);

/// One parsed record.  Syndrome/expected-correction bits live in the file's
/// shared arenas rather than in the record, so a record stays small and the
/// bits stay contiguous.
struct playback_record {
  std::uint64_t tick;
  operation op;
  std::uint64_t decoder_id;
  /// Index/count into `playback_file::syndromes`.  0/0 for non-enqueue and
  /// source-referenced enqueue records.
  std::size_t syndrome_offset;
  std::uint64_t syndrome_count;
  /// Index/count into `playback_file::corrections` (one byte per observable
  /// bit).  0/0 unless `op == get_corrections` and the line carried bits.
  std::size_t corrections_offset;
  std::uint64_t corrections_count;
  /// `source_id=N` for enqueue records referencing a source instead of
  /// inline bits.  -1 means inline.
  std::int64_t source_id = -1;
};

/// Metadata parsed from `# PLAYBACK_META` header lines.  Zero means "not
/// specified" -- load_playback never fabricates a value.
struct playback_meta {
  std::uint64_t syndrome_size = 0; ///< Total syndrome bits per shot.
};

/// A parsed playback file: every record, plus one flat buffer holding every
/// record's syndrome bits at one byte per bit (0x00 / 0x01).  That is exactly
/// the shape `rpc_producer::enqueue_syndromes` wants, so the send path copies
/// nothing -- the payloads are sliced once, here, before the run starts.
struct playback_file {
  std::vector<std::uint8_t> syndromes;    ///< Syndrome bit arena.
  std::vector<std::uint8_t> corrections;  ///< Expected correction bit arena.
  std::vector<playback_record> records;
  std::map<std::uint64_t, playback_meta> meta; ///< Keyed by decoder_id.
};

/// A scheduled send, fully resolved at load time: a slice of the arena plus
/// its deadline as an offset from `t0`.  Neither is recomputed once the run
/// starts.
struct event {
  operation op;
  /// One byte per bit (0x00/0x01), into `playback_file::syndromes`.  Null
  /// unless `takes_syndromes(op)`, and also null for source-backed enqueue
  /// events (then `source` is non-null and `run()` pulls from it instead).
  const std::uint8_t *syndrome_data;
  /// Bit count for `syndrome_data`.  0 for source-backed events until resolved.
  std::uint64_t num_syndromes;
  /// Resolved from `source_id=N` by `build_events`.  Non-null exactly when the
  /// record named a source; null for inline-bit events.
  syndrome_source *source = nullptr;
  std::uint64_t offset_ns;
  /// Optional expected correction bits (one byte per observable, same layout
  /// as `sink::corrections()`), from `playback_record::corrections_*`.  Null
  /// unless the record carried expected bits.  `run()` diffs these against
  /// the sink's output into `record::correction_mismatch`.
  const std::uint8_t *corrections_data = nullptr;
  std::uint64_t corrections_count = 0;
};

/// Maps source IDs (from `source_id=N` playback lines) to syndrome sources.
/// Passed to `build_events` so enqueue records with a source reference get
/// their `event::source` pointer resolved at load time.
using source_registry = std::map<std::int64_t, syndrome_source *>;

/// Parse a playback file.  Throws `std::runtime_error` naming the file and line
/// on any malformed record.
playback_file load_playback(const std::string &path);

/// Resolve the records belonging to `decoder_id` into a run schedule.
///
/// Timestamps become offsets from `t0`: used as-is when absolute, prefix-summed
/// when `deltas` is set.  Either way the arithmetic happens HERE, once, so the
/// run loop never derives one deadline from another and a late wake-up cannot
/// accumulate.
///
/// @param file        Parsed file.
/// @param decoder_id  Keep only records for this decoder; the rest belong to
///                    other emulator processes.
/// @param tick_ns     Wall-clock duration of one tick, in nanoseconds.  The
///                    resolved deadline for a record is `tick * tick_ns`.
/// @param deltas      Read `tick` as a gap from the previous kept record
///                    rather than as an absolute offset.
/// @param out_skipped If non-null, receives the number of records dropped
///                    because they belong to another decoder.  Silent
///                    filtering would make a mistyped `--decoder-id` look like
///                    an empty file.
///
/// @param sources     Maps `source_id=N` to a syndrome_source; the pointer is
///                    stored in `event::source`.  Pointers must outlive the
///                    returned events.
///
/// Throws if absolute timestamps go backwards, if no record matches, or if a
/// record names a `source_id` that is not in `sources`.
std::vector<event> build_events(const playback_file &file,
                                std::uint64_t decoder_id,
                                std::uint64_t tick_ns, bool deltas,
                                std::size_t *out_skipped = nullptr,
                                const source_registry &sources = {});

/// Parse a tick duration written with an explicit unit -- `500ns`, `1us`,
/// `2.5ms`, `1s` -- into nanoseconds.  `us` may also be spelled `µs`.
/// A unit is REQUIRED.
///
/// Returns false on a malformed value, an unknown unit, or an overflow.
bool parse_duration_ns(const std::string &text, std::uint64_t &out);

//===----------------------------------------------------------------------===//
// Sinks
//===----------------------------------------------------------------------===//

/// Opaque here; defined in rpc_call.h, included by playback_emulator.cpp and
/// the concrete sinks.  See the file comment above for why the split exists.
struct rpc_call;

/// Destination for scheduled payloads.  `send` is the only thing the runner
/// calls on the timing thread, and it sits directly in the critical path:
/// implementations must not allocate, log, or block on anything avoidable.
///
/// `send`/`try_get_corrections` build a generic `rpc_call` (see
/// `build_rpc_call` in rpc_call.h) and hand it to the one method a concrete
/// sink implements, `transport()`.  A sink therefore knows only how to move
/// an already-built RPC frame to and from its destination -- a ring slot, a
/// UDP datagram, or (`null_sink`) nowhere at all -- never how to build a
/// frame, decide blocking/retry policy, or unpack a correction reply.  Adding
/// an RPC to the protocol means teaching `build_rpc_call` one more case, not
/// touching every sink.
class sink {
public:
  virtual ~sink() = default;

  /// @param e   The event whose deadline has just passed.
  /// @param tag Application breadcrumb; the runner passes the event index.
  ///
  /// A no-op for `stream_until`, which has no wire RPC of its own: `run()`
  /// drives it as a client-side loop of synthesized `enqueue` events and
  /// `try_get_corrections()` polls instead of a single `send()`.
  void send(const event &e, std::uint64_t tag);

  /// Name for the run header.
  virtual const char *name() const = 0;

  /// Optional warm-up, called during the lead-in so first-call costs (lazy
  /// page mapping, plugin init, branch predictors) do not land on event 0.
  virtual void warm_up() {}

  /// Optional post-run summary, printed alongside the timing stats.  Never
  /// called on the timing thread.
  virtual void report() const {}

  /// Correction bits from every `get_corrections`, one byte per bit (0 or 1),
  /// laid out row-major as `reads x correction_width()`.  Empty for sinks that
  /// never read any.
  const std::vector<std::uint8_t> &corrections() const {
    return corrections_log_;
  }

  /// Bits per read: the decoder's declared observable count, fixed at
  /// construction.  0 when the sink reads no corrections.
  std::size_t correction_width() const { return num_observables_; }

  /// Total syndrome bits the decoder expects per shot.  0 when unknown (null /
  /// udp_server sinks have no local decoder to ask).
  virtual std::size_t syndrome_size() const { return 0; }

  /// Number of NOT_READY retries on the most-recent `get_corrections` send.
  /// Reset to 0 at the start of every send call; non-zero only when the
  /// sink retried under `wait_for_ready`.  `run()` reads this after each
  /// send and stores the value in `record::not_ready_retries`.
  std::uint32_t last_not_ready_retries() const {
    return last_not_ready_retries_;
  }

  /// Non-blocking poll used by `stream_until`: ask for corrections without
  /// retrying on NOT_READY.  A consuming `get_corrections` call, so a
  /// successful poll (bits appended to `corrections()`) needs no follow-up
  /// read.  Returns false on NOT_READY.
  bool try_get_corrections();

  /// Set the NOT_READY retry policy for this sink.
  void set_wait_for_ready(bool v) { wait_for_ready_ = v; }

protected:
  /// @param decoder_id      Wire routing key baked into every rpc_call this
  ///                        sink builds.
  /// @param num_observables Decoder's declared observable count; drives
  ///                        `correction_width()` and the size of the reply
  ///                        scratch buffer passed to `transport()`.
  sink(std::uint64_t decoder_id, std::uint64_t num_observables);

  /// Update the observable count (and reply scratch buffer size) after
  /// construction, for a sink that only learns it once it has parsed its own
  /// config and so cannot pass it to the base constructor's member-init list.
  void set_num_observables(std::uint64_t n);

  /// The ONLY method a concrete sink implements: perform one RPC round trip
  /// already fully described by `call`.
  ///
  ///  - If `!call.blocking`: publish the frame and return; status is ignored
  ///    (fire-and-forget, e.g. enqueue).
  ///  - If `call.blocking`: must not return until a status is known (0 == OK;
  ///    `wire::RpcStatus::NOT_READY`; or another `wire::RpcStatus`).  Must NOT
  ///    throw on NOT_READY -- that is the caller's retry decision.
  ///  - If `call.blocking && status == 0 && call.expected_result_bits > 0`:
  ///    write the reply's bit-packed result body into `result_buf` (sized
  ///    from `num_observables` above).
  virtual std::int32_t transport(const rpc_call &call,
                                 std::uint8_t *result_buf) = 0;

  bool wait_for_ready_ = false;
  std::uint64_t decoder_id_;
  std::uint64_t num_observables_;

private:
  std::int32_t dispatch_once(const rpc_call &call);
  std::int32_t dispatch_with_retry(const rpc_call &call);
  void unpack_and_log_corrections(const std::uint8_t *packed,
                                  std::size_t n_bits);

  std::vector<std::uint8_t> corrections_log_;
  std::vector<std::uint8_t> result_scratch_;
  std::uint32_t last_not_ready_retries_ = 0;
  std::uint64_t next_request_id_ = 1;
};

/// Discards everything.  Running the same schedule against this first gives
/// the emulator's jitter floor.  Still builds and serializes the generic RPC
/// object for every event, same as a real sink, so the jitter floor reflects
/// that cost too; it just never delivers the frame anywhere.
class null_sink : public sink {
public:
  null_sink() : sink(/*decoder_id=*/0, /*num_observables=*/0) {}
  const char *name() const override { return "null"; }

  /// Kept so the compiler cannot optimize the payload read away.
  std::uint64_t checksum() const { return checksum_; }

protected:
  std::int32_t transport(const rpc_call &call,
                         std::uint8_t *result_buf) override;

private:
  std::uint64_t checksum_ = 0;
  std::vector<std::uint8_t> frame_scratch_;
};

//===----------------------------------------------------------------------===//
// Running and reporting
//===----------------------------------------------------------------------===//

/// Per-event timing and outcome, captured into preallocated storage on the
/// timing thread and formatted only after the run.  Timestamps are relative
/// to `t0`.
struct record {
  std::uint64_t deadline_ns; ///< Where the event was supposed to go out.
  std::uint64_t call_ns;     ///< Sink call start.  Lateness = call - deadline.
  std::uint64_t return_ns;   ///< Sink call end.  Latency = return - call.
  /// NOT_READY retries for this event.  0 unless the sink retried a read.
  std::uint32_t not_ready_retries = 0;
  /// True when this read carried expected correction bits that differed from
  /// (or mismatched the width of) the actual read.
  bool correction_mismatch = false;
  /// Syndrome rounds fed by a `stream_until` event.  0 for other operations.
  std::uint64_t syndromes_streamed = 0;
  /// True when this event actually took a correction off the decoder: always
  /// for `get_corrections`; for `stream_until` only if the source didn't run
  /// dry first (in which case no correction was produced or consumed).
  bool read_completed = false;
};

/// Apply the requested real-time config.  Returns one human-readable warning
/// per setting it could not apply.
std::vector<std::string> apply_rt_config(const run_config &cfg);

/// Sample how far past a requested wake-up `clock_nanosleep` actually returns,
/// so `spin_slack_ns` can be derived from the machine.
/// Returns the largest overshoot over `samples` trials, in nanoseconds.
std::uint64_t measure_wakeup_overshoot(int samples = 200);

/// Replay `events` into `dst`.  Returns one record per event, in order.
std::vector<record> run(const std::vector<event> &events, sink &dst,
                        const run_config &cfg);

/// Lateness (`call - deadline`) and sink latency (`return - call`) are
/// the two distributions a run produces.  Both come back ascending-sorted,
/// one entry per record, ready to hand to `quantile()`.
std::vector<std::uint64_t> lateness_ns(const std::vector<record> &records);
std::vector<std::uint64_t> latency_ns(const std::vector<record> &records);

/// Value at quantile `q` of an ascending-sorted distribution: 0.5 is the
/// median, 1.0 the max.  
std::uint64_t quantile(const std::vector<std::uint64_t> &sorted, double q);

/// Events whose predecessor was still in the sink when their deadline passed
/// -- i.e. where the schedule shifted.  Needs `records` in play order.
std::size_t count_overruns(const std::vector<record> &records);

/// Print the run summary.  `quantiles` selects the columns, as fractions;
/// 1.0 prints as `max`.
void print_stats(const std::vector<record> &records,
                 const std::vector<double> &quantiles);

/// Dump the raw per-event records for offline analysis.
///
/// Each row carries the event index, operation name, timing columns, and a
/// `correction` column.  For `get_corrections` events the correction is the
/// actual output from the sink, encoded as a run of `0`/`1` characters (one
/// per observable, same order as `sink::corrections()`).  All other rows
/// carry `-` in that column.
void write_csv(const std::string &path,
               const std::vector<record> &records,
               const std::vector<event> &events,
               const std::vector<std::uint8_t> &corrections,
               std::size_t correction_width);

} // namespace cudaq::qec::emulator

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
/// This header is deliberately free of CUDA-Q dependencies so the timing core
/// can be built and characterized (against `null_sink`) on any machine.

#include <cstdint>
#include <map>
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
  /// Retry policy for `get_corrections` when the decoder answers NOT_READY.
  ///
  /// The decoder session has a simple state machine: `collecting` while it
  /// accumulates syndrome bits, then `result_ready` once enough have arrived
  /// and the decode has completed.  A `get_corrections` RPC that arrives while
  /// the session is still `collecting` returns NOT_READY.  This happens when
  /// the playback schedule fires `get_corrections` before the decoder has
  /// finished -- either the timing budget is too tight or the decoder itself
  /// is slow.
  ///
  /// Default (false): NOT_READY is treated as an error.  The sink increments
  /// `reads_not_ready()` and throws, aborting the run.  This is the right
  /// choice for a correctly sized schedule: it surfaces misconfiguration
  /// loudly rather than silently returning stale corrections.
  ///
  /// True: the sink spins retrying (within `kReclaimTimeoutMs`) until the
  /// session transitions to `result_ready`.  The timing thread blocks past
  /// its original deadline, so latency and overrun stats reflect the extra
  /// wait.  Use this to separate "decoder is inherently too slow for this
  /// schedule" from "the schedule is slightly too tight but the decoder
  /// eventually catches up".  `reads_not_ready()` counts each retry attempt,
  /// so you can see how many times the decoder was polled before it was ready.
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
//   tick        Unsigned integer TICK INDEX, counted from the start of the run
//               (t0).  A tick is a fixed wall-clock duration supplied
//               separately (`--tick=1us`, or `tick_ns` to `build_events`), so a
//               file describes a cadence in the units the hardware actually
//               runs at, and the same file can be replayed faster or slower by
//               changing one parameter.
//
//               Absolute by default -- tick 7 is the 7th tick boundary, not
//               "7 after the last one" -- and must be non-decreasing.  Pass
//               `deltas = true` to `build_events` to read the column as gaps in
//               ticks instead, which is often easier to write by hand.
//   operation   One of the three below.  An unknown operation is a parse error,
//               never a silent skip.
//   decoder_id  Which syndrome source this record belongs to.  ONE file can
//               describe a whole multi-source experiment; each emulator
//               process selects its own decoder's records and ignores the rest
//               (see `build_events`).
//
// Operations and their operands:
//
//   enqueue <bits>
//       Push one round of syndrome bits.  `bits` is a run of `0`/`1`
//       characters, one per syndrome bit, no separators.  Fire-and-forget on
//       the wire: the dispatcher ACKs but returns no body.
//
//   get_corrections [expected_bits]
//       Read back the accumulated corrections.  Always returns the decoder's
//       declared observable count -- that is the only width it accepts -- and
//       never resets.  Follow it with an explicit `reset` when a shot boundary
//       needs the accumulator cleared, so that clearing is visible in the file
//       rather than hidden in a flag.
//
//       `expected_bits` is an OPTIONAL run of `0`/`1` characters, one per
//       observable, written immediately after the decoder_id column.  When
//       present, `count_correction_mismatches` compares each read's actual
//       corrections against those bits and reports the mismatch count.  The
//       timing loop never reads or acts on expected_bits; they are advisory
//       and evaluated only after the run completes.
//
//   reset
//       Clear the decoder's queued syndromes and zero its corrections.
//
// Blank lines are ignored and `#` starts a comment that runs to end of line.
//
// Example:
//     # tick    op               dec  operands      (with --tick=1us)
//     0         reset            0
//     1         enqueue          0    010
//     2         enqueue          0    110
//     2         enqueue          1    001
//     3         get_corrections  0                 # no expectation
//     3         get_corrections  0    10           # expected: obs0=1 obs1=0
//     4         reset            0
//
//===----------------------------------------------------------------------===//

/// Operations a record can carry, one-to-one with the three decoding RPCs.
enum class operation { enqueue, get_corrections, reset };

const char *to_string(operation op);
bool parse_operation(const std::string &text, operation &out);

/// True when `op` carries syndrome bits.
bool takes_syndromes(operation op);

/// True when `op` reads a result back, and so must wait for its response.
bool returns_corrections(operation op);

/// One parsed record.  The syndrome bits (and any expected correction bits)
/// live in the file's shared arenas rather than in the record, so a record
/// stays small and the bits stay contiguous.
struct playback_record {
  std::uint64_t tick;
  operation op;
  std::uint64_t decoder_id;
  /// Syndrome bits: index into `playback_file::bits` and byte count.
  /// Both are 0 for non-enqueue records.
  std::size_t bit_offset;
  std::uint64_t bit_count;
  /// Optional expected correction bits, parallel to the syndrome bit fields.
  /// `expected_offset` is an index into `playback_file::expected_bits`;
  /// `expected_count` is the number of bytes (one byte per observable bit,
  /// same layout as `sink::corrections()`).  Both are 0 unless
  /// `op == get_corrections` AND the playback file carried a bit string on
  /// that line.
  std::size_t expected_offset;
  std::uint64_t expected_count;
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
  std::vector<std::uint8_t> bits;          ///< Syndrome bit arena.
  std::vector<std::uint8_t> expected_bits; ///< Expected correction bit arena.
  std::vector<playback_record> records;
  std::map<std::uint64_t, playback_meta> meta; ///< Keyed by decoder_id.
};

/// A scheduled send, fully resolved at load time: a slice of the arena plus its
/// deadline as an offset from `t0`.  Neither field is recomputed or mutated
/// once the run starts.
struct event {
  operation op;
  const std::uint8_t *data; ///< Into `playback_file::bits`.  One bit per BYTE
                            ///< (0x00 / 0x01), which is the shape
                            ///< `rpc_producer::enqueue_syndromes` wants
                            ///< Null unless `takes_syndromes(op)`.
  std::uint64_t num_bits;   ///< Syndrome bits, matching `record::bit_count`.
                            ///< Equal to the byte length of `data` only because
                            ///< of the one-bit-per-byte layout -- it is a BIT
                            ///< count, and goes on the wire as such.
  std::uint64_t offset_ns;
  /// Optional expected correction bits, populated from `playback_record::
  /// expected_offset/expected_count`.  Null (and expected_count == 0) for
  /// every event except get_corrections events whose playback record carried
  /// expected bits.  One byte per observable (0x00/0x01), same layout as
  /// `sink::corrections()`.  The run loop never reads these; they are only
  /// used by `count_correction_mismatches` after the run completes.
  const std::uint8_t *expected_data = nullptr;
  std::uint64_t expected_count = 0;
};

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
/// Throws if absolute timestamps go backwards, or if no record matches.
std::vector<event> build_events(const playback_file &file,
                                std::uint64_t decoder_id,
                                std::uint64_t tick_ns, bool deltas,
                                std::size_t *out_skipped = nullptr);

/// Parse a tick duration written with an explicit unit -- `500ns`, `1us`,
/// `2.5ms`, `1s` -- into nanoseconds.  `us` may also be spelled `µs`.
/// A unit is REQUIRED.
///
/// Returns false on a malformed value, an unknown unit, or an overflow.
bool parse_duration_ns(const std::string &text, std::uint64_t &out);

//===----------------------------------------------------------------------===//
// Sinks
//===----------------------------------------------------------------------===//

/// Destination for scheduled payloads.  `send` is the only thing the runner
/// calls on the timing thread, and it sits directly in the critical path:
/// implementations must not allocate, log, or block on anything avoidable.
class sink {
public:
  virtual ~sink() = default;

  /// @param e   The event whose deadline has just passed.
  /// @param tag Application breadcrumb; the runner passes the event index.
  virtual void send(const event &e, std::uint64_t tag) = 0;

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
  virtual const std::vector<std::uint8_t> &corrections() const {
    static const std::vector<std::uint8_t> empty;
    return empty;
  }

  /// Bits per read: the decoder's declared observable count.  0 when the sink
  /// reads no corrections.
  virtual std::size_t correction_width() const { return 0; }

  /// Total syndrome bits the decoder expects per shot.  0 when unknown (null /
  /// udp_server sinks have no local decoder to ask).
  virtual std::size_t syndrome_size() const { return 0; }

  /// Per-call outcome counts for `get_corrections` across the whole run.
  ///
  /// `reads_ok()`: calls that received a well-formed correction vector.
  ///
  /// `reads_not_ready()`: calls (or retry attempts, when wait_for_ready is
  /// set) where the decoder answered NOT_READY.  Without wait_for_ready, this
  /// is at most 1 per failed event, because the first NOT_READY throws and
  /// aborts the run.  With wait_for_ready, it counts every retry, so a single
  /// get_corrections event can contribute multiple counts here before the
  /// decoder finally responds OK.
  ///
  /// `reads_error()`: calls that received any other non-zero status
  /// (INTERNAL_ERROR, BAD_REQUEST, SYNDROMES_DROPPED, ...).
  virtual std::uint64_t reads_ok() const { return 0; }
  virtual std::uint64_t reads_not_ready() const { return 0; }
  virtual std::uint64_t reads_error() const { return 0; }

  /// Set the NOT_READY retry policy for this sink.
  void set_wait_for_ready(bool v) { wait_for_ready_ = v; }

protected:
  bool wait_for_ready_ = false;
};

/// Discards everything.  Running the same schedule against this first gives
/// the emulator's jitter floor.
class null_sink : public sink {
public:
  void send(const event &e, std::uint64_t tag) override;
  const char *name() const override { return "null"; }

  /// Kept so the compiler cannot optimize the payload read away.
  std::uint64_t checksum() const { return checksum_; }

private:
  std::uint64_t checksum_ = 0;
};

//===----------------------------------------------------------------------===//
// Running and reporting
//===----------------------------------------------------------------------===//

/// Per-event timing, captured into preallocated storage on the timing thread
/// and formatted only after the run.  All three are relative to `t0`.
struct record {
  std::uint64_t deadline_ns; ///< Where the event was supposed to go out.
  std::uint64_t call_ns;   ///< When the sink call began.  Lateness is
                           ///< `call_ns - deadline_ns`: how far past its
                           ///< deadline the record actually went out.
  std::uint64_t return_ns; ///< When the sink call returned.  Service time is
                           ///< `return_ns - call_ns`: how long the realtime
                           ///< path held us.
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
void write_csv(const std::string &path, const std::vector<record> &records);

/// Compare expected correction bits (from the playback file) against the
/// corrections the sink actually produced.  Returns {reads_with_expected,
/// mismatches}: `reads_with_expected` counts get_corrections events that
/// carried expected bits; `mismatches` counts those whose actual corrections
/// differed.  A width mismatch (expected_count != correction_width) is also
/// counted as a mismatch.  
std::pair<std::size_t, std::size_t>
count_correction_mismatches(const std::vector<event> &events, const sink &dst);

} // namespace cudaq::qec::emulator

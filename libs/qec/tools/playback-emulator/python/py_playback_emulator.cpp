/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file py_playback_emulator.cpp
/// @brief Python bindings for the playback emulator.
///
/// Deliberately a small surface: one `run_playback()` that does load ->
/// resolve -> run -> summarize, plus the result object.  The timing loop must
/// stay in C++ (the whole point of the tool is that nothing between two
/// deadlines is interpreted), so there is no reason to expose the pieces
/// individually and every reason not to invite a Python-driven send loop.

#include "playback_emulator.h"
#include "session_sink.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <stdexcept>

namespace nb = nanobind;
using namespace cudaq::qec::emulator;

namespace {

/// Everything one run produces: how well the schedule was met, and what the
/// decoder said.
struct playback_result {
  std::size_t events = 0;
  std::size_t overruns = 0;
  std::size_t records_skipped = 0;
  std::size_t records_total = 0;
  std::uint64_t span_ns = 0;

  /// Ascending-sorted, one entry per event.  
  std::vector<std::uint64_t> lateness;
  std::vector<std::uint64_t> latency;

  std::uint64_t lateness_at(double q) const { return quantile(lateness, q); }
  std::uint64_t latency_at(double q) const { return quantile(latency, q); }

  std::vector<std::uint8_t> corrections;
  std::size_t correction_width = 0;
  std::vector<std::string> warnings;
};

// sentinel: spin_slack_ns == 0 means "calibrate automatically"
playback_result run_playback(const std::string &playback,
                            const std::string &sink_name,
                            const std::string &config, std::uint64_t decoder_id,
                            std::uint64_t spin_slack_ns,
                            std::uint64_t lead_in_ns,
                            const std::string &tick, bool deltas, int pin_cpu,
                            const std::string &server_host,
                            int server_port, int server_slots,
                            int server_slot_size, int server_observables,
                            const std::string &dem_file) {
  run_config cfg;
  cfg.lead_in_ns = lead_in_ns;
  cfg.pin_cpu = pin_cpu;

  playback_result out;

  // Parse and resolve before touching the clock, exactly as the CLI does.
  const playback_file file = load_playback(playback);
  std::uint64_t tick_ns = 0;
  if (!parse_duration_ns(tick, tick_ns))
    throw std::invalid_argument(
        "run_playback: could not parse tick='" + tick +
        "'; give a number and a unit, e.g. '500ns', '1us', '2.5ms', '1s'");
  const auto events = build_events(file, decoder_id, tick_ns, deltas,
                                   &out.records_skipped);
  out.records_total = file.records.size();
  out.span_ns = events.empty() ? 0 : events.back().offset_ns;

  out.warnings = apply_rt_config(cfg);
  // Calibrate unless the caller supplied an explicit value (spin_slack_ns > 0).
  cfg.spin_slack_ns = spin_slack_ns ? spin_slack_ns
                                    : 2 * measure_wakeup_overshoot();

  std::unique_ptr<sink> dst;
  if (sink_name == "null")
    dst = std::make_unique<null_sink>();
  else if (sink_name == "inproc_rpc")
    dst = make_inproc_rpc_sink(config, decoder_id, dem_file);
  else if (sink_name == "ring_buffer_injector")
    dst = make_ring_buffer_injector_sink(config, decoder_id, dem_file);
  else if (sink_name == "udp_server")
    dst = make_udp_server_sink(server_host,
                               static_cast<std::uint16_t>(server_port),
                               decoder_id,
                               static_cast<std::uint32_t>(server_slots),
                               static_cast<std::uint32_t>(server_slot_size),
                               static_cast<std::uint64_t>(server_observables));
  else
    throw std::invalid_argument("run_playback: unknown sink '" + sink_name +
                                "'; expected null, inproc_rpc, ring_buffer_injector or udp_server");

  const auto meta_it = file.meta.find(decoder_id);
  if (meta_it != file.meta.end() && meta_it->second.syndrome_size &&
      dst->syndrome_size() && dst->syndrome_size() != meta_it->second.syndrome_size)
    out.warnings.push_back(
        "playback expects syndrome_size=" +
        std::to_string(meta_it->second.syndrome_size) +
        " but decoder has syndrome_size=" +
        std::to_string(dst->syndrome_size()) +
        " -- corrections will likely be wrong");

  std::vector<record> records;
  {
    // The timing thread must not contend with the interpreter for the GIL --
    // a Python thread waking mid-run would show up directly as lateness.
    nb::gil_scoped_release release;
    records = run(events, *dst, cfg);
  }

  out.events = records.size();
  out.overruns = count_overruns(records);
  out.lateness = lateness_ns(records);
  out.latency = latency_ns(records);
  out.corrections = dst->corrections();
  out.correction_width = dst->correction_width();
  return out;
}

} // namespace

NB_MODULE(qec_playback_emulator, m) {
  m.doc() = "Schedule-accurate playback of recorded syndromes into the "
            "CUDA-Q QEC realtime decoding API.";

  nb::class_<playback_result>(m, "PlaybackResult")
      .def_ro("events", &playback_result::events,
              "Records played for the selected decoder.")
      .def_ro("records_total", &playback_result::records_total)
      .def_ro("records_skipped", &playback_result::records_skipped,
              "Records belonging to other decoders.")
      .def_ro("span_ns", &playback_result::span_ns)
      .def_ro("overruns", &playback_result::overruns,
              "Events whose predecessor was still in the sink at their "
              "deadline.")
      .def("lateness", &playback_result::lateness_at, nb::arg("q"),
           "Lateness in ns at quantile q (0.5 = median, 1.0 = max).")
      .def("latency", &playback_result::latency_at, nb::arg("q"),
           "Sink latency in ns at quantile q (0.5 = median, 1.0 = max).")
      .def_ro("lateness_ns", &playback_result::lateness,
              "The full sorted lateness distribution, for histograms or "
              "quantiles numpy can compute better than we can.")
      .def_ro("latency_ns", &playback_result::latency,
              "The full sorted latency distribution.")
      .def_ro("corrections", &playback_result::corrections,
              "Correction bits, one int per bit, laid out row-major as "
              "reads x correction_width.  Reshape with numpy for a 2-D view; "
              "for a single-observable decoder it is already one bit per read.")
      .def_ro("correction_width", &playback_result::correction_width,
              "Bits per read -- the decoder's observable count.")
      .def_ro("warnings", &playback_result::warnings,
              "Real-time knobs that could not be applied.");

  m.def("run_playback", &run_playback, nb::arg("playback"),
        nb::arg("sink") = "null", nb::arg("config") = "",
        nb::arg("decoder_id") = 0, nb::arg("spin_slack_ns") = 0,
        nb::arg("lead_in_ns") = 50000000, nb::arg("tick") = "1us",
        nb::arg("deltas") = false,
        nb::arg("pin_cpu") = -1,
        nb::arg("server_host") = "127.0.0.1", nb::arg("server_port") = 0,
        nb::arg("server_slots") = 8, nb::arg("server_slot_size") = 256,
        nb::arg("server_observables") = 1, nb::arg("dem_file") = "",
        R"doc(Play a playback file and return timing stats plus corrections.

sink: "null" (jitter floor, no decoder), "inproc_rpc" (one ACK-waited RPC per
record, ring depth 1), "ring_buffer_injector" (no ACK wait, full ring depth), or
"udp_server" (frames shipped to a REMOTE decoding_server; needs server_port
rather than config).  The two in-process sinks need `config`, a multi-decoder
YAML.

spin_slack_ns: the tail of each inter-event gap the runner spins rather than
sleeps, in nanoseconds.  0 (default) measures the machine's worst-case
clock_nanosleep overshoot and sets slack = 2x that automatically.  Pass an
explicit value to override.

One source per process: rpc_producer enforces single-producer with a
process-global flag, so do not call this concurrently from threads.)doc");

  m.def("measure_wakeup_overshoot", &measure_wakeup_overshoot,
        nb::arg("samples") = 200,
        "Worst observed clock_nanosleep overshoot in ns, for sizing "
        "spin_slack_ns on this machine.");
}

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
/// Single entry point: run_playback().  The playback file specifies the timing
/// schedule; enqueue records carry either inline syndrome bits or `source_id=N`
/// references.  When sources are provided, syndrome data is pre-generated from
/// those sources (while the GIL is held) before the C++ timing run begins so
/// the hot loop never calls back into Python.

#include "playback_emulator.h"
#include "syndrome_source.h"
#include "sinks.h"

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <stdexcept>

namespace nb = nanobind;
using namespace cudaq::qec::emulator;

namespace {

struct playback_result {
  std::size_t events = 0;
  std::size_t overruns = 0;
  std::size_t records_skipped = 0;
  std::size_t records_total = 0;
  std::uint64_t span_ns = 0;

  std::vector<std::uint64_t> lateness;
  std::vector<std::uint64_t> latency;

  std::uint64_t lateness_at(double q) const { return quantile(lateness, q); }
  std::uint64_t latency_at(double q) const { return quantile(latency, q); }

  std::vector<std::uint8_t> corrections;
  std::size_t correction_width = 0;
  std::vector<std::string> warnings;

  std::size_t reads_with_expected = 0;
  std::size_t correction_mismatches = 0;
  std::uint64_t not_ready_retries = 0;
  std::uint64_t syndromes_streamed = 0;
};

static std::unique_ptr<sink>
make_sink(const std::string &sink_name, std::uint64_t decoder_id,
          const std::string &config, const std::string &dem_file,
          const std::string &server_host, int server_port, int server_slots,
          int server_slot_size, int server_observables) {
  if (sink_name == "null")
    return std::make_unique<null_sink>();
  if (sink_name == "ring_buffer_injector")
    return make_ring_buffer_injector_sink(config, decoder_id, dem_file);
  if (sink_name == "udp_server")
    return make_udp_server_sink(server_host,
                                static_cast<std::uint16_t>(server_port),
                                decoder_id,
                                static_cast<std::uint32_t>(server_slot_size),
                                static_cast<std::uint64_t>(server_observables));
  if (sink_name == "udp_ring")
    return make_udp_ring_sink(server_host,
                              static_cast<std::uint16_t>(server_port),
                              decoder_id,
                              static_cast<std::uint32_t>(server_slots),
                              static_cast<std::uint32_t>(server_slot_size),
                              static_cast<std::uint64_t>(server_observables));
  throw std::invalid_argument("unknown sink '" + sink_name +
                              "'; expected null, ring_buffer_injector, "
                              "udp_server, or udp_ring");
}

static playback_result
collect_result(const std::vector<record> &records,
               const std::vector<event> &events, sink &dst,
               playback_result &&out, const std::string &csv_file) {
  out.events = records.size();
  out.overruns = count_overruns(records);
  out.lateness = lateness_ns(records);
  out.latency = latency_ns(records);
  out.corrections = dst.corrections();
  out.correction_width = dst.correction_width();
  for (const auto &r : records) {
    out.not_ready_retries += r.not_ready_retries;
    out.syndromes_streamed += r.syndromes_streamed;
    if (r.correction_mismatch)
      ++out.correction_mismatches;
  }
  for (const auto &e : events)
    if ((e.op == operation::get_corrections ||
         e.op == operation::stream_until) &&
        e.corrections_data != nullptr)
      ++out.reads_with_expected;
  if (!csv_file.empty())
    write_csv(csv_file, records, events, dst.corrections(),
              dst.correction_width());
  return std::move(out);
}

/// Load a playback file, resolve its syndrome sources, run, report.
///
/// `sources` maps the integers used in `source_id=N` playback lines to
/// SyndromeSource instances.  `enqueue` events draw their single round here,
/// while the GIL is held, so the timing loop sees only static bytes;
/// `stream_until` keeps its source and pulls rounds inside run().
playback_result run_playback(const std::string &playback,
                             nb::dict sources,
                             const std::string &sink_name,
                             const std::string &config,
                             std::uint64_t decoder_id,
                             std::uint64_t spin_slack_ns,
                             std::uint64_t lead_in_ns,
                             const std::string &tick, bool deltas, int pin_cpu,
                             const std::string &server_host, int server_port,
                             int server_slots, int server_slot_size,
                             int server_observables,
                             const std::string &dem_file, bool wait_for_ready,
                             const std::string &csv_file) {
  run_config cfg;
  cfg.lead_in_ns = lead_in_ns;
  cfg.pin_cpu = pin_cpu;
  cfg.wait_for_ready = wait_for_ready;

  playback_result out;

  source_registry registry;
  for (auto [key, value] : sources)
    registry[nb::cast<std::int64_t>(key)] = &nb::cast<syndrome_source &>(value);

  const playback_file file = load_playback(playback);
  std::uint64_t tick_ns = 0;
  if (!parse_duration_ns(tick, tick_ns))
    throw std::invalid_argument("run_playback: bad tick='" + tick + "'");

  auto events = build_events(file, decoder_id, tick_ns, deltas,
                             &out.records_skipped, registry);
  out.records_total = file.records.size();
  out.span_ns = events.empty() ? 0 : events.back().offset_ns;

  // Draw each source-backed `enqueue`'s single round now, while the GIL is
  // held, so the timing loop never calls a (possibly Python) source.  Rounds
  // live in a vector-of-vectors: each inner buffer keeps its address even if
  // the outer vector reallocates.  `stream_until` keeps its source pointer --
  // its round count is not known until the decoder answers.
  std::vector<std::vector<std::uint8_t>> arena;
  arena.reserve(events.size());
  for (auto &e : events) {
    if (e.op != operation::enqueue || e.source == nullptr)
      continue;
    arena.push_back(e.source->next_round());
    e.syndrome_data = arena.back().data();
    e.num_syndromes = arena.back().size();
    e.source = nullptr; // resolved: run() takes the static path
  }

  out.warnings = apply_rt_config(cfg);
  cfg.spin_slack_ns =
      spin_slack_ns ? spin_slack_ns : 2 * measure_wakeup_overshoot();

  auto dst = make_sink(sink_name, decoder_id, config, dem_file, server_host,
                       server_port, server_slots, server_slot_size,
                       server_observables);

  const auto meta_it = file.meta.find(decoder_id);
  if (meta_it != file.meta.end() && meta_it->second.syndrome_size &&
      dst->syndrome_size() &&
      dst->syndrome_size() != meta_it->second.syndrome_size)
    out.warnings.push_back(
        "playback expects syndrome_size=" +
        std::to_string(meta_it->second.syndrome_size) +
        " but decoder has syndrome_size=" +
        std::to_string(dst->syndrome_size()) +
        " -- corrections will likely be wrong");

  std::vector<record> records;
  {
    nb::gil_scoped_release release;
    records = run(events, *dst, cfg);
  }
  return collect_result(records, events, *dst, std::move(out), csv_file);
}

// Trampoline: lets Python subclass syndrome_source.
struct PySyndromeSource : public syndrome_source {
  NB_TRAMPOLINE(syndrome_source, 4);

  std::vector<std::uint8_t> next_round() override {
    NB_OVERRIDE_PURE(next_round);
  }
  std::size_t syndrome_size() const override {
    NB_OVERRIDE_PURE(syndrome_size);
  }
  std::size_t enqueues_per_shot() const override {
    NB_OVERRIDE(enqueues_per_shot);
  }
  void set_rounds(std::size_t r) override { NB_OVERRIDE(set_rounds, r); }
};

static std::string circuit_to_text(nb::object circuit_or_str) {
  if (nb::isinstance<nb::str>(circuit_or_str))
    return nb::cast<std::string>(circuit_or_str);
  return nb::cast<std::string>(nb::str(circuit_or_str));
}

} // namespace

NB_MODULE(qec_playback_emulator, m) {
  m.doc() = "Schedule-accurate playback of recorded syndromes into the "
            "CUDA-Q QEC realtime decoding API.";

  nb::class_<playback_result>(m, "PlaybackResult")
      .def_ro("events", &playback_result::events)
      .def_ro("records_total", &playback_result::records_total)
      .def_ro("records_skipped", &playback_result::records_skipped)
      .def_ro("span_ns", &playback_result::span_ns)
      .def_ro("overruns", &playback_result::overruns)
      .def("lateness", &playback_result::lateness_at, nb::arg("q"),
           "Lateness in ns at quantile q (0.5 = median, 1.0 = max).")
      .def("latency", &playback_result::latency_at, nb::arg("q"))
      .def_ro("lateness_ns", &playback_result::lateness)
      .def_ro("latency_ns", &playback_result::latency)
      .def_ro("corrections", &playback_result::corrections)
      .def_ro("correction_width", &playback_result::correction_width)
      .def_ro("warnings", &playback_result::warnings)
      .def_ro("reads_with_expected", &playback_result::reads_with_expected)
      .def_ro("correction_mismatches", &playback_result::correction_mismatches)
      .def_ro("not_ready_retries", &playback_result::not_ready_retries)
      .def_ro("syndromes_streamed", &playback_result::syndromes_streamed,
              "Total syndrome rounds fed by stream_until events.");

  // --- Syndrome sources ---

  nb::class_<syndrome_source, PySyndromeSource>(m, "SyndromeSource",
      R"doc(Abstract base for syndrome sources.

Subclass and implement next_round() and syndrome_size().  Optionally override
enqueues_per_shot() and set_rounds().  Pass instances as values in the
`sources` dict of run_playback(); the playback file references them by ID
with `source_id=N` on enqueue lines.
)doc")
      .def(nb::init<>())
      .def("next_round", &syndrome_source::next_round)
      .def("syndrome_size", &syndrome_source::syndrome_size)
      .def("enqueues_per_shot", &syndrome_source::enqueues_per_shot)
      .def("set_rounds", &syndrome_source::set_rounds, nb::arg("r"));

  nb::class_<static_syndrome_source, syndrome_source>(m,
      "StaticSyndromeSource",
      R"doc(Syndrome source backed by pre-sourced bits arranged into rounds.

rounds: list of round arrays, each a list of 0/1 ints.  Rounds may have
        different lengths (e.g. syndrome rounds vs. data-qubit round).
)doc")
      .def(nb::init<std::vector<std::vector<std::uint8_t>>>(),
           nb::arg("rounds"))
      .def("next_round", &static_syndrome_source::next_round)
      .def("syndrome_size", &static_syndrome_source::syndrome_size);

  nb::class_<stim_circuit_source, syndrome_source>(m, "StimCircuitSource",
      R"doc(Syndrome source backed by stim circuit sampling.

Each next_round() returns one shot's full measurement record (all rounds
concatenated).  enqueues_per_shot() is 1.

Args:
    circuit: stim.Circuit or circuit text string.
    shots:   Shots to pre-sample (default 1000).
    seed:    RNG seed (default 42).
)doc")
      .def("__init__",
           [](stim_circuit_source *self, nb::object c, std::size_t shots,
              std::uint64_t seed) {
             new (self) stim_circuit_source(circuit_to_text(c), shots, seed);
           },
           nb::arg("circuit"), nb::arg("shots") = 1000, nb::arg("seed") = 42)
      .def("next_round", &stim_circuit_source::next_round)
      .def("syndrome_size", &stim_circuit_source::syndrome_size);

  nb::class_<stim_memory_source, syndrome_source>(m, "StimMemorySource",
      R"doc(Syndrome source for stim memory circuits.

Streams r rounds of syndrome extraction followed by one data-qubit readout
round per shot.  set_rounds(r) reconfigures r at runtime.
enqueues_per_shot() returns r + (data groups).

Args:
    circuit: stim.Circuit or circuit text string.
    shots:   Shots to pre-sample (default 1000).
    seed:    RNG seed (default 42).
)doc")
      .def("__init__",
           [](stim_memory_source *self, nb::object c, std::size_t shots,
              std::uint64_t seed) {
             new (self) stim_memory_source(circuit_to_text(c), shots, seed);
           },
           nb::arg("circuit"), nb::arg("shots") = 1000, nb::arg("seed") = 42)
      .def("set_rounds", &stim_memory_source::set_rounds, nb::arg("r"))
      .def("next_round", &stim_memory_source::next_round)
      .def("syndrome_size", &stim_memory_source::syndrome_size)
      .def("enqueues_per_shot", &stim_memory_source::enqueues_per_shot);

  // --- Entry point ---

  m.def("run_playback", &run_playback, nb::arg("playback"),
        nb::arg("sources") = nb::dict(),
        nb::arg("sink") = "null", nb::arg("config") = "",
        nb::arg("decoder_id") = 0, nb::arg("spin_slack_ns") = 0,
        nb::arg("lead_in_ns") = 50000000, nb::arg("tick") = "1us",
        nb::arg("deltas") = false, nb::arg("pin_cpu") = -1,
        nb::arg("server_host") = "127.0.0.1", nb::arg("server_port") = 0,
        nb::arg("server_slots") = 8, nb::arg("server_slot_size") = 256,
        nb::arg("server_observables") = 1, nb::arg("dem_file") = "",
        nb::arg("wait_for_ready") = false, nb::arg("csv") = "",
        R"doc(Load a playback file and replay it into the decoder.

playback: path to a playback file.  `enqueue` lines carry either inline
          syndrome bits or `source_id=N`; `stream_until` always takes
          `source_id=N` and pulls rounds until the decoder is ready, absorbing
          the correction read.

sources: dict mapping the integers used in `source_id=N` to SyndromeSource
         instances (StaticSyndromeSource, StimCircuitSource, StimMemorySource,
         or your own Python subclass).  An unregistered id is an error.

sink: "null", "ring_buffer_injector", "udp_server", or "udp_ring".
tick: tick duration, e.g. "100us".  Must match the tick column in the file.
)doc");

  m.def("measure_wakeup_overshoot", &measure_wakeup_overshoot,
        nb::arg("samples") = 200,
        "Worst observed clock_nanosleep overshoot in ns.");
}

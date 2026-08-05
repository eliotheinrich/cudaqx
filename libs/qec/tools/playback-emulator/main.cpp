/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file main.cpp
/// @brief `qec_playback_emulator`: replays recorded syndrome bits into the
/// realtime decoding API on a user-supplied schedule.
///
/// Usage:
///   qec_playback_emulator --playback=<file> [--sink=null|inproc_rpc|ring_buffer_injector]
///                         [--config=<decoders.yaml>] [--decoder-id=N]
///                         [--delta-timestamps]
///                         [--spin-slack-ns=N]
///                         [--lead-in-ns=N] [--cpu=N]
///                         [--csv=<out.csv>]
///
/// `--playback` is a playback file: one `<timestamp> <operation> <decoder_id>
/// [operands...]` record per line.  See playback_emulator.h for the format.
///
/// `--decoder-id` selects both which decoder in `--config` to realize and which
/// records to play; records for other decoders are skipped, so ONE playback
/// file can drive a whole multi-source experiment with one process per decoder.

#include "session_sink.h"
#include "playback_emulator.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace cudaq::qec::emulator;

namespace {

struct options {
  std::string playback_file;
  std::string sink_name = "null";
  std::string config_file;
  std::string csv_file;
  std::string dem_file;
  std::size_t decoder_id = 0;
  std::string server_host = "127.0.0.1";
  int server_port = 0;
  int server_slots = 8;
  int server_slot_size = 256;
  int server_observables = 1;
  bool deltas = false;
  std::string tick = "";
  std::uint64_t tick_ns = 0;
  bool spin_slack_explicit = false;
  /// Columns for the summary, as fractions.  1.0 is the max.
  std::vector<double> quantiles = {0.5, 0.9, 0.99, 0.999, 1.0};
  run_config run;
};

[[noreturn]] void usage(int code) {
  std::fprintf(
      code ? stderr : stdout,
      "usage: qec_playback_emulator --playback=<file>\n"
      "                           [--sink=null|inproc_rpc|ring_buffer_injector|\n"
      "                                   udp_server]\n"
      "                           [--server-host=H] [--server-port=N]\n"
      "                           [--server-slots=N] [--server-slot-size=N]\n"
      "                           [--server-observables=N]\n"
      "                           [--config=<decoders.yaml>] [--decoder-id=N]\n"
      "                           --tick=<dur>  (e.g. 1us; ns|us|ms|s)\n"
      "                           [--delta-ticks]\n"
      "                           [--spin-slack-ns=N]\n"
      "                           [--lead-in-ns=N] [--cpu=N]\n"
      "                           [--csv=<out.csv>]\n"
      "                           [--percentiles=50,90,99,99.9,100]\n"
      "                           [--dem=<model.dem>]  (chromobius)\n");
  std::exit(code);
}

/// Returns true and fills `value` if `arg` is `--<key>=<value>`.
bool match(const std::string &arg, const char *key, std::string &value) {
  const std::string prefix = std::string("--") + key + "=";
  if (arg.rfind(prefix, 0) != 0)
    return false;
  value = arg.substr(prefix.size());
  return true;
}

options parse_args(int argc, char **argv) {
  options opts;
  std::string value;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
      usage(0);
    else if (match(arg, "playback", value))
      opts.playback_file = value;
    else if (match(arg, "sink", value))
      opts.sink_name = value;
    else if (match(arg, "config", value))
      opts.config_file = value;
    else if (match(arg, "csv", value))
      opts.csv_file = value;
    else if (match(arg, "dem", value))
      opts.dem_file = value;
    else if (match(arg, "decoder-id", value))
      opts.decoder_id = std::stoull(value);
    else if (match(arg, "server-host", value))
      opts.server_host = value;
    else if (match(arg, "server-port", value))
      opts.server_port = std::stoi(value);
    else if (match(arg, "server-slots", value))
      opts.server_slots = std::stoi(value);
    else if (match(arg, "server-slot-size", value))
      opts.server_slot_size = std::stoi(value);
    else if (match(arg, "server-observables", value))
      opts.server_observables = std::stoi(value);
    else if (match(arg, "spin-slack-ns", value)) {
      opts.run.spin_slack_ns = std::stoull(value);
      opts.spin_slack_explicit = true;
    } else if (match(arg, "lead-in-ns", value))
      opts.run.lead_in_ns = std::stoull(value);
    else if (match(arg, "cpu", value))
      opts.run.pin_cpu = std::stoi(value);
    else if (match(arg, "percentiles", value)) {
      opts.quantiles.clear();
      std::istringstream fields(value);
      std::string field;
      while (std::getline(fields, field, ','))
        if (!field.empty())
          opts.quantiles.push_back(std::stod(field) / 100.0);
      if (opts.quantiles.empty()) {
        std::fprintf(stderr, "--percentiles needs at least one value, e.g. "
                             "--percentiles=50,99,100\n");
        usage(1);
      }
    }
    else if (match(arg, "tick", value))
      opts.tick = value;
    else if (arg == "--delta-ticks" || arg == "--delta-timestamps")
      opts.deltas = true;
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      usage(1);
    }
  }
  if (opts.playback_file.empty()) {
    std::fprintf(stderr, "--playback is required\n");
    usage(1);
  }
  if (opts.tick.empty()) {
    std::fprintf(stderr, "--tick=<duration> is required, e.g. --tick=1us "
                         "(units: ns, us, ms, s)\n");
    usage(1);
  }
  if (!parse_duration_ns(opts.tick, opts.tick_ns)) {
    std::fprintf(stderr, "could not parse --tick=%s; give a number and a unit, "
                         "e.g. 500ns, 1us, 2.5ms, 1s\n", opts.tick.c_str());
    usage(1);
  }
  if (opts.sink_name == "udp_server" && opts.server_port == 0) {
    std::fprintf(stderr, "--sink=udp_server requires --server-port=<N> "
                         "(the port= from the server's READY line)\n");
    usage(1);
  }
  if (opts.sink_name != "null" && opts.sink_name != "udp_server" &&
      opts.config_file.empty()) {
    std::fprintf(stderr, "--sink=%s requires --config=<decoders.yaml>\n",
                 opts.sink_name.c_str());
    usage(1);
  }
  return opts;
}

} // namespace

int main(int argc, char **argv) try {
  options opts = parse_args(argc, argv);

  // ---- Everything below happens before t0. -------------------------------
  const auto file = load_playback(opts.playback_file);
  std::size_t skipped = 0;
  const auto events = build_events(file, opts.decoder_id, opts.tick_ns,
                                   opts.deltas, &skipped);

  for (const auto &warning : apply_rt_config(opts.run))
    std::fprintf(stderr, "warning: %s\n", warning.c_str());

  if (!opts.spin_slack_explicit) {
    // Derive the spin window from this machine: 2x the worst observed
    // clock_nanosleep overshoot so the sleep phase reliably lands inside the
    // spin window.  Skipped when the caller passes --spin-slack-ns explicitly.
    const std::uint64_t worst = measure_wakeup_overshoot();
    opts.run.spin_slack_ns = 2 * worst;
    std::printf("spin-slack: worst wake-up overshoot %llu ns -> slack %llu ns\n",
                static_cast<unsigned long long>(worst),
                static_cast<unsigned long long>(opts.run.spin_slack_ns));
  }

  std::unique_ptr<sink> dst;
  if (opts.sink_name == "null")
    dst = std::make_unique<null_sink>();
  else if (opts.sink_name == "inproc_rpc")
    dst = make_inproc_rpc_sink(opts.config_file, opts.decoder_id, opts.dem_file);
  else if (opts.sink_name == "ring_buffer_injector")
    dst = make_ring_buffer_injector_sink(opts.config_file, opts.decoder_id,
                                       opts.dem_file);
  else if (opts.sink_name == "udp_server")
    dst = make_udp_server_sink(
        opts.server_host, static_cast<std::uint16_t>(opts.server_port),
        opts.decoder_id, static_cast<std::uint32_t>(opts.server_slots),
        static_cast<std::uint32_t>(opts.server_slot_size),
        static_cast<std::uint64_t>(opts.server_observables));
  else {
    std::fprintf(stderr, "unknown --sink: %s\n", opts.sink_name.c_str());
    return 1;
  }

  const auto meta_it = file.meta.find(opts.decoder_id);
  if (meta_it != file.meta.end() && meta_it->second.syndrome_size &&
      dst->syndrome_size() && dst->syndrome_size() != meta_it->second.syndrome_size)
    std::fprintf(stderr,
                 "warning: playback expects syndrome_size=%llu but decoder "
                 "has syndrome_size=%llu -- corrections will likely be wrong\n",
                 static_cast<unsigned long long>(meta_it->second.syndrome_size),
                 static_cast<unsigned long long>(dst->syndrome_size()));

  std::printf("playback %s: %zu records, %zu bits\n",
              opts.playback_file.c_str(), file.records.size(),
              file.bits.size());
  std::printf("tick     %s = %llu ns\n", opts.tick.c_str(),
              static_cast<unsigned long long>(opts.tick_ns));
  std::printf("decoder  %zu: %zu events spanning %llu ns (%zu records skipped "
              "for other decoders)\n",
              opts.decoder_id, events.size(),
              static_cast<unsigned long long>(events.back().offset_ns),
              skipped);
  std::printf("sink=%s slack=%llu ns lead-in=%llu ns\n", dst->name(),
              static_cast<unsigned long long>(opts.run.spin_slack_ns),
              static_cast<unsigned long long>(opts.run.lead_in_ns));

  // ---- The run. ----------------------------------------------------------
  const auto records = run(events, *dst, opts.run);

  print_stats(records, opts.quantiles);
  dst->report();
  if (!opts.csv_file.empty()) {
    write_csv(opts.csv_file, records);
    std::printf("per-event records written to %s\n", opts.csv_file.c_str());
  }
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}

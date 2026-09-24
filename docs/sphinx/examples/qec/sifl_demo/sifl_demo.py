# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
"""Stable vs. runaway real-time decoding (SIFL) on the playback emulator.

A surface-code memory experiment streams one stabilizer round every T us to
two decoders ("rings"). Shot i streams rounds to ring i % 2 until shot i-1's
result lands, then reads out the data qubits. A decode costing a + b*r for r
rounds gives a backlog r_i = (a + b*r_{i-1}) / T, which settles at a / (T - b)
only if T > b. Faster cadences run away until the stream cap.

Run with ./run_sifl_demo.sh, or build per_round_decoder.cpp (see its header)
and run:
  python3 sifl_demo.py [path/to/libper_round_decoder.so]
"""
import ctypes, os, sys, tempfile
import numpy as np
import stim
import cudaq_qec as qec

# Loading the plugin registers `per_round_decoder` with CUDA-Q QEC.
plugin = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libper_round_decoder.so")
ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)

DISTANCE, P, MAX_ROUNDS, SHOTS, PREAMBLE = 5, 0.01, 60, 30, 10
STREAM_CAP = MAX_ROUNDS - PREAMBLE


def circuit(rounds):
    return stim.Circuit.generated("surface_code:rotated_memory_z",
                                  rounds=rounds,
                                  distance=DISTANCE,
                                  before_measure_flip_probability=P,
                                  after_clifford_depolarization=P)


round_width = circuit(2).num_measurements - circuit(1).num_measurements
terminal_width = circuit(1).num_measurements - round_width


def write_dem(rounds, path):
    """Writes the full DEM of an r-round circuit as H, O, D and rates lines."""
    c = circuit(rounds)
    text = str(c.detector_error_model(decompose_errors=True))
    dem = qec.dem_from_stim_text(text, use_decomp_suggestions=True)
    dem.canonicalize_for_rounds(round_width, remove_zero_syndrome_errors=True)
    # D: flipping measurement i alone fires exactly the detectors in column i.
    flips = np.eye(c.num_measurements, dtype=np.bool_)
    D = c.compile_m2d_converter().convert(measurements=flips,
                                          append_observables=False).T
    lines = [
        qec.pcm_to_sparse_vec(np.ascontiguousarray(M, dtype=np.uint8))
        for M in (dem.detector_error_matrix, dem.observables_flips_matrix, D)
    ] + [dem.error_rates]
    with open(path, "w") as f:
        f.write("\n".join(" ".join(map(str, line)) for line in lines) + "\n")


def ring(decoder_id, dem_dir):
    config = qec.decoder_config()
    config.id, config.type = decoder_id, "per_round_decoder"
    # Placeholders: the decoder builds its sub-decoders from dem_dir. The large
    # D_sparse index lets a shot carry any number of measurements.
    config.block_size, config.syndrome_size = 1, 1
    config.H_sparse, config.O_sparse = [0, -1], [0, -1]
    config.D_sparse = [9_999_999, -1]
    config.decoder_custom_args = dict(dem_dir=dem_dir,
                                      round_width=round_width,
                                      terminal_width=terminal_width,
                                      max_rounds=MAX_ROUNDS)
    return config


# Shot i streams to ring i % 2 until shot i-1's correction lands.
lines = [
    f"0 stream source=0 rounds={PREAMBLE}", "- enqueue_data source=0",
    "- get_corrections return_size=1 signal=shot0"
]
for i in range(1, SHOTS):
    lines += [
        f"- stream session={i % 2} source=0 every=1 min_rounds=1 "
        f"max_rounds={STREAM_CAP} until=shot{i - 1}",
        f"- enqueue_data session={i % 2} source=0",
        f"- get_corrections session={i % 2} return_size=1 signal=shot{i}"
    ]
schedule = "\n".join(lines) + "\n"
source = dict(type="stim_memory",
              seed=1,
              code="surface_code",
              task="rotated_memory_z",
              distance=DISTANCE,
              rounds=10_000,
              before_measure_flip_probability=P,
              after_clifford_depolarization=P)

with tempfile.TemporaryDirectory() as dem_dir:
    print(f"Building {MAX_ROUNDS} DEMs for distance {DISTANCE}...")
    for r in range(1, MAX_ROUNDS + 1):
        write_dem(r, f"{dem_dir}/r{r}.txt")
    decoders = qec.multi_decoder_config()
    decoders.decoders = [ring(0, dem_dir), ring(1, dem_dir)]

    pb = qec.playback
    print(f"Rounds streamed per shot (stream cap {STREAM_CAP}):")
    for period_us in (2, 5, 10, 20, 50):
        result = pb.run(schedule,
                        tick_ns=period_us * 1000,
                        decoders=decoders,
                        sources={0: source})
        rounds = [
            r.rounds_streamed
            for r in result.records
            if r.op == pb.operation.stream
        ]
        assert all(r.read_completed
                   for r in result.records
                   if r.op == pb.operation.get_corrections)
        print(f"  T = {period_us:>2} us  {rounds}")
# [End Documentation]

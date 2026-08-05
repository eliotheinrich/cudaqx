# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Sample a d=5 superdense color-code memory circuit and write the three files
# run_playback.py needs: playback.txt, decoder.yaml, dem.txt, and truth.npy.
#
# Env: ISING_DECODING — clone of NVIDIA/Ising-Decoding

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.environ.get("ISING_DECODING", "/workspaces/ising-decoding"),
                 "code")
)
from qec.color_code.memory_circuit import MemoryCircuit

sys.path.insert(
    0, 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
from playback_from_circuit import write_playback_from_circuit


import numpy as np
import cudaq_qec as qec

HERE = os.path.dirname(os.path.abspath(__file__))

DISTANCE   = 5
ROUNDS     = 5
ERROR_RATE = 3e-3
SHOTS      = 200
SEED       = 42


# From Ising-Decoding
memory = MemoryCircuit(distance=DISTANCE, n_rounds=ROUNDS, basis="Z",
                       add_boundary_detectors=True,
                       idle_error=ERROR_RATE, sqgate_error=ERROR_RATE,
                       tqgate_error=ERROR_RATE, spam_error=ERROR_RATE)
circuit = memory.stim_circuit

# H and O come from the Stim DEM via cudaq_qec.
dem_text = str(circuit.detector_error_model(decompose_errors=False))
dem = qec.dem_from_stim_text(dem_text)
H = np.array(dem.detector_error_matrix,    dtype=np.uint8)
O = np.array(dem.observables_flips_matrix, dtype=np.uint8)

# m2d (measurement → detector map) is extracted manually from the circuit's
# DETECTOR instructions.  It cannot be obtained automatically until the
# color-code memory circuit is available as a CUDA-Q kernel, at which point
# qec.decoder_context_from_memory_circuit() would provide it directly.
m2d = []
n = 0
MEASURE_OPS = {"M", "MZ", "MX", "MY", "MR", "MRX", "MRY", "MRZ"}
for inst in circuit.flattened():
    if inst.name in MEASURE_OPS:
        n += len(inst.targets_copy())
    elif inst.name == "DETECTOR":
        m2d.append(sorted(n + t.value for t in inst.targets_copy()))

# Ground truth: the last column of the detector sample (the observable flip).
# Both the detector sampler and the measurement sampler used in
# write_playback_from_circuit draw from the same noise realisations when
# given the same seed, so the observable column is consistent with the
# measurement records written to playback.txt.
dets_and_obs = circuit.compile_detector_sampler(seed=SEED).sample(
    SHOTS, append_observables=True).astype(np.uint8)
truth = dets_and_obs[:, -1]


# ── Playback Emulator ─────────────────────────────────────────────────────────
#
# Three files drive the emulator:
#
#   playback.txt   timed stream of reset / enqueue / get_corrections records
#   decoder.yaml   H / O / D matrices + decoder type (chromobius)
#   dem.txt        Stim DEM; Chromobius needs this instead of H for color codes

# playback.txt — paced by the circuit's own TICK structure.
# Provided by playback_from_circuit.py
write_playback_from_circuit(
    circuit, os.path.join(HERE, "playback.txt"), shots=SHOTS, seed=SEED)

# decoder.yaml — built with the cudaq_qec config API
dc = qec.decoder_config()
dc.id            = 0
dc.type          = "chromobius"
dc.block_size    = H.shape[1]
dc.syndrome_size = H.shape[0]
dc.H_sparse      = qec.pcm_to_sparse_vec(H)
dc.O_sparse      = qec.pcm_to_sparse_vec(O)
dc.D_sparse      = qec.d_sparse(m2d)

mdc = qec.multi_decoder_config()
mdc.decoders = [dc]
with open(os.path.join(HERE, "decoder.yaml"), "w") as f:
    f.write(mdc.to_yaml_str())

# dem.txt — per-detector color/basis coordinates.
with open(os.path.join(HERE, "dem.txt"), "w") as f:
    f.write(dem_text)

# server_decoder.yaml — same matrices, but multi_error_lut instead of chromobius.
# Chromobius requires a DEM string that cannot be delivered via the server YAML,
# so the decoding-server demo uses a matrix-based decoder instead.
dc_srv = qec.decoder_config()
dc_srv.id            = 0
dc_srv.type          = "multi_error_lut"
dc_srv.block_size    = H.shape[1]
dc_srv.syndrome_size = H.shape[0]
dc_srv.H_sparse      = qec.pcm_to_sparse_vec(H)
dc_srv.O_sparse      = qec.pcm_to_sparse_vec(O)
dc_srv.D_sparse      = qec.d_sparse(m2d)
dc_srv.decoder_custom_args = {"lut_error_depth": 2}
mdc_srv = qec.multi_decoder_config()
mdc_srv.decoders = [dc_srv]
with open(os.path.join(HERE, "server_decoder.yaml"), "w") as f:
    f.write(mdc_srv.to_yaml_str())

# truth.npy — ground-truth observable flips for scoring in run_playback.py.
np.save(os.path.join(HERE, "truth.npy"), truth)

print("d=%d r=%d p=%g  %d shots" % (DISTANCE, ROUNDS, ERROR_RATE, SHOTS))
print("  H=%dx%d  wrote decoder.yaml  server_decoder.yaml  dem.txt  "
      "playback.txt  truth.npy" % H.shape)

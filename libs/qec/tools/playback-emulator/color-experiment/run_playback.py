# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Replay the artifacts from write_playback.py through Chromobius and score.
#
# Env: PLAYBACK_EMULATOR — directory holding the built qec_playback_emulator module

import os
import sys

sys.path.insert(
    0,
    os.environ.get("PLAYBACK_EMULATOR")
)
import qec_playback_emulator as pe

import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
SINKS = ("ring_buffer_injector",)

truth = np.load(os.path.join(HERE, "truth.npy"))
shots = len(truth)


# ── Playback Emulator ─────────────────────────────────────────────────────────
#
# run_playback drives the timing loop and returns corrections for each shot.
# ring_buffer_injector writes wire frames directly — realistic 1 µs cadence.

results = {}
for sink in SINKS:
    results[sink] = pe.run_playback(
        playback = os.path.join(HERE, "playback.txt"),
        config   = os.path.join(HERE, "decoder.yaml"),
        dem_file = os.path.join(HERE, "dem.txt"),
        sink     = sink,
        tick     = "50us",
    )


# ── Scoring ───────────────────────────────────────────────────────────────────

print("%-22s %8s %8s %18s   logical errors" %
      ("sink", "events", "overruns", "latency p50 (ns)"))
for sink, r in results.items():
    w     = max(r.correction_width, 1)
    got   = [int(b) for b in r.corrections[0::w]]
    wrong = sum(g ^ t for g, t in zip(got, truth))
    print("%-22s %8d %8d %18d   %d/%d (%.1f%%)"
          % (sink, r.events, r.overruns, r.latency(0.5),
             wrong, shots, 100.0 * wrong / shots))


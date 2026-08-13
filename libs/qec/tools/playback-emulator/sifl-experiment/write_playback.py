#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.  All rights reserved.    #
# Apache License 2.0.                                                          #
# ============================================================================ #
"""Write the playback file and decoder config for the SIFL memory-circuit demo.

Timing flow
───────────
Init shot (once):
  reset
  enqueue syndrome_round × INIT_ROUNDS   (source_id=0)
  enqueue data_readout                   (source_id=1) → decoder fires; decode 0 starts

Each SIFL cycle (× N_SIFL):
  stream_until  source_id=0   feeds syndrome rounds while decode N-1 is running;
                               exits when try_get_corrections() → True, and
                               ABSORBS that read -- the corrections land in the
                               CSV on this row, so no get_corrections is needed
  enqueue       source_id=1   data-qubit readout: adds data bits to the decoder
                               for decode N; decode N fires at the shot boundary

stream_until's successful poll is a consuming read (reset=1): the server clears
the corrections and returns the session to `collecting` without calling
reset_decoder(), so the decoder's bits_since_decode_ survives into the next
shot -- which is what carries the SIFL feedback loop.

SIFL feedback
─────────────
The done-detection call in stream_until computes:
  sleep_bits = bits_since_decode_  (all bits since the last decode trigger)
  launch_decode(sleep_bits)        → sleep(sleep_bits × us_per_bit µs)
More syndrome rounds accumulated during stream_until → larger sleep_bits →
longer sleep → more syndrome rounds accumulate in the next stream_until →
geometric runaway when syndrome_width × us_per_bit > RTT.

Parameters (must match run.py):
  D_DIST      = 3      repetition code distance
  INIT_ROUNDS = 2      stabiliser rounds per shot
  N_SIFL      = 6      SIFL cycles
"""

import os

HERE           = os.path.dirname(os.path.abspath(__file__))
D_DIST         = 3
INIT_ROUNDS    = 2
N_SIFL         = 6
SYNDROME_WIDTH = D_DIST - 1          # 2
DATA_WIDTH     = D_DIST              # 3
BITS_PER_SHOT  = INIT_ROUNDS * SYNDROME_WIDTH + DATA_WIDTH   # 7

DECODER_YAML = """\
---
decoders:
  - id: 0
    type: dummy_sifl_decoder
    block_size: 1000000
    syndrome_size: 1
    H_sparse: [999999, -1]
    O_sparse: [0, -1]
    D_sparse: [999999, -1]
    decoder_custom_args:
      us_per_bit: {us_per_bit}
      num_obs: 1
      bits_per_shot: {bits_per_shot}
"""

with open(os.path.join(HERE, "decoder.yaml"), "w") as f:
    f.write(DECODER_YAML.format(us_per_bit=1.0, bits_per_shot=BITS_PER_SHOT))

with open(os.path.join(HERE, "playback.txt"), "w") as f:
    f.write("# SIFL memory-circuit demo  d=%d  INIT_ROUNDS=%d\n"
            % (D_DIST, INIT_ROUNDS))
    f.write("# source_id=0: syndrome rounds   source_id=1: data-qubit readout\n")
    f.write("# PLAYBACK_META decoder_id=0\n")

    # Init shot: seed the first decode
    t = 0
    f.write("%d  reset              0\n" % t); t += 1
    for _ in range(INIT_ROUNDS):
        f.write("%d  enqueue            0  source_id=0\n" % t); t += 1
    f.write("%d  enqueue            0  source_id=1  "
            "# data readout → decode 0 fires\n" % t); t += 1

    # SIFL cycles
    for cycle in range(N_SIFL):
        f.write("%d  stream_until       0  source_id=0  "
                "# rounds while decode %d runs; absorbs its corrections\n"
                % (t, cycle)); t += 1
        f.write("%d  enqueue            0  source_id=1  "
                "# data readout → decode %d fires\n" % (t, cycle + 1)); t += 1

print("Written:")
for name in ("decoder.yaml", "playback.txt"):
    path = os.path.join(HERE, name)
    print(f"  {name}  ({sum(1 for _ in open(path))} lines)")

print(f"\nd={D_DIST}  INIT_ROUNDS={INIT_ROUNDS}  "
      f"BITS_PER_SHOT={BITS_PER_SHOT}  N_SIFL={N_SIFL}")

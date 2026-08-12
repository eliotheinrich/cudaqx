# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Replay write_playback.py's artifacts through a decoding_server subprocess:
#
#   playback emulator --UDP--> decoding_server --> trt_decoder(ONNX)
#                                                    -> chromobius(residual)
#
# Run write_playback.py first.
#
# Env: PLAYBACK_EMULATOR (dir with the built module), DECODING_SERVER (binary)

import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("PLAYBACK_EMULATOR", "/workspaces/cudaqx/build/bin"))

import numpy as np, qec_playback_emulator as pe

SERVER = os.environ.get("DECODING_SERVER", "/home/.cudaqx/bin/decoding_server")
truth = np.load(os.path.join(HERE, "truth.npy"))

# --port=0 lets the OS pick; the server announces readiness on stdout.
proc = subprocess.Popen(
    [SERVER, "--config=" + os.path.join(HERE, "trt_decoder.yaml"),
     "--transport=udp", "--port=0"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
port = next((int(m.group(1)) for line in proc.stdout
             for m in [re.search(r"QEC_DECODING_SERVER_READY port=(\d+)", line)] if m), None)
if port is None:
    proc.terminate(); proc.wait()
    sys.exit("decoding_server did not report a port")
print("decoding_server ready on port %d" % port)

try:
    r = pe.run_playback(playback=os.path.join(HERE, "playback.txt"),
                        sink="udp_server", server_port=port,
                        server_observables=1, tick="500us")
finally:
    proc.terminate(); proc.wait()

# Latency is bimodal: enqueues are fire-and-forget, while reset and
# get_corrections block on a UDP round-trip.
width = max(r.correction_width, 1)
wrong = sum(int(b) ^ t for b, t in zip(r.corrections[0::width], truth))
print("events %d  overruns %d  latency p50 %d ns  p90 %d ns (blocking calls)"
      % (r.events, r.overruns, r.latency(0.5), r.latency(0.9)))
print("logical errors %d/%d (%.1f%%)" % (wrong, len(truth), 100.0 * wrong / len(truth)))

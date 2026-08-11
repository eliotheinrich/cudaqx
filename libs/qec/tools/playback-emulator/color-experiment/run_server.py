# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Replay playback.txt through a decoding_server subprocess over UDP.
# Demonstrates the two-process path: emulator → UDP → server → corrections.
#
# write_playback.py must have been run first.
#
# Env:
#   PLAYBACK_EMULATOR  — directory holding the built qec_playback_emulator module
#   DECODING_SERVER    — path to the decoding_server binary

import os
import re
import subprocess
import sys

sys.path.insert(
    0,
    os.environ.get("PLAYBACK_EMULATOR", "/workspaces/cudaqx/build/bin")
)

import qec_playback_emulator as pe

import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
SERVER = os.environ.get("DECODING_SERVER", "/home/.cudaqx/bin/decoding_server")

truth = np.load(os.path.join(HERE, "truth.npy"))
shots = len(truth)


# ── Decoding Server ───────────────────────────────────────────────────────────
#
# --port=0 lets the OS assign a free port.  The server prints
# "QEC_DECODING_SERVER_READY port=N ..." when it is ready to accept frames.

proc = subprocess.Popen(
    [SERVER,
     "--config=" + os.path.join(HERE, "server_decoder.yaml"),
     "--transport=udp", "--port=0"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)

port = None
for line in proc.stdout:
    m = re.search(r"QEC_DECODING_SERVER_READY port=(\d+)", line)
    if m:
        port = int(m.group(1)); break

if port is None:
    proc.terminate(); proc.wait()
    sys.exit("decoding_server did not report a port")

print("decoding_server ready on port %d" % port)


# ── Playback Emulator ─────────────────────────────────────────────────────────
#
# udp_server sink sends the same reset/enqueue/get_corrections frames over
# UDP to the server instead of an in-process session.

try:
    r = pe.run_playback(
        playback           = os.path.join(HERE, "playback.txt"),
        sink               = "udp_server",
        server_port        = port,
        server_observables = 1,
        tick               = "50us",
    )
finally:
    proc.terminate()
    proc.wait()


# ── Scoring ───────────────────────────────────────────────────────────────────

w     = max(r.correction_width, 1)
got   = [int(b) for b in r.corrections[0::w]]
wrong = sum(g ^ t for g, t in zip(got, truth))

# Latency is bimodal: most events are fire-and-forget enqueues (fast, ~µs);
# reset and get_corrections block on a UDP round-trip (slow, ~100s µs).
# p50 is dominated by enqueues; p90 captures the blocking calls.
print("events = %-6d  overruns = %-6d" % (r.events, r.overruns))
print("latency p50  = %7d ns  (enqueue stream — fire-and-forget)"
      % r.latency(0.5))
print("latency p90  = %7d ns  (reset / get_corrections — UDP round-trip)"
      % r.latency(0.9))
print("logical errors %d/%d (%.1f%%)" % (wrong, shots, 100.0 * wrong / shots))

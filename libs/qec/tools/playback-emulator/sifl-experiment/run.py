#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.  All rights reserved.    #
# Apache License 2.0.                                                          #
# ============================================================================ #
"""SIFL memory-circuit runaway demo.

Playback structure (per SIFL cycle)
────────────────────────────────────
  stream_until  source_id=0    feeds syndrome rounds to the decoder while the
                                PREVIOUS decode is running; try_get_corrections()
                                polls after each round and exits when the decode
                                is done.  That final poll is a CONSUMING read,
                                so the corrections land on this event's CSV row
                                and no separate get_corrections is needed.
  enqueue data_readout          data-qubit readout: adds DATA_WIDTH bits to
                                bits_since_decode_; once bits_since_decode_
                                reaches bits_per_shot the decoder fires the next
                                decode

SIFL feedback
─────────────
syndrome rounds in stream_until → bits_since_decode_ grows → longer sleep
→ more rounds fit in next stream_until → syndromes_streamed grows.
Runaway when  syndrome_width × us_per_bit  >  RTT.

Usage — spawn server automatically:
  python3 write_playback.py
  python3 run.py

Usage — connect to a running decoding_server:
  python3 run.py --port 12345
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "build"))
sys.path.insert(0, os.path.join(HERE, ".."))

import stim                          # noqa: E402
import qec_playback_emulator as pe   # noqa: E402

D_DIST         = 3
INIT_ROUNDS    = 2
N_SIFL         = 6
SYNDROME_WIDTH = D_DIST - 1
DATA_WIDTH     = D_DIST
BITS_PER_SHOT  = INIT_ROUNDS * SYNDROME_WIDTH + DATA_WIDTH   # 7

SHOTS = 10000
TICK  = "100us"

BUILD_BIN   = os.environ.get("BUILD_BIN",  "/workspaces/cudaqx/build/bin")
PLUGIN_DIR  = os.environ.get("PLUGIN_DIR", "/workspaces/cudaqx/build/lib/decoder-plugins")

DECODING_SERVER = os.path.join(BUILD_BIN, "decoding_server")
PLAYBACK        = os.path.join(HERE, "playback.txt")
DECODER_YAML    = os.path.join(HERE, "decoder.yaml")

for path in (PLAYBACK, DECODER_YAML):
    if not os.path.exists(path):
        sys.exit(f"Missing: {path} — run write_playback.py first.")


def build_sources(circuit: stim.Circuit, seed: int) -> dict:
    """Split StimMemorySource into syndrome and data-readout streams."""
    src = pe.StimMemorySource(circuit, shots=SHOTS, seed=seed)
    src.set_rounds(INIT_ROUNDS)
    syndrome_rounds, data_rounds = [], []
    for _ in range(SHOTS):
        for _ in range(INIT_ROUNDS):
            r = src.next_round()
            if not r:
                break
            syndrome_rounds.append(r)
        dr = src.next_round()
        if dr:
            data_rounds.append(dr)
    return {
        0: pe.StaticSyndromeSource(syndrome_rounds),
        1: pe.StaticSyndromeSource(data_rounds),
    }


def spawn_server(us_per_bit: float) -> tuple:
    if not os.path.exists(DECODING_SERVER):
        sys.exit(f"Missing: {DECODING_SERVER}")
    cfg = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    cfg.write(open(DECODER_YAML).read()
              .replace("us_per_bit: 1.0", f"us_per_bit: {us_per_bit}"))
    cfg.flush()
    env = os.environ.copy()
    env["CUDAQ_QEC_DECODER_PLUGIN_PATH"] = PLUGIN_DIR
    proc = subprocess.Popen(
        [DECODING_SERVER, f"--config={cfg.name}",
         "--transport=udp", "--port=0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )
    port = None
    for line in proc.stdout:
        m = re.search(r"READY port=(\d+)", line)
        if m:
            port = int(m.group(1))
            break
    os.unlink(cfg.name)
    if port is None:
        proc.terminate(); proc.wait()
        raise RuntimeError("decoding_server did not announce a port")
    return proc, port


def run_experiment(us_per_bit: float, port: int) -> list[dict]:
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=D_DIST, rounds=INIT_ROUNDS,
        after_clifford_depolarization=0.01,
    )
    sources = build_sources(circuit, seed=42)
    csv_path = tempfile.mktemp(suffix=".csv")
    pe.run_playback(
        PLAYBACK,
        sources=sources,
        sink="udp_server",
        server_host="127.0.0.1",
        server_port=port,
        server_observables=1,
        tick=TICK,
        wait_for_ready=True,
        csv=csv_path,
    )
    # stream_until absorbs the correction read, so everything for a SIFL cycle
    # lives on that one row: rounds streamed, the correction bits, and the
    # wall-clock the whole stream-and-poll loop took.
    return [{"streamed":   int(r["syndromes_streamed"]),
             "correction": r["correction"],
             "latency_ns": int(r["latency_ns"])}
            for r in csv.DictReader(open(csv_path))
            if r["op"] == "stream_until"]


parser = argparse.ArgumentParser(description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--port", type=int, default=None)
args = parser.parse_args()

EXPERIMENTS = [
    (10.0,  "STABLE   (10 µs/bit)"),
    (60.0,  "CRITICAL (60 µs/bit)"),
    (250.0, "RUNAWAY  (250 µs/bit)"),
]

print(f"d={D_DIST} rep code  INIT_ROUNDS={INIT_ROUNDS}  "
      f"bits_per_shot={BITS_PER_SHOT}")
print(f"Decoder fires at data-readout boundary.  "
      f"stream_until absorbs the correction read.\n")

rtt_estimate = None

for us_per_bit, label in EXPERIMENTS:
    print(f"us_per_bit={us_per_bit:<6}  [{label}]  "
          f"(init sleep ≈ {BITS_PER_SHOT * us_per_bit:.0f} µs)")
    print(f"  {'cycle':>5}  {'streamed':>10}  {'growth':>8}  "
          f"{'correction':>10}  {'stream_µs':>10}")

    srv, port = (None, args.port) if args.port else spawn_server(us_per_bit)

    try:
        cycles = run_experiment(us_per_bit, port)
    finally:
        if srv:
            srv.terminate(); srv.wait()

    prev = None
    for i, c in enumerate(cycles):
        s = c["streamed"]
        g = f"{s / prev:.2f}×" if prev else "—"
        us = c["latency_ns"] // 1000
        print(f"  {i:>5}  {s:>10}  {g:>8}  "
              f"{c['correction']:>10}  {us:>10}")
        prev = s or prev

    if cycles and cycles[0]["streamed"] > INIT_ROUNDS + 1:
        extra = cycles[0]["streamed"] - (INIT_ROUNDS + 1)
        rtt_estimate = BITS_PER_SHOT * us_per_bit / extra
    print()

if rtt_estimate:
    threshold = rtt_estimate / SYNDROME_WIDTH
    print(f"Estimated RTT ≈ {rtt_estimate:.0f} µs  |  "
          f"threshold ≈ {threshold:.0f} µs/bit  "
          f"(RTT / syndrome_width={SYNDROME_WIDTH})")

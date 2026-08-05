# qec_playback_emulator

This quick-start guide demonstrates the basic architecture and usage pattern for
the playback emulator. The API is subject to change; I am seeking feedback on the 
basic design before I proceed with more advanced features (i.e. dynamic circuits, 
multi-source syndromes, etc). The current prototype is capable of, for example, 
replaying syndromes sampled via the Ising-Decoder superdense color code memory 
circuits through the realtime path. See `color-experiment/` for a demo.

In essence, the tool replays a pre-recorded sequence of realtime-decoding RPC calls 
(`reset` / `enqueue` / `get_corrections`) on a precise hardware-independent schedule, 
then reports how accurately it hit that schedule.

## Architecture

```
  Stim circuit
       │
       ▼
 playback_from_circuit.py          (optional; any generator works)
       │  writes
       ▼
  playback file  
  (tick, op, decoder_id, bits)
       │  load_playback()
       ▼ 
  playback_file
  (flat buffer + record list)
       │  build_events(decoder_id, tick_ns)
       ▼ 
  event[]
  (offset_ns, op, *bits)
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  run loop  (single timing thread)           │
  │                                             │
  │  t0 = now() + lead_in_ns                    │
  │  for each event:                            │
  │    sleep until deadline - spin_slack_ns     │
  │    spin  until deadline                     │
  │    sink.send(event)          ──────────────────────────────┐
  │    record {deadline, call, return}          │              │
  └─────────────────────────────────────────────┘              │
                                                               │
            ┌──────────────────────────────────────────────────┘
            │  sink (chosen at startup)
            ├─ null_sink            discard; pure timing floor
            ├─ inproc_rpc_sink      rpc_producer → in-process
            ├─ ring_buffer_injector write directly into ring → in-process
            └─ udp_server_sink      UDP transport → decoding_server
```

**Timing thread discipline.**  Everything that could allocate, block, or page-fault
happens before `t0`: the file is parsed, the event table and result buffer are
allocated and pre-faulted, and the sink is warmed up.  Between two deadlines the
only work on the timing thread is the sleep+spin and one `sink.send()` call.
Deadlines are absolute offsets from `t0`, computed once. A late wake-up shifts
only the event that caused it, not all subsequent ones.

**Sink choice.**  The `null` sink gives the jitter floor with no decoder in the path.  
The two in-process sinks (`inproc_rpc` and `ring_buffer_injector`) drive a local 
`qec_realtime_session` and measure the realtime decode path end-to-end.  `udp_server` 
sends events to a realtime decoding server at the specified address/port.

One emulator process per decoder source; multiple sources sharing one playback file 
are distinguished by `decoder_id`.

## Playback file format

```
# tick  operation        decoder_id  [operands]
0       reset            0
500     enqueue          0           101
520     enqueue          0           010
540     get_corrections  0
1540    reset            0
...
```

Timestamps are in 'ticks'; the real duration of one tick is set with `--tick`
(e.g. `--tick=1us`).  A single file may contain records for multiple
`decoder_id` values; each process filters to its own ID and skips the rest.

Use `playback_from_circuit.py` to generate a playback file from a Stim circuit.

## Build

**Prerequisites:** CUDA-Q and CUDA-QX built and installed.

**1. Build the emulator**

```bash
cmake -S . -B build -G Ninja \
  -DCUDAQ_PREFIX=/usr/local/cudaq \
  -DCUDAQX_PREFIX=~/.cudaqx
cmake --build build
```

Run both commands from the `playback-emulator/` directory.
Both prefixes default to the values shown above; adjust if your installation
differs.  After the build:

```
build/qec_playback_emulator                   # CLI binary
build/qec_playback_emulator.cpython-*.so      # Python module
```

**2. Install Ising-Decoding** (for the color-code demo only)

```bash
git clone https://github.com/NVIDIA/ising-decoding /path/to/ising-decoding
pip install -r /path/to/ising-decoding/code/requirements_public_inference.txt
```

**3. Run the color-code demos**

```bash
cd color-experiment

# Stage 1 — sample the circuit and write artifacts
ISING_DECODING=/path/to/ising-decoding python3 write_playback.py

# Stage 2a — replay in-process through Chromobius
PLAYBACK_EMULATOR=../build python3 run_playback.py

# Stage 2b — replay through a decoding_server subprocess over UDP
PLAYBACK_EMULATOR=../build \
DECODING_SERVER=~/.cudaqx/bin/decoding_server \
python3 run_server.py
```

## Quick start (CLI)

```sh
# 1. Measure the jitter floor (no decoder needed).
qec_playback_emulator --playback=my.txt --tick=1us --sink=null --csv=jitter.csv

# 2a. Drive an in-process decoder.
qec_playback_emulator --playback=my.txt --tick=1us \
    --sink=ring_buffer_injector --config=decoder.yaml

# 2b. Drive a remote decoding_server over UDP.
qec_playback_emulator --playback=my.txt --tick=1us \
    --sink=udp_server --server-host=127.0.0.1 --server-port=<N>
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--playback=<file>` | *(required)* | Playback file to replay. |
| `--tick=<dur>` | *(required)* | Wall-clock duration of one tick (e.g. `1us`, `500ns`, `2.5ms`). |
| `--sink=<name>` | `null` | `null` — timing only; `inproc_rpc` — in-process, waits for each ACK; `ring_buffer_injector` — in-process, full ring depth; `udp_server` — send events to a remote `decoding_server`. |
| `--config=<file>` | | Decoder YAML config (required for `inproc_rpc` / `ring_buffer_injector`). |
| `--dem=<file>` | | DEM file for chromobius (alternative to `--config`). |
| `--decoder-id=N` | `0` | Which decoder entry in `--config` to load; also selects records from the playback file. |
| `--delta-ticks` | | Treat timestamps as inter-event gaps rather than absolute offsets from `t0`. |
| `--lead-in-ns=N` | `50000000` | Warm-up gap (ns) between `run()` call and first event (`t0`). Used to absorb cold-start decoder costs. |
| `--spin-slack-ns=N` | *(auto)* | Busy-spin window (ns) at the tail of each sleep. Auto-calibrated from `2 × worst wakeup overshoot` if omitted. |
| `--cpu=N` | | Pin the timing thread to CPU N. |
| `--csv=<file>` | | Write per-event timing records to a CSV file. |
| `--percentiles=A,B,...` | `50,90,99,99.9,100` | Lateness percentiles to print at the end of the run. |
| `--server-host=H` | `127.0.0.1` | `udp_server`: decoding_server hostname. |
| `--server-port=N` | *(required for udp_server)* | `udp_server`: port from the server's `QEC_DECODING_SERVER_READY` line. |
| `--server-slots=N` | `8` | `udp_server`: number of ring-buffer slots. |
| `--server-slot-size=N` | `256` | `udp_server`: bytes per slot. |
| `--server-observables=N` | `1` | `udp_server`: number of logical observable bits returned per `get_corrections`. |

## Python API

```python
import qec_playback_emulator as pe

r = pe.run_playback(playback="my.txt", config="decoder.yaml",
                    sink="ring_buffer_injector", tick="1us")
print(r.events, r.overruns, r.lateness(0.5))   # events, overruns, p50 lateness ns
```

Python keyword arguments mirror the CLI flags (hyphens → underscores).
`spin_slack_ns=0` (default) triggers auto-calibration.
`r.corrections` is flat (`reads × correction_width`); `r.lateness(q)` and
`r.latency(q)` return the q-th quantile of lateness / latency in ns.

# Playback emulator feedback — patchwork lattice-surgery spike (2026-08-05)

Context: replayed a patchwork-compiled d=5 lattice-surgery experiment
(`single_monolithic.toml`: XZ joint measurement, 321 qubits, 1292 detectors,
3 observables, 35 measurement instructions/shot with widths 24–105) through
`ring_buffer_injector` against a converted pymatching `multi_decoder_config`.
2000 shots, bit-exact agreement with patchwork-native pymatching at p=1e-4 and
p=2e-3 — and then through the full two-process `udp_server` →
`decoding_server` path (also bit-exact at both noise levels, with two wire
robustness findings, items 8–9). The tool basically Just Worked for this — the wire format
(`num_syndromes` per enqueue, bit-packed, no padding) and the decoder's
measurement buffer both handle phase-varying group widths with no changes.
Concrete friction found:

## 1. `write_playback_from_circuit` skips the reference frame
Stim detectors are measurement XORs *relative to the circuit's noiseless
reference sample*; the cudaq-qec `D_sparse` path applies raw XORs with no
constant offset. `write_playback_from_circuit` writes raw sampled bits, so any
circuit whose `reference_sample()` is nonzero decodes against inverted
detectors. Your color-code circuits and this patchwork config happen to have
all-zero reference samples, so it's latent — but patchwork does not guarantee
it. Suggest: `row XOR circuit.reference_sample()` before writing bits (that's
what our `spike/make_playback.py` does), or at least an assert + loud comment.

## 2. Measurement-coverage / trailing-enqueue trap
The decoder decodes as soon as `max(D_sparse)+1` bits arrive
(`calculate_num_msyn_per_decode`), and `DecodingSession::on_enqueue` moves the
shot back to `collecting` on ANY accepted input after `result_ready` — so if a
circuit has trailing measurements not referenced by any detector, enqueuing
them silently turns every `get_corrections` into NOT_READY. Nothing in the
playback generator checks this. Suggest: `playback_from_circuit` should
compute the referenced-measurement bound and either truncate or refuse.

## 3. Same-tick enqueues make `overruns` meaningless as a health metric
Patchwork rounds issue 2–3 measurement instructions per TICK, so their events
share a deadline and every second/third one counts as an overrun by
construction (we measured overruns == exactly the count of same-deadline
followers, ~17/shot). You already special-case the *last* two groups
(`decoder_deadline_ticks`, boundary bump); the general case needs either
intra-tick sub-ordering (e.g. deadline + k·epsilon for the k-th event on a
tick), coalescing same-tick groups into one enqueue (changes what's being
emulated), or an overruns stat that excludes same-deadline followers.

## 4. Expected-corrections verification belongs in the tool
We had to dump `r.corrections` to JSON and diff against a native pymatching
run out-of-band. The playback file (or a sidecar) could carry expected
correction bits per `get_corrections` record and the result could report a
mismatch count — the emulator already knows the read order, and "it ran on
time AND returned the right bits" is the actual pass criterion. This was the
single biggest missing feature for using the tool as a seam test.

## 5. `get_corrections` returns NOT_READY, and the result hides it
If the deadline is too tight, the session answers NOT_READY; the run completes
"successfully" and corrections silently stay stale/zero (we only caught this
risk by reading `DecodingSession::on_get_corrections`). An RPC status
histogram (`OK / NOT_READY / INTERNAL_ERROR` counts) in `PlaybackResult` would
make this failure mode visible. A `--wait-for-ready` retry mode would also
help separate "decoder too slow" from "schedule too tight".

## 6. Python API parity nits
- `csv=` exists on the CLI but not in `run_playback(**kwargs)`.
- `PLAYBACK_META syndrome_size=...` is written by `playback_from_circuit.py`
  but nothing reads it; for phase-varying circuits a single width is not even
  well-defined. Consider dropping it or writing the group-width list instead.
- `mlockall failed: Cannot allocate memory` warning inside a default docker
  container (no CAP_IPC_LOCK / memlock ulimit); maybe document that it is
  benign or suggest `--ulimit memlock=-1`.

## 7. Build prerequisites vs the shipped dev image
The emulator (and everything realtime in cudaqx main) needs the CUDA-Q
realtime headers/libs *matching the repo's `.cudaq_version` pin*. The current
cudaqx-dev image's `/usr/local/cudaq` is behind the pin (no
`gpu_roce_bridge_common.h`, no `cudaq_bridge_create_from_library`), so
`decoding_server` and the device-graph component don't build there — the
in-process sinks are fine. Mitigation that worked: the realtime subproject
builds standalone from the pinned commit in ~2 minutes and
`-DCUDAQ_REALTIME_ROOT=<fresh install>` unblocks everything. Worth a README
note on which cudaq commit the emulator was validated against.

## 8. udp_server sink: `claim()` matches responses by slot only — silent
## stale-corrections after a stall
Found on the two-process path (M4). After a ~850 us host-scheduler stall
mid-run, a completed 2000-shot run returned corrections that were content-
correct but stale by exactly one ring lap: the recorded correction for shot i
was the decode result of shot i-8 (offset climbed 1..8 over 9 shots, then
locked at ring depth). `remote_sink::claim()`/`await_reply()` gate on
`rx_flags_[slot]` alone; the response's `counter` is never compared to the
request tag, so a late response left in a slot is attributed to the NEXT
request that used it, permanently. Because status stays OK, nothing fails —
only content verification (item 4!) catches it. Fix: check the response
counter/tag in claim() and fail loudly (or resync) on mismatch.

## 9. udp_server sink: 8-slot rings are too shallow on a jittery host
On WSL2 (scheduler hiccups of 300–900 us), 2000-shot runs with the default
8×256 geometry stochastically died with `remote_sink: timed out awaiting a
response in slot N` at a random point (server logs prove every received frame
was dispatched; kernel UDP counters show no rcvbuf drops — the loss is in the
slot-recycling window). `--num-slots=16` on the server plus `server_slots=16`
on the sink made full runs pass reliably. Two asks: (a) surface the failure
with more context (how many events sent, which op timed out — we had to infer
shot position from the slot number), and (b) document ring depth as the first
knob to turn for timeouts on non-RT hosts.

## 10. run_server.py leaks the server on emulator failure
`run_playback` raising (e.g. the item-9 timeout) skips the
`proc.terminate()`; the leftover `decoding_server` spins at 100% CPU
indefinitely (its ring-poll loop). Wrap in try/finally. Same pattern bit our
copy first.
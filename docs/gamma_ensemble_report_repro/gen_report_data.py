# Reproduces the data behind the gamma-ensemble Relay-BP user guide.
# Fixed report config (nothing generalised): uniform_circuit Z-DEMs of the BB
# codes [[72,12,6]], [[144,12,12]], [[288,12,18]] at p=0.002, num_sets=600,
# proc_float=fp32, FirstConv, N in {1,2,4,8}, 150k shots. Records per-shot decode
# latency, iteration count and logical-error flag, plus the per-iteration time,
# into report_data.npz. Needs the built nv-qldpc-decoder plugin and the relay
# testdata checkout (env: PYTHONPATH/LD_LIBRARY_PATH for cudaq + cudaqx build).
import glob, os, time, statistics as st, numpy as np, stim, cudaq_qec as qec

RELAY = "/workspaces/relay/tests/testdata/bicycle_bivariate"
OUT = os.path.join(os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-report-repro"), "report_data.npz")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
SHOTS = int(os.environ.get("SHOTS", "150000"))
NS, NUM_SETS = [1, 2, 4, 8], 600
CODES = [("bb72", "BB [[72,12,6]]", "72_12_6", [-0.24, 0.66]),
         ("bb144", "BB [[144,12,12]]", "144_12_12", [-0.24, 0.66]),
         ("bb288", "BB [[288,12,18]]", "288_12_18", [-0.161, 0.815])]


def _detect_data_qubits(circuit):
    """Data qubits = those measured exactly once (final readout only)."""
    times_measured = [0] * circuit.num_qubits
    for inst in circuit:
        if inst.name.startswith("M") and not inst.gate_args_copy():
            for t in inst.targets_copy():
                times_measured[t.qubit_value] += 1
    return [q for q, n in enumerate(times_measured) if n == 1]


def filter_detectors_by_basis(circuit, basis):
    """Keep only detectors sensitive to *basis*-stabilizer errors; basis="Z"
    keeps the Z stabilizers (fire on X errors) -> the Z-component DEM."""
    pauli_error = "Z" if basis == "X" else "X"
    circuit = circuit.flattened(); noiseless = circuit.without_noise()
    ref_det = noiseless.compile_detector_sampler().sample(1, separate_observables=True)[0][0]
    sensitive = np.zeros(len(ref_det), dtype=bool)
    data_qubits = list(_detect_data_qubits(noiseless)); to_test = list(data_qubits); inst_idx = 0
    while to_test:
        for q in to_test:
            injected = stim.Circuit(); injected += noiseless
            injected.insert(inst_idx, stim.CircuitInstruction(f"{pauli_error}_ERROR", [q], [1.0]))
            inj_det = injected.compile_detector_sampler().sample(1, separate_observables=True)[0][0]
            sensitive[np.where(ref_det != inj_det)] = True
        to_test = []
        for inst in noiseless[inst_idx:]:
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(data_qubits); break
    filtered = stim.Circuit(); det_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            if not sensitive[det_idx]:
                det_idx += 1; continue
            det_idx += 1
        filtered.append(inst)
    return filtered


def build(tag):
    f = [x for x in glob.glob(f"{RELAY}/*") if f"_{tag}_memory_Z," in x
         and "error_rate=0.002," in x and "noise_model=uniform_circuit" in x][0]
    dem = filter_detectors_by_basis(stim.Circuit.from_file(f), "Z").detector_error_model()
    H = np.zeros((dem.num_detectors, dem.num_errors), np.uint8)
    L = np.zeros((dem.num_observables, dem.num_errors), np.uint8); er = np.zeros(dem.num_errors); e = 0
    for ins in dem.flattened():
        if ins.type == "error":
            er[e] = ins.args_copy()[0]
            for t in ins.targets_copy():
                if t.is_relative_detector_id(): H[t.val, e] = 1
                elif t.is_logical_observable_id(): L[t.val, e] = 1
            e += 1
    c = filter_detectors_by_basis(stim.Circuit.from_file(f), "Z")
    det, obs = c.compile_detector_sampler(seed=42).sample(SHOTS, separate_observables=True)
    return H, er, L, det.astype(np.uint8), obs.astype(np.uint8)


def make(H, er, N, gd, num_sets=NUM_SETS, max_iter=50, iter_per_check=1, pre_iter=1, crit="FirstConv"):
    return qec.get_decoder("nv-qldpc-decoder", H, max_iterations=max_iter, iter_per_check=iter_per_check,
        error_rate_vec=er, use_sparsity=True, bp_method=3, composition=1, gamma0=0.125, gamma_dist=gd,
        clip_value=200.0, repeatable=True, proc_float="fp32", gamma_ensemble_size=N,
        srelay_config={"pre_iter": pre_iter, "num_sets": num_sets, "stopping_criterion": crit},
        opt_results={"num_iter": True})


def per_iter_ns(H, er, N, gd, s0):
    def med(mi):
        d = make(H, er, N, gd, num_sets=max(8, N), max_iter=mi, iter_per_check=100000, pre_iter=0, crit="All")
        for _ in range(2): d.decode(s0)
        ts = []
        for _ in range(5):
            t = time.perf_counter(); r = d.decode(s0); ts.append(time.perf_counter() - t)
        return st.median(ts), int(r.opt_results["num_iter"])
    tlo, ilo = med(20); thi, ihi = med(80)
    return 1e9 * (thi - tlo) / (ihi - ilo)


store = {}
for slug, label, tag, gd in CODES:
    H, er, L, syn, truelog = build(tag)
    store[f"{slug}__label"] = label
    print(f"=== {label} | Z-DEM {H.shape[0]}x{H.shape[1]} | p=0.002 | num_sets={NUM_SETS} | {SHOTS} shots ===", flush=True)
    for N in NS:
        d = make(H, er, N, gd)
        for i in range(3): d.decode(syn[i])
        wall = np.empty(SHOTS); it = np.empty(SHOTS, np.int32); lerr = np.empty(SHOTS, bool)
        for i in range(SHOTS):
            t = time.perf_counter(); r = d.decode(syn[i]); wall[i] = 1e3 * (time.perf_counter() - t)
            it[i] = int(r.opt_results["num_iter"])
            pred = (np.array(r.result) > 0.5).astype(np.uint8)
            lerr[i] = not np.array_equal((L @ pred) & 1, truelog[i])
        store[f"{slug}__N{N}__wall"] = wall; store[f"{slug}__N{N}__iter"] = it; store[f"{slug}__N{N}__lerr"] = lerr
        np.savez(OUT, **store)
        print(f"  N={N:2d}: med={np.median(wall):6.2f}ms p99={np.percentile(wall,99):7.2f}ms LER={lerr.mean():.4f}", flush=True)
    s0 = np.zeros(H.shape[0], np.uint8); s0[0] = 1
    store[f"{slug}__periter"] = np.array([per_iter_ns(H, er, N, gd, s0) for N in NS])
    np.savez(OUT, **store)
    print("  per-iter(ns): " + ", ".join(f"N{n}={p:.0f}" for n, p in zip(NS, store[f"{slug}__periter"])), flush=True)
print("saved", OUT, flush=True)

# ============================================================================ #
# Ensemble RelayBP latency report -- data generation.
#
# A modified version of libs/qec/unittests/decoders/nv-qldpc-decoder/
# timing_stim_relay.py: sweeps gamma_ensemble_size N in {1,2,4,8} and the
# FirstConv / NConv stopping criteria (NOT All) over several codes:
#   - Tesseract bivariate-bicycle codes [[72,12,6]], [[144,12,12]]
#   - representative surface-code configs (d=5 r=5, d=9 r=9, d=9 r=27)
#
# For every code it records, per syndrome: wall-clock latency, num_iter (total
# BP iterations), and converged. It also measures the PURE per-iteration time
# (deterministic-budget two-point slope) per N. All raw arrays are saved to
# report_data.npz incrementally (crash-safe). gen_report.py turns this into
# tables + figures + report.md.
#
#   export PYTHONPATH=/usr/local/cudaq:/workspaces/cudaqx/build/python
#   python3 -u run_report.py
# ============================================================================ #
import glob
import time
import statistics as st
import numpy as np
import stim
import cudaq
import cudaq_qec as qec

BB_DIR = "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes"
DATA = "/workspaces/qec-ensemble-test/docs_report/report_data.npz"

NUM_SETS, MAX_ITER, PRE_ITER = 64, 50, 1
ENSEMBLE = [1, 2, 4, 8]
CRITERIA = [("FirstConv", "FirstConv", {}), ("NConv(5)", "NConv", {"stop_nconv": 5})]
GAMMA_DIST = [-0.24, 0.66]
# per-iteration measurement (deterministic budget) knobs
PI_LO, PI_HI, PI_REPS = 20, 80, 5

# code specs: (slug, label, kind, params, shots, gamma0)
CODES = [
    ("surf_d5_r5",  "surface d=5, r=5",  "surface", dict(d=5,  r=5,  p=0.008), 15000, 0.3),
    ("surf_d9_r9",  "surface d=9, r=9",  "surface", dict(d=9,  r=9,  p=0.008), 15000, 0.3),
    ("surf_d9_r27", "surface d=9, r=27", "surface", dict(d=9,  r=27, p=0.008), 10000, 0.3),
    ("bb72",  "BB [[72,12,6]]",   "bb", dict(tag="72,12,6",   p="0.003"), 10000, 0.125),
    ("bb144", "BB [[144,12,12]]", "bb", dict(tag="144,12,12", p="0.001"), 8000, 0.125),
]


def stim_to_cudaq_dem(dem):
    H = np.zeros((dem.num_detectors, dem.num_errors), np.uint8)
    er = np.zeros(dem.num_errors)
    e = 0
    for ins in dem.flattened():
        if ins.type == "error":
            er[e] = ins.args_copy()[0]
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    H[t.val, e] = 1
            e += 1
    cd = qec.DetectorErrorModel()
    cd.detector_error_matrix = H
    cd.error_rates = er
    cd.observables_flips_matrix = np.zeros((dem.num_observables, dem.num_errors), np.uint8)
    cd.error_ids = np.arange(dem.num_errors, dtype=np.int32)
    return cd


def build_bb(tag, p, shots):
    f = [x for x in glob.glob(f"{BB_DIR}/*") if tag in x and "_Z" in x and f"p={p}," in x][0]
    circuit = stim.Circuit.from_file(f)
    cd = stim_to_cudaq_dem(circuit.detector_error_model())
    cd.canonicalize_for_rounds(1)
    H = cd.detector_error_matrix
    sampler = circuit.compile_detector_sampler(seed=42)
    dets, _ = sampler.sample(shots, separate_observables=True, bit_packed=True)
    syn = np.unpackbits(dets, bitorder="little", axis=1).astype(np.uint8)[:, :H.shape[0]]
    return H, cd.error_rates, syn


def build_surface(d, r, p, shots):
    cudaq.set_target("stim")
    cudaq.set_random_seed(7)
    code = qec.get_code("surface_code", distance=d)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    syn, _ = qec.sample_memory_circuit(code, qec.operation.prep0, shots, r, noise)
    syn = syn.reshape((shots, r, -1))
    syn = syn[:, :, :syn.shape[2] // 2].reshape((shots, -1)).astype(np.uint8)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, r, noise)
    return dem.detector_error_matrix, np.array(dem.error_rates), syn


def make_decoder(H, er, N, criterion, extra, gamma0, max_iter=MAX_ITER,
                 iter_per_check=1, num_sets=NUM_SETS, pre_iter=PRE_ITER):
    srelay = {"pre_iter": pre_iter, "num_sets": num_sets, "stopping_criterion": criterion}
    srelay.update(extra)
    return qec.get_decoder(
        "nv-qldpc-decoder", H, max_iterations=max_iter, iter_per_check=iter_per_check,
        error_rate_vec=er, use_sparsity=True, bp_method=3, composition=1,
        gamma0=gamma0, gamma_dist=GAMMA_DIST, clip_value=200.0, repeatable=True,
        srelay_config=srelay, gamma_ensemble_size=N, opt_results={"num_iter": True})


def per_iter_ns(H, er, N, gamma0, syndrome):
    # deterministic budget: stopping=All + iter_per_check huge -> num_iter = 8*max_iter
    def med(mi):
        d = make_decoder(H, er, N, "All", {}, gamma0, max_iter=mi,
                         iter_per_check=100000, num_sets=8, pre_iter=0)
        for _ in range(2):
            d.decode(syndrome)
        ts = []
        for _ in range(PI_REPS):
            t0 = time.perf_counter()
            r = d.decode(syndrome)
            ts.append(time.perf_counter() - t0)
        return st.median(ts), int(r.opt_results["num_iter"])
    t_lo, it_lo = med(PI_LO)
    t_hi, it_hi = med(PI_HI)
    return 1e9 * (t_hi - t_lo) / (it_hi - it_lo)


store = {"codes": np.array([c[0] for c in CODES], dtype=object)}
for slug, label, kind, params, shots, gamma0 in CODES:
    t0 = time.time()
    if kind == "bb":
        H, er, syn = build_bb(params["tag"], params["p"], shots)
        pstr = params["p"]
    else:
        H, er, syn = build_surface(params["d"], params["r"], params["p"], shots)
        pstr = str(params["p"])
    det, err = H.shape
    store[f"{slug}__meta"] = np.array(
        [label, f"{det}x{err}", pstr, NUM_SETS, MAX_ITER, PRE_ITER, gamma0, shots],
        dtype=object)
    store[f"{slug}__crits"] = np.array([c[0] for c in CRITERIA], dtype=object)
    store[f"{slug}__ens"] = np.array(ENSEMBLE)
    print(f"\n=== {label} | DEM {det}x{err} | p={pstr} | {shots} shots "
          f"(built in {time.time()-t0:.0f}s) ===", flush=True)

    for ci, (clab, criterion, extra) in enumerate(CRITERIA):
        for N in ENSEMBLE:
            dec = make_decoder(H, er, N, criterion, extra, gamma0)
            for i in range(3):
                dec.decode(syn[i])
            wall = np.empty(shots); it = np.empty(shots, np.int32); cv = np.empty(shots, bool)
            for i in range(shots):
                t = time.perf_counter()
                r = dec.decode(syn[i])
                wall[i] = 1e3 * (time.perf_counter() - t)
                it[i] = int(r.opt_results["num_iter"]); cv[i] = bool(r.converged)
            store[f"{slug}__{ci}__N{N}__wall"] = wall
            store[f"{slug}__{ci}__N{N}__iter"] = it
            store[f"{slug}__{ci}__N{N}__conv"] = cv
            np.savez(DATA, **store)
            print(f"  {clab:10s} N={N}: med={np.median(wall):7.2f}ms "
                  f"p99={np.percentile(wall,99):8.2f}ms conv={cv.mean():.3f}", flush=True)

    # pure per-iteration time per N (deterministic budget)
    s0 = np.zeros(det, np.uint8); s0[0] = 1
    per = np.array([per_iter_ns(H, er, N, gamma0, s0) for N in ENSEMBLE])
    store[f"{slug}__periter"] = per
    np.savez(DATA, **store)
    print("  per-iter(ns): " + ", ".join(f"N{n}={p:.0f}" for n, p in zip(ENSEMBLE, per)),
          flush=True)

print(f"\nsaved {DATA}\ndone", flush=True)

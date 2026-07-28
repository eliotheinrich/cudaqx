# Ensemble RelayBP report + LER data (Z-component DEMs).
# Codes: surface d=9,r=9 and BB [[72,12,6]], [[144,12,12]], [[288,12,18]].
# Sweeps N in {1,2,4,8,16} and FirstConv / NConv(5). Per-code num_sets +
# gamma_dist (288 uses the tuned num_sets=600, gamma_dist=[-0.161,0.815]).
# Records per syndrome: wall latency, num_iter, converged, AND whether the
# decoded logical is WRONG (logical error) -- so the deadline figure can show
# LER(t) = P(latency > t or logical error). Also measures the pure per-iteration
# time per N. Saved incrementally to report_data.npz.
import glob, os, time, statistics as st, numpy as np, stim, cudaq, cudaq_qec as qec
from basis_filter import filter_detectors_by_basis

BB_DIR = os.environ.get("BB_DIR", "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes")
BB_NOISE = os.environ.get("BB_NOISE", "si1000")          # or "uniform_circuit" (relay testdata)
RELAY_BB_DIR = os.environ.get("RELAY_BB_DIR", "/workspaces/relay/tests/testdata/bicycle_bivariate")
BB_ONLY = os.environ.get("BB_ONLY", "0") == "1"          # skip the surface code
DATA_ROOT = os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-bb")
DATA = os.path.join(DATA_ROOT, "docs_report", "report_data.npz")
os.makedirs(os.path.dirname(DATA), exist_ok=True)
SHOTS = int(os.environ.get("SHOTS", "15000"))
SHOTS_CAP = int(os.environ.get("SHOTS_CAP", "0"))       # smoke override
MAX_ITER, PRE_ITER = 50, 1
ENSEMBLE = [1, 2, 4, 8, 16]
PROC_FLOAT = os.environ.get("PROC_FLOAT", "fp64")        # decoder float precision ("fp32"/"fp64")
CRITERIA = [("FirstConv", "FirstConv", {}), ("NConv(5)", "NConv", {"stop_nconv": 5})]
if os.environ.get("FIRSTCONV_ONLY", "0") == "1":
    CRITERIA = CRITERIA[:1]
GAMMA0 = 0.125
PI_LO, PI_HI, PI_REPS = 20, 80, 5
SURF_P = float(os.environ.get("SURF_P", "0.004"))
BB72_P = os.environ.get("BB72_P", "0.003")
BB144_P = os.environ.get("BB144_P", "0.003")
BB288_P = os.environ.get("BB288_P", "0.003")
# (slug, label, kind, params, num_sets, gamma_dist)
CODES = [
    ("surf_d9_r9", "surface d=9, r=9", "surface", dict(d=9, r=9, p=SURF_P), 600, [-0.24, 0.66]),
    ("bb72",  "BB [[72,12,6]]",   "bb", dict(tag="72,12,6",   p=BB72_P),  600, [-0.24, 0.66]),
    ("bb144", "BB [[144,12,12]]", "bb", dict(tag="144,12,12", p=BB144_P), 600, [-0.24, 0.66]),
    ("bb288", "BB [[288,12,18]]", "bb", dict(tag="288,12,18", p=BB288_P), 600, [-0.161, 0.815]),
]


def build_bb(tag, p, shots):
    """Z-component DEM. Returns H, error_rates, L (obs x err), syndromes, true logical flips."""
    if BB_NOISE == "uniform_circuit":                    # relay repo, native Z-basis memory circuit
        utag = tag.replace(",", "_")
        f = [x for x in glob.glob(f"{RELAY_BB_DIR}/*")
             if f"_{utag}_memory_Z," in x and f"error_rate={p}," in x and "noise_model=uniform_circuit" in x][0]
    else:                                                # tesseract si1000 file
        f = [x for x in glob.glob(f"{BB_DIR}/*") if tag in x and "_Z" in x and f"p={p}," in x][0]
    c = filter_detectors_by_basis(stim.Circuit.from_file(f), "Z")
    dem = c.detector_error_model()
    ND, NE, NO = dem.num_detectors, dem.num_errors, dem.num_observables
    H = np.zeros((ND, NE), np.uint8); L = np.zeros((NO, NE), np.uint8); er = np.zeros(NE); e = 0
    for ins in dem.flattened():
        if ins.type == "error":
            er[e] = ins.args_copy()[0]
            for t in ins.targets_copy():
                if t.is_relative_detector_id(): H[t.val, e] = 1
                elif t.is_logical_observable_id(): L[t.val, e] = 1
            e += 1
    det, obs = c.compile_detector_sampler(seed=42).sample(shots, separate_observables=True)
    return H, er, L, det.astype(np.uint8), obs.astype(np.uint8)


def build_surface(d, r, p, shots):
    cudaq.set_target("stim"); cudaq.set_random_seed(7)
    code = qec.get_code("surface_code", distance=d); noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    Lz = np.array(code.get_observables_z())
    syn, data = qec.z_sample_memory_circuit(code, qec.operation.prep0, shots, r, noise)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, r, noise)
    truelog = (Lz @ np.asarray(data).T) % 2
    return (np.array(dem.detector_error_matrix), np.array(dem.error_rates),
            np.array(dem.observables_flips_matrix), np.asarray(syn).astype(np.uint8),
            truelog.T.astype(np.uint8))


def make_decoder(H, er, N, criterion, extra, gamma_dist, num_sets, max_iter=MAX_ITER,
                 iter_per_check=1, pre_iter=PRE_ITER):
    srelay = {"pre_iter": pre_iter, "num_sets": num_sets, "stopping_criterion": criterion}; srelay.update(extra)
    return qec.get_decoder("nv-qldpc-decoder", H, max_iterations=max_iter, iter_per_check=iter_per_check,
        error_rate_vec=er, use_sparsity=True, bp_method=3, composition=1, gamma0=GAMMA0,
        gamma_dist=gamma_dist, clip_value=200.0, repeatable=True, srelay_config=srelay,
        proc_float=PROC_FLOAT, gamma_ensemble_size=N, opt_results={"num_iter": True})


def per_iter_ns(H, er, N, gamma_dist, syndrome):
    def med(mi):
        d = make_decoder(H, er, N, "All", {}, gamma_dist, num_sets=max(8, N), max_iter=mi,
                         iter_per_check=100000, pre_iter=0)
        for _ in range(2): d.decode(syndrome)
        ts = []
        for _ in range(PI_REPS):
            t0 = time.perf_counter(); r = d.decode(syndrome); ts.append(time.perf_counter() - t0)
        return st.median(ts), int(r.opt_results["num_iter"])
    t_lo, it_lo = med(PI_LO); t_hi, it_hi = med(PI_HI)
    return 1e9 * (t_hi - t_lo) / (it_hi - it_lo)


store = {"codes": np.array([c[0] for c in CODES], dtype=object)}
t_start = time.time()
for slug, label, kind, params, num_sets, gamma_dist in CODES:
    if BB_ONLY and kind != "bb":
        continue
    shots = SHOTS_CAP if SHOTS_CAP else SHOTS
    t0 = time.time()
    if kind == "bb":
        H, er, L, syn, truelog = build_bb(params["tag"], params["p"], shots); pstr = params["p"]
    else:
        H, er, L, syn, truelog = build_surface(params["d"], params["r"], params["p"], shots); pstr = str(params["p"])
    det, err = H.shape
    store[f"{slug}__meta"] = np.array([label, f"{det}x{err}", pstr, num_sets, MAX_ITER, PRE_ITER, GAMMA0, shots], dtype=object)
    store[f"{slug}__crits"] = np.array([c[0] for c in CRITERIA], dtype=object)
    store[f"{slug}__ens"] = np.array(ENSEMBLE)
    print(f"\n=== {label} | Z-DEM {det}x{err} | p={pstr} | num_sets={num_sets} | {shots} shots "
          f"(built {time.time()-t0:.0f}s, elapsed {(time.time()-t_start)/60:.0f}m) ===", flush=True)
    for ci, (clab, criterion, extra) in enumerate(CRITERIA):
        for N in ENSEMBLE:
            dec = make_decoder(H, er, N, criterion, extra, gamma_dist, num_sets)
            for i in range(3): dec.decode(syn[i])
            wall = np.empty(shots); it = np.empty(shots, np.int32)
            cv = np.empty(shots, bool); lerr = np.empty(shots, bool)
            for i in range(shots):
                t = time.perf_counter(); r = dec.decode(syn[i]); wall[i] = 1e3 * (time.perf_counter() - t)
                it[i] = int(r.opt_results["num_iter"]); cv[i] = bool(r.converged)
                pred = (np.array(r.result) > 0.5).astype(np.uint8)
                lerr[i] = not np.array_equal((L @ pred) & 1, truelog[i])
            store[f"{slug}__{ci}__N{N}__wall"] = wall
            store[f"{slug}__{ci}__N{N}__iter"] = it
            store[f"{slug}__{ci}__N{N}__conv"] = cv
            store[f"{slug}__{ci}__N{N}__lerr"] = lerr
            np.savez(DATA, **store)
            print(f"  {clab:10s} N={N:2d}: med={np.median(wall):7.2f}ms p99={np.percentile(wall,99):8.2f}ms "
                  f"conv={cv.mean():.3f} LER={lerr.mean():.4f}", flush=True)
    s0 = np.zeros(det, np.uint8); s0[0] = 1
    per = np.array([per_iter_ns(H, er, N, gamma_dist, s0) for N in ENSEMBLE])
    store[f"{slug}__periter"] = per
    np.savez(DATA, **store)
    print("  per-iter(ns): " + ", ".join(f"N{n}={p:.0f}" for n, p in zip(ENSEMBLE, per)), flush=True)
print(f"\nsaved {DATA}\ndone ({(time.time()-t_start)/60:.1f} min)", flush=True)

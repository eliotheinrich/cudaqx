# ============================================================================ #
# Deadline experiment: for each code, stopping criterion (FirstConv, NConv(5)),
# and ensemble size N in {1,2,4,8}, decode many syndromes and record per-shot
# wall-clock latency + converged. A high iteration budget (num_sets=128,
# max_iter=50 -> 6400 iters) makes the DEADLINE the binding constraint, not the
# leg cap (100% convergence, so nothing is cut off early).
#
# LER(t), computed in plot_deadline.py, is then: fraction of decodes NOT
# converged by deadline t  =  P( latency > t OR never converged ).
#
# Saves per-shot arrays to deadline_data.npz incrementally (crash-safe) so the
# plots regenerate instantly.
#
#   export PYTHONPATH=/usr/local/cudaq:/workspaces/cudaqx/build/python
#   python3 -u run_deadline.py
# ============================================================================ #
import glob, time, numpy as np, stim, cudaq, cudaq_qec as qec

BB_DIR = "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes"
DATA = "/workspaces/qec-ensemble-test/docs_deadline/deadline_data.npz"
NUM_SETS, MAX_ITER, PRE_ITER = 128, 50, 1
SHOTS = 15000
ENSEMBLE = [1, 2, 4, 8]
CRITERIA = [("FirstConv", "FirstConv", {}), ("NConv(5)", "NConv", {"stop_nconv": 5})]
GAMMA_DIST = [-0.24, 0.66]
CODES = [
    ("surf_d5_r5",  "surface d=5, r=5",  "surface", dict(d=5, r=5,  p=0.008), 0.3),
    ("surf_d9_r9",  "surface d=9, r=9",  "surface", dict(d=9, r=9,  p=0.008), 0.3),
    ("surf_d9_r27", "surface d=9, r=27", "surface", dict(d=9, r=27, p=0.008), 0.3),
    ("bb72",  "BB [[72,12,6]]",   "bb", dict(tag="72,12,6",   p="0.003"), 0.125),
    ("bb144", "BB [[144,12,12]]", "bb", dict(tag="144,12,12", p="0.001"), 0.125),
]


def stim_to_cudaq_dem(dem):
    H = np.zeros((dem.num_detectors, dem.num_errors), np.uint8); er = np.zeros(dem.num_errors); e = 0
    for ins in dem.flattened():
        if ins.type == "error":
            er[e] = ins.args_copy()[0]
            for t in ins.targets_copy():
                if t.is_relative_detector_id(): H[t.val, e] = 1
            e += 1
    cd = qec.DetectorErrorModel(); cd.detector_error_matrix = H; cd.error_rates = er
    cd.observables_flips_matrix = np.zeros((dem.num_observables, dem.num_errors), np.uint8)
    cd.error_ids = np.arange(dem.num_errors, dtype=np.int32); return cd


def build_bb(tag, p):
    f = [x for x in glob.glob(f"{BB_DIR}/*") if tag in x and "_Z" in x and f"p={p}," in x][0]
    c = stim.Circuit.from_file(f); cd = stim_to_cudaq_dem(c.detector_error_model()); cd.canonicalize_for_rounds(1)
    H = cd.detector_error_matrix; s = c.compile_detector_sampler(seed=42)
    dets, _ = s.sample(SHOTS, separate_observables=True, bit_packed=True)
    syn = np.unpackbits(dets, bitorder="little", axis=1).astype(np.uint8)[:, :H.shape[0]]
    return H, cd.error_rates, syn


def build_surface(d, r, p):
    cudaq.set_target("stim"); cudaq.set_random_seed(7)
    code = qec.get_code("surface_code", distance=d); noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    syn, _ = qec.sample_memory_circuit(code, qec.operation.prep0, SHOTS, r, noise)
    syn = syn.reshape((SHOTS, r, -1)); syn = syn[:, :, :syn.shape[2] // 2].reshape((SHOTS, -1)).astype(np.uint8)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, r, noise)
    return dem.detector_error_matrix, np.array(dem.error_rates), syn


def make_decoder(H, er, N, criterion, extra, gamma0):
    s = {"pre_iter": PRE_ITER, "num_sets": NUM_SETS, "stopping_criterion": criterion}; s.update(extra)
    return qec.get_decoder("nv-qldpc-decoder", H, max_iterations=MAX_ITER, error_rate_vec=er,
        use_sparsity=True, bp_method=3, composition=1, gamma0=gamma0, gamma_dist=GAMMA_DIST,
        clip_value=200.0, repeatable=True, srelay_config=s, gamma_ensemble_size=N,
        opt_results={"num_iter": True})


store = {"codes": np.array([c[0] for c in CODES], dtype=object)}
t_start = time.time()
for slug, label, kind, params, gamma0 in CODES:
    t0 = time.time()
    if kind == "bb":
        H, er, syn = build_bb(params["tag"], params["p"]); pstr = params["p"]
    else:
        H, er, syn = build_surface(params["d"], params["r"], params["p"]); pstr = str(params["p"])
    det, err = H.shape
    store[f"{slug}__meta"] = np.array([label, f"{det}x{err}", pstr, NUM_SETS, MAX_ITER, PRE_ITER, gamma0, SHOTS], dtype=object)
    store[f"{slug}__crits"] = np.array([c[0] for c in CRITERIA], dtype=object)
    store[f"{slug}__ens"] = np.array(ENSEMBLE)
    print(f"\n=== {label} | DEM {det}x{err} | p={pstr} | {SHOTS} shots (built {time.time()-t0:.0f}s, elapsed {(time.time()-t_start)/60:.0f}m) ===", flush=True)
    for ci, (clab, criterion, extra) in enumerate(CRITERIA):
        for N in ENSEMBLE:
            dec = make_decoder(H, er, N, criterion, extra, gamma0)
            for i in range(3): dec.decode(syn[i])
            wall = np.empty(SHOTS); it = np.empty(SHOTS, np.int32); cv = np.empty(SHOTS, bool)
            for i in range(SHOTS):
                t = time.perf_counter(); r = dec.decode(syn[i])
                wall[i] = 1e3 * (time.perf_counter() - t)
                it[i] = int(r.opt_results["num_iter"]); cv[i] = bool(r.converged)
            store[f"{slug}__{ci}__N{N}__wall"] = wall
            store[f"{slug}__{ci}__N{N}__iter"] = it
            store[f"{slug}__{ci}__N{N}__conv"] = cv
            np.savez(DATA, **store)
            print(f"  {clab:10s} N={N}: median={np.median(wall):7.2f}ms p99={np.percentile(wall,99):8.2f}ms conv={cv.mean():.4f}", flush=True)
print(f"\nsaved {DATA}\ndone ({(time.time()-t_start)/60:.1f} min)", flush=True)

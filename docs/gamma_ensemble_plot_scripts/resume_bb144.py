# Resume: finish the 3 missing subruns (bb144, NConv(5), N=2,4,8) and append to
# the existing deadline_data.npz. Same seed -> identical syndromes as the run.
import glob, time, numpy as np, stim, cudaq_qec as qec
DATA = "/workspaces/qec-ensemble-test/docs_deadline/deadline_data.npz"
BB_DIR = "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes"
NUM_SETS, MAX_ITER, PRE_ITER, SHOTS = 128, 50, 1, 15000
GAMMA_DIST, GAMMA0 = [-0.24, 0.66], 0.125
SLUG, CI = "bb144", 1                      # ci=1 -> NConv(5)
MISSING_N = [2, 4, 8]


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


f = [x for x in glob.glob(f"{BB_DIR}/*") if "144,12,12" in x and "_Z" in x and "p=0.001," in x][0]
c = stim.Circuit.from_file(f); cd = stim_to_cudaq_dem(c.detector_error_model()); cd.canonicalize_for_rounds(1)
H = cd.detector_error_matrix; er = cd.error_rates
dets, _ = c.compile_detector_sampler(seed=42).sample(SHOTS, separate_observables=True, bit_packed=True)
syn = np.unpackbits(dets, bitorder="little", axis=1).astype(np.uint8)[:, :H.shape[0]]
print(f"bb144 DEM {H.shape}, {SHOTS} shots; finishing NConv(5) N={MISSING_N}", flush=True)

store = dict(np.load(DATA, allow_pickle=True))
t_start = time.time()
for N in MISSING_N:
    srelay = {"pre_iter": PRE_ITER, "num_sets": NUM_SETS, "stopping_criterion": "NConv", "stop_nconv": 5}
    dec = qec.get_decoder("nv-qldpc-decoder", H, max_iterations=MAX_ITER, error_rate_vec=er,
        use_sparsity=True, bp_method=3, composition=1, gamma0=GAMMA0, gamma_dist=GAMMA_DIST,
        clip_value=200.0, repeatable=True, srelay_config=srelay, gamma_ensemble_size=N,
        opt_results={"num_iter": True})
    for i in range(3): dec.decode(syn[i])
    wall = np.empty(SHOTS); it = np.empty(SHOTS, np.int32); cv = np.empty(SHOTS, bool)
    for i in range(SHOTS):
        t = time.perf_counter(); r = dec.decode(syn[i])
        wall[i] = 1e3 * (time.perf_counter() - t); it[i] = int(r.opt_results["num_iter"]); cv[i] = bool(r.converged)
    store[f"{SLUG}__{CI}__N{N}__wall"] = wall
    store[f"{SLUG}__{CI}__N{N}__iter"] = it
    store[f"{SLUG}__{CI}__N{N}__conv"] = cv
    np.savez(DATA, **store)
    print(f"  NConv(5) N={N}: median={np.median(wall):.2f}ms p99={np.percentile(wall,99):.2f}ms "
          f"conv={cv.mean():.4f}  (elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)
print("done", flush=True)

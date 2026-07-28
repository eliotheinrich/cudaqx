# BB72-only convergence probe: 150k shots at p=0.002, but with a much larger
# relay budget (MAX_ITER x NUM_SETS) than the standard run (50 x 300), to test
# whether the LER floor is genuine misdecoding or the decoder exiting early
# because it exhausted a fixed number of relay sets.
# Records per shot: wall, num_iter, converged, logical-error. Prints the
# fraction that hit the full iteration ceiling. Saves report_data.npz in the
# standard layout so plot_deadline_ler.py / plot_bb.py can read it.
import os, glob, time, statistics as st, numpy as np, stim, cudaq_qec as qec
from basis_filter import filter_detectors_by_basis

BB_DIR = os.environ.get("BB_DIR", "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes")
DATA_ROOT = os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-bb72-highiter")
DATA = os.path.join(DATA_ROOT, "docs_report", "report_data.npz")
os.makedirs(os.path.dirname(DATA), exist_ok=True)
SHOTS = int(os.environ.get("SHOTS", "150000"))
P = os.environ.get("BB72_P", "0.002")
MAX_ITER = int(os.environ.get("MAX_ITER", "200"))     # was 50
NUM_SETS = int(os.environ.get("NUM_SETS", "2000"))    # was 300
PRE_ITER, GAMMA0, GAMMA_DIST = 1, 0.125, [-0.24, 0.66]
ENSEMBLE = [1, 2, 4, 8, 16]
CRITERIA = [("FirstConv", "FirstConv", {}), ("NConv(5)", "NConv", {"stop_nconv": 5})]
CEIL = PRE_ITER + NUM_SETS * MAX_ITER


def build_bb(tag, p, shots):
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


def make_decoder(H, er, N, criterion, extra):
    srelay = {"pre_iter": PRE_ITER, "num_sets": NUM_SETS, "stopping_criterion": criterion}; srelay.update(extra)
    return qec.get_decoder("nv-qldpc-decoder", H, max_iterations=MAX_ITER, iter_per_check=1,
        error_rate_vec=er, use_sparsity=True, bp_method=3, composition=1, gamma0=GAMMA0,
        gamma_dist=GAMMA_DIST, clip_value=200.0, repeatable=True, srelay_config=srelay,
        gamma_ensemble_size=N, opt_results={"num_iter": True})


H, er, L, syn, truelog = build_bb("72,12,6", P, SHOTS)
det, nerr = H.shape
store = {"codes": np.array(["bb72"], dtype=object),
         "bb72__meta": np.array(["BB [[72,12,6]]", f"{det}x{nerr}", P, NUM_SETS, MAX_ITER, PRE_ITER, GAMMA0, SHOTS], dtype=object),
         "bb72__crits": np.array([c[0] for c in CRITERIA], dtype=object),
         "bb72__ens": np.array(ENSEMBLE)}
print(f"=== BB [[72,12,6]] | Z-DEM {det}x{nerr} | p={P} | budget MAX_ITER={MAX_ITER} x NUM_SETS={NUM_SETS} "
      f"(ceiling={CEIL} iters, vs standard 50x300={1+300*50}) | {SHOTS} shots ===", flush=True)
t0 = time.time()
for ci, (clab, criterion, extra) in enumerate(CRITERIA):
    for N in ENSEMBLE:
        dec = make_decoder(H, er, N, criterion, extra)
        for i in range(3): dec.decode(syn[i])
        wall = np.empty(SHOTS); it = np.empty(SHOTS, np.int32)
        cv = np.empty(SHOTS, bool); lerr = np.empty(SHOTS, bool)
        for i in range(SHOTS):
            t = time.perf_counter(); r = dec.decode(syn[i]); wall[i] = 1e3 * (time.perf_counter() - t)
            it[i] = int(r.opt_results["num_iter"]); cv[i] = bool(r.converged)
            pred = (np.array(r.result) > 0.5).astype(np.uint8)
            lerr[i] = not np.array_equal((L @ pred) & 1, truelog[i])
        store[f"bb72__{ci}__N{N}__wall"] = wall; store[f"bb72__{ci}__N{N}__iter"] = it
        store[f"bb72__{ci}__N{N}__conv"] = cv; store[f"bb72__{ci}__N{N}__lerr"] = lerr
        np.savez(DATA, **store)
        print(f"  {clab:10s} N={N:2d}: LER={lerr.mean():.4f} conv={cv.mean():.4f} "
              f"%at-ceiling={np.mean(it>=CEIL):.4f} med_iter={int(np.median(it))} max_iter={it.max()} "
              f"med={np.median(wall):.2f}ms p99={np.percentile(wall,99):.2f}ms", flush=True)
# dummy per-iter so plot_bb.py doesn't choke (not the focus here)
store["bb72__periter"] = np.zeros(len(ENSEMBLE))
np.savez(DATA, **store)
print(f"\nsaved {DATA}\ndone ({(time.time()-t0)/60:.1f} min)", flush=True)

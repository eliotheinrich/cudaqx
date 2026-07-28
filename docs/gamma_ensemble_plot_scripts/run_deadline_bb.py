# BB-only deadline experiment (Z-component DEMs), N in {1,2,4,8,16},
# FirstConv / NConv(5). Per-code num_sets + gamma_dist (288 tuned). No surface.
import glob, os, time, numpy as np, stim, cudaq_qec as qec
from basis_filter import filter_detectors_by_basis

BB_DIR = os.environ.get("BB_DIR", "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes")
DATA_ROOT = os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-bb")
DATA = os.path.join(DATA_ROOT, "docs_deadline", "deadline_data.npz")
os.makedirs(os.path.dirname(DATA), exist_ok=True)
SHOTS = int(os.environ.get("SHOTS", "15000"))
MAX_ITER, PRE_ITER = 50, 1
ENSEMBLE = [1, 2, 4, 8, 16]
CRITERIA = [("FirstConv", "FirstConv", {}), ("NConv(5)", "NConv", {"stop_nconv": 5})]
GAMMA0 = 0.125
BB72_P = os.environ.get("BB72_P", "0.003")
BB144_P = os.environ.get("BB144_P", "0.003")
BB288_P = os.environ.get("BB288_P", "0.003")
CODES = [
    ("bb72",  "BB [[72,12,6]]",   "72,12,6",   BB72_P,  300, [-0.24, 0.66]),
    ("bb144", "BB [[144,12,12]]", "144,12,12", BB144_P, 300, [-0.24, 0.66]),
    ("bb288", "BB [[288,12,18]]", "288,12,18", BB288_P, 600, [-0.161, 0.815]),
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


def build_bb(tag, p, shots):
    f = [x for x in glob.glob(f"{BB_DIR}/*") if tag in x and "_Z" in x and f"p={p}," in x][0]
    c = filter_detectors_by_basis(stim.Circuit.from_file(f), "Z")
    cd = stim_to_cudaq_dem(c.detector_error_model()); cd.canonicalize_for_rounds(1)
    H = cd.detector_error_matrix
    dets, _ = c.compile_detector_sampler(seed=42).sample(shots, separate_observables=True, bit_packed=True)
    syn = np.unpackbits(dets, bitorder="little", axis=1).astype(np.uint8)[:, :H.shape[0]]
    return H, cd.error_rates, syn


def make_decoder(H, er, N, criterion, extra, gamma_dist, num_sets):
    s = {"pre_iter": PRE_ITER, "num_sets": num_sets, "stopping_criterion": criterion}; s.update(extra)
    return qec.get_decoder("nv-qldpc-decoder", H, max_iterations=MAX_ITER, error_rate_vec=er,
        use_sparsity=True, bp_method=3, composition=1, gamma0=GAMMA0, gamma_dist=gamma_dist,
        clip_value=200.0, repeatable=True, srelay_config=s, gamma_ensemble_size=N,
        opt_results={"num_iter": True})


store = {"codes": np.array([c[0] for c in CODES], dtype=object)}
t_start = time.time()
for slug, label, tag, p, num_sets, gamma_dist in CODES:
    t0 = time.time()
    H, er, syn = build_bb(tag, p, SHOTS)
    det, err = H.shape
    store[f"{slug}__meta"] = np.array([label, f"{det}x{err}", p, num_sets, MAX_ITER, PRE_ITER, GAMMA0, SHOTS], dtype=object)
    store[f"{slug}__crits"] = np.array([c[0] for c in CRITERIA], dtype=object)
    store[f"{slug}__ens"] = np.array(ENSEMBLE)
    print(f"\n=== {label} | Z-DEM {det}x{err} | p={p} | num_sets={num_sets} | {SHOTS} shots "
          f"(built {time.time()-t0:.0f}s, elapsed {(time.time()-t_start)/60:.0f}m) ===", flush=True)
    for ci, (clab, criterion, extra) in enumerate(CRITERIA):
        for N in ENSEMBLE:
            dec = make_decoder(H, er, N, criterion, extra, gamma_dist, num_sets)
            for i in range(3): dec.decode(syn[i])
            wall = np.empty(SHOTS); it = np.empty(SHOTS, np.int32); cv = np.empty(SHOTS, bool)
            for i in range(SHOTS):
                t = time.perf_counter(); r = dec.decode(syn[i]); wall[i] = 1e3 * (time.perf_counter() - t)
                it[i] = int(r.opt_results["num_iter"]); cv[i] = bool(r.converged)
            store[f"{slug}__{ci}__N{N}__wall"] = wall
            store[f"{slug}__{ci}__N{N}__iter"] = it
            store[f"{slug}__{ci}__N{N}__conv"] = cv
            np.savez(DATA, **store)
            print(f"  {clab:10s} N={N:2d}: median={np.median(wall):7.2f}ms p99={np.percentile(wall,99):8.2f}ms conv={cv.mean():.4f}", flush=True)
print(f"\nsaved {DATA}\ndone ({(time.time()-t_start)/60:.1f} min)", flush=True)

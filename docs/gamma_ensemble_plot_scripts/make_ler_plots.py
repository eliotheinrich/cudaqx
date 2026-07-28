# Logical-error-rate under a hard deadline (Z-component DEMs).
#
# LER(t) = P( latency > t  OR  logical error )  -- a shot succeeds only if it
# both finishes by the deadline t AND the decoder's logical prediction is
# correct. Contrast with the deadline-miss rate P(latency > t), which ignores
# decode correctness. As t -> infinity, LER(t) -> the intrinsic LER floor.
#
# Records per shot: wall latency + whether the decoded logical is wrong, for
# N in {1,2,4,8} under FirstConv and NConv(5). Writes ler_data.npz and two
# figures (surface, BB). Output dir is env-overridable.
#
#   python3 make_ler_plots.py            # full run (SHOTS=4000)
#   SHOTS=200 python3 make_ler_plots.py  # quick smoke
import glob, os, time, numpy as np, stim, cudaq, cudaq_qec as qec
from basis_filter import filter_detectors_by_basis

BB_DIR = os.environ.get("BB_DIR", "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes")
OUT = os.environ.get("LER_OUT", "/workspaces/gamma_ensemble_reports/ler_plots")
os.makedirs(OUT, exist_ok=True)
SHOTS = int(os.environ.get("SHOTS", "4000"))
NUM_SETS, MAX_ITER, PRE_ITER = 300, 50, 1
GAMMA_DIST, GAMMA0 = [-0.24, 0.66], 0.125
ENSEMBLE = [1, 2, 4, 8]
CRITERIA = [("FirstConv", "FirstConv", {}), ("NConv(5)", "NConv", {"stop_nconv": 5})]
# (slug, label, kind, params, shots) -- Z-component, baseline p
CODES = [
    ("surf_d9_r9",  "surface d=9, r=9",  "surface", dict(d=9, r=9,  p=0.008), SHOTS),
    ("surf_d9_r18", "surface d=9, r=18", "surface", dict(d=9, r=18, p=0.008), SHOTS),
    ("bb72",  "BB [[72,12,6]]",   "bb", dict(tag="72,12,6",   p="0.003"), SHOTS),
    ("bb144", "BB [[144,12,12]]", "bb", dict(tag="144,12,12", p="0.001"), SHOTS),
]


def build_bb(tag, p, shots):
    f = [x for x in glob.glob(f"{BB_DIR}/*") if tag in x and "_Z" in x and f"p={p}," in x][0]
    c = filter_detectors_by_basis(stim.Circuit.from_file(f), "Z")   # Z-component
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
    return H, er, L, det.astype(np.uint8), obs.astype(np.uint8)   # syn, true logical


def build_surface(d, r, p, shots):
    cudaq.set_target("stim"); cudaq.set_random_seed(7)
    code = qec.get_code("surface_code", distance=d); noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    Lz = np.array(code.get_observables_z())                          # (nlog x ndata)
    syn, data = qec.z_sample_memory_circuit(code, qec.operation.prep0, shots, r, noise)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, r, noise)
    true_log = (Lz @ np.asarray(data).T) % 2                         # (nlog x shots)
    L = np.array(dem.observables_flips_matrix)
    return dem.detector_error_matrix, np.array(dem.error_rates), L, \
        np.asarray(syn).astype(np.uint8), true_log.T.astype(np.uint8)


def make_decoder(H, er, N, criterion, extra):
    s = {"pre_iter": PRE_ITER, "num_sets": NUM_SETS, "stopping_criterion": criterion}; s.update(extra)
    return qec.get_decoder("nv-qldpc-decoder", H, max_iterations=MAX_ITER, error_rate_vec=er,
        use_sparsity=True, bp_method=3, composition=1, gamma0=GAMMA0, gamma_dist=GAMMA_DIST,
        clip_value=200.0, repeatable=True, srelay_config=s, gamma_ensemble_size=N,
        opt_results={"num_iter": True})


store = {}
t_start = time.time()
for slug, label, kind, params, shots in CODES:
    if kind == "bb":
        H, er, L, syn, truelog = build_bb(params["tag"], params["p"], shots)
    else:
        H, er, L, syn, truelog = build_surface(params["d"], params["r"], params["p"], shots)
    det, err = H.shape
    store[f"{slug}__meta"] = np.array([label, f"{det}x{err}", shots], dtype=object)
    print(f"\n=== {label} | Z-DEM {det}x{err} | {shots} shots (elapsed {(time.time()-t_start)/60:.0f}m) ===", flush=True)
    for ci, (clab, criterion, extra) in enumerate(CRITERIA):
        for N in ENSEMBLE:
            dec = make_decoder(H, er, N, criterion, extra)
            for i in range(3): dec.decode(syn[i])
            wall = np.empty(shots); lerr = np.empty(shots, bool)
            for i in range(shots):
                t = time.perf_counter(); r = dec.decode(syn[i]); wall[i] = 1e3 * (time.perf_counter() - t)
                pred = (np.array(r.result) > 0.5).astype(np.uint8)
                pred_log = (L @ pred) & 1
                lerr[i] = not np.array_equal(pred_log, truelog[i])
            store[f"{slug}__{ci}__N{N}__wall"] = wall
            store[f"{slug}__{ci}__N{N}__lerr"] = lerr
            print(f"  {clab:10s} N={N}: LER_floor={lerr.mean():.4f}  median={np.median(wall):6.2f}ms  p99={np.percentile(wall,99):7.2f}ms", flush=True)
store["codes"] = np.array([c[0] for c in CODES], dtype=object)
np.savez(f"{OUT}/ler_data.npz", **store)
print(f"\nsaved {OUT}/ler_data.npz ({(time.time()-t_start)/60:.1f} min)", flush=True)

# -------------------------------- plots -------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
Z = np.load(f"{OUT}/ler_data.npz", allow_pickle=True)
COLORS = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
NS = ENSEMBLE; CRITS = [(0, "FirstConv"), (1, "NConv(5)")]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 11,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID})


def make(codes, out):
    fig, axes = plt.subplots(len(CRITS), len(codes), figsize=(6.2 * len(codes), 8.2), sharey=True)
    axes = np.atleast_2d(axes)
    for j, slug in enumerate(codes):
        allw = np.concatenate([Z[f"{slug}__{ci}__N{N}__wall"] for ci, _ in CRITS for N in NS])
        grid = np.logspace(np.log10(allw.min() * 0.7), np.log10(allw.max() * 1.3), 160)
        for i, (ci, clab) in enumerate(CRITS):
            ax = axes[i, j]
            for N in NS:
                w = Z[f"{slug}__{ci}__N{N}__wall"]; le = Z[f"{slug}__{ci}__N{N}__lerr"]
                ler = np.array([((w > t) | le).mean() for t in grid])
                floor = le.mean()
                ax.plot(grid, ler, color=COLORS[N], lw=2.2, label=f"N = {N}" + (" (RelayBP)" if N == 1 else ""))
                ax.axhline(floor, color=COLORS[N], ls=(0, (1, 2)), lw=1.0, alpha=0.7)   # LER floor
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(str(Z[f"{slug}__meta"][0]), fontsize=12.5, fontweight="bold")
            if i == len(CRITS) - 1:
                ax.set_xlabel("hard deadline (ms)")
            if j == 0:
                ax.set_ylabel("logical error rate  LER(t)")
            ax.grid(True, which="major", color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    for i, (ci, clab) in enumerate(CRITS):
        axes[i, 0].annotate(clab, xy=(-0.24, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=12, fontweight="bold", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=9.5, loc="lower left",
                      title="dotted = intrinsic LER floor", title_fontproperties={"size": 8.5})
    fig.suptitle("Logical error rate vs hard deadline  —  LER(t) = P(latency > t or logical error)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(out, dpi=150, facecolor=SURF); print("wrote", out.split("/")[-1])


make(["surf_d9_r9", "surf_d9_r18"], f"{OUT}/ler_surface.png")
make(["bb72", "bb144"], f"{OUT}/ler_bb.png")

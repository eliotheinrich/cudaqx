# Hyperparameter search for BB72 [[72,12,6]] Z-DEM at p=0.002, to lower the
# intrinsic LER floor (currently ~0.060 at FirstConv N=1 on the default
# gamma_dist=[-0.24,0.66]).  Optimizes (gamma0, gamma_dist_min, gamma_dist_max)
# DIRECTLY against the nv-qldpc-decoder we deploy, so the result transfers 1:1.
# Objective = LER on a fixed tuning set (deterministic; repeatable=True).
# Strategy: seed the default + known-good points, random search, then a
# shrinking pattern (coordinate) search.  Validates the winner vs the default
# on a fresh 40k set and checks N=16 / NConv don't regress.
import os, glob, numpy as np, stim, cudaq_qec as qec
from basis_filter import filter_detectors_by_basis

BB_DIR = os.environ.get("BB_DIR", "/workspaces/tesseract-decoder/testdata/bivariatebicyclecodes")
P = os.environ.get("BB72_P", "0.002")
N_TUNE = int(os.environ.get("N_TUNE", "15000"))
N_VAL = int(os.environ.get("N_VAL", "40000"))
NUM_SETS, MAX_ITER, PRE_ITER = 300, 50, 1
BOUNDS = np.array([[0.0, 0.40], [-0.50, -0.02], [0.30, 1.20]])   # gamma0, gmin, gmax
DEFAULT = np.array([0.125, -0.24, 0.66])
RNG = np.random.default_rng(0)


def build_bb(tag, p, shots, seed):
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
    det, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return H, er, L, det.astype(np.uint8), obs.astype(np.uint8)


def ler(H, er, L, syn, truelog, x, N=1, criterion="FirstConv", extra=None):
    g0, gmin, gmax = x
    if gmin >= gmax - 1e-3:
        return 1.0
    srelay = {"pre_iter": PRE_ITER, "num_sets": NUM_SETS, "stopping_criterion": criterion}
    if extra: srelay.update(extra)
    dec = qec.get_decoder("nv-qldpc-decoder", H, max_iterations=MAX_ITER, iter_per_check=1,
        error_rate_vec=er, use_sparsity=True, bp_method=3, composition=1, gamma0=float(g0),
        gamma_dist=[float(gmin), float(gmax)], clip_value=200.0, repeatable=True,
        srelay_config=srelay, gamma_ensemble_size=N, opt_results={"num_iter": True})
    for i in range(3): dec.decode(syn[i])
    bad = 0
    for i in range(len(syn)):
        r = dec.decode(syn[i]); pred = (np.array(r.result) > 0.5).astype(np.uint8)
        bad += not np.array_equal((L @ pred) & 1, truelog[i])
    return bad / len(syn)


print(f"building BB72 Z-DEM at p={P}: tune={N_TUNE} (seed 1234), val={N_VAL} (seed 42)", flush=True)
H, er, L, synT, logT = build_bb("72,12,6", P, N_TUNE, seed=1234)
_, _, _, synV, logV = build_bb("72,12,6", P, N_VAL, seed=42)
print(f"  Z-DEM {H.shape[0]}x{H.shape[1]}", flush=True)

evals = []
def obj(x):
    v = ler(H, er, L, synT, logT, x)
    evals.append((v, tuple(x)))
    return v

# 1) seeds
seeds = [DEFAULT, [0.15, -0.22628, 0.62160], [0.125, -0.161, 0.815], [0.10, -0.30, 0.55], [0.20, -0.35, 0.80]]
best_x, best_v = None, 1e9
print("\n[seeds]", flush=True)
for s in seeds:
    v = obj(np.array(s, float))
    print(f"  g0={s[0]:.3f} dist=[{s[1]:+.3f},{s[2]:+.3f}]  LER={v:.4f}", flush=True)
    if v < best_v: best_v, best_x = v, np.array(s, float)

# 2) random search
NR = int(os.environ.get("N_RAND", "55"))
print(f"\n[random search x{NR}]", flush=True)
for k in range(NR):
    x = BOUNDS[:, 0] + RNG.random(3) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    if x[1] >= x[2] - 0.05: continue
    v = obj(x)
    if v < best_v:
        best_v, best_x = v, x.copy()
        print(f"  * new best LER={v:.4f}  g0={x[0]:.3f} dist=[{x[1]:+.3f},{x[2]:+.3f}]", flush=True)

# 3) pattern (coordinate) search
print(f"\n[pattern search from LER={best_v:.4f}]", flush=True)
step = np.array([0.06, 0.06, 0.10])
while step.max() > 0.006:
    improved = False
    for d in range(3):
        for sgn in (+1, -1):
            x = best_x.copy(); x[d] += sgn * step[d]
            x = np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])
            if x[1] >= x[2] - 0.05: continue
            v = obj(x)
            if v < best_v - 1e-4:
                best_v, best_x, improved = v, x.copy(), True
                print(f"  * LER={v:.4f}  g0={x[0]:.3f} dist=[{x[1]:+.3f},{x[2]:+.3f}] (step {step[d]:.3f})", flush=True)
    if not improved:
        step *= 0.5

print(f"\n=== BEST on tuning set: LER={best_v:.4f}  gamma0={best_x[0]:.4f}  "
      f"gamma_dist=[{best_x[1]:.4f}, {best_x[2]:.4f}]  ({len(evals)} evals) ===", flush=True)

# 4) validation on fresh 40k, default vs tuned, across configs
print(f"\n=== VALIDATION on {N_VAL} fresh shots (seed 42) ===", flush=True)
configs = [("FirstConv", 1, None), ("FirstConv", 16, None), ("NConv(5)", 1, {"stop_nconv": 5})]
print(f"{'config':16} | {'default LER':>12} | {'tuned LER':>12} | change")
for clab, N, extra in configs:
    crit = "NConv" if clab.startswith("NConv") else "FirstConv"
    vd = ler(H, er, L, synV, logV, DEFAULT, N=N, criterion=crit, extra=extra)
    vt = ler(H, er, L, synV, logV, best_x, N=N, criterion=crit, extra=extra)
    print(f"{clab+' N='+str(N):16} | {vd:12.4f} | {vt:12.4f} | {(vt-vd)/vd*100:+.1f}%", flush=True)
print(f"\ntuned params: gamma0={best_x[0]:.4f}, gamma_dist=[{best_x[1]:.4f}, {best_x[2]:.4f}]", flush=True)

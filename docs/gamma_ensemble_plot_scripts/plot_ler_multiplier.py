# LER performance multiplier vs N=1, per code, as a function of hard deadline t.
# multiplier_N(t) = LER_{N=1}(t) / LER_N(t),  LER_N(t) = P(latency>t OR logical error).
# >1 means ensemble size N has a lower LER than regular RelayBP at deadline t.
# Ratios are masked where the denominator is too small to be statistically meaningful.
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-bb-uniform-p002")
REP = os.path.join(ROOT, "docs_report"); OUT = os.path.join(ROOT, "docs_deadline")
os.makedirs(OUT, exist_ok=True)
Z = np.load(os.path.join(REP, "report_data.npz"), allow_pickle=True)
ORDER = [s for s in ["surf_d9_r9", "bb72", "bb144", "bb288"] if f"{s}__meta" in Z.files]
NS_ENS = [int(x) for x in os.environ.get("ENS", "1,2,4,8,16").split(",") if int(x) != 1]; CI = 0
YSCALE = os.environ.get("YSCALE", "log")           # "log" or "linear"
NCOL = {2: "#eb6834", 4: "#1baf7a", 8: "#eda100", 16: "#7b3294"}
MIN_COUNT = 10                                    # need >=10 failing shots for a meaningful ratio
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 10.5,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID})

fig, axes = plt.subplots(1, len(ORDER), figsize=(5 * len(ORDER), 4.6), sharey=True)
for ax, slug in zip(np.atleast_1d(axes), ORDER):
    shots = len(Z[f"{slug}__{CI}__N1__wall"])
    w1 = Z[f"{slug}__{CI}__N1__wall"]; le1 = Z[f"{slug}__{CI}__N1__lerr"]
    allw = np.concatenate([Z[f"{slug}__{CI}__N{N}__wall"] for N in [1] + NS_ENS])
    grid = np.logspace(np.log10(allw.min() * 0.7), np.log10(allw.max() * 1.3), 200)
    ler1 = np.array([((w1 > t) | le1).mean() for t in grid])
    for N in NS_ENS:
        w = Z[f"{slug}__{CI}__N{N}__wall"]; le = Z[f"{slug}__{CI}__N{N}__lerr"]
        lerN = np.array([((w > t) | le).mean() for t in grid])
        mult = np.where(lerN * shots >= MIN_COUNT, ler1 / np.where(lerN > 0, lerN, np.nan), np.nan)
        ax.plot(grid, mult, color=NCOL[N], lw=2.2, label=f"N = {N}")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.set_xscale("log"); ax.set_yscale(YSCALE)
    ax.set_title(f"{Z[f'{slug}__meta'][0]}", fontsize=11, fontweight="bold")
    ax.set_xlabel("hard deadline (ms)")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("LER$_{1}$(t) / LER$_{N}$(t)")
np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9.5, title="ensemble size",
                              title_fontproperties={"weight": "bold", "size": 9})
fig.suptitle("LER improvement over regular RelayBP (N=1) vs hard deadline", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT, "ler_multiplier.png" if YSCALE == "log" else "ler_multiplier_linear.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print("wrote", out)

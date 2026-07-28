# Combined deadline vs deadline-miss-rate figures with shared y-axis.
#   miss(t) = fraction of decodes NOT converged by deadline t.
# One figure per code family (surface, BB); rows = stopping criteria, columns =
# codes (code description as the column title). Reads deadline_data.npz.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.join(
    os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-test"), "docs_deadline")
D = np.load(f"{HERE}/deadline_data.npz", allow_pickle=True)
COLORS = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
NS = [1, 2, 4, 8]
CRITS = [(0, "FirstConv"), (1, "NConv(5)")]
SURF3 = os.environ.get("SURF3", "surf_d9_r27")   # third surface code slug (overridable)
DISP = {"surf_d9_r9": "surface code d=9, r=9", "surf_d9_r27": "surface code d=9, r=27",
        "bb72": "BB [[72,12,6]]", "bb144": "BB [[144,12,12]]"}
def disp(slug):                                  # title: DISP, else the code's own label
    return DISP.get(slug) or str(D[f"{slug}__meta"][0])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 11,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID})


def lat_eff(slug, ci, N):
    w = D[f"{slug}__{ci}__N{N}__wall"].astype(float)
    c = D[f"{slug}__{ci}__N{N}__conv"]
    w = w.copy(); w[~c] = np.inf
    return w


def make(codes, out):
    fig, axes = plt.subplots(len(CRITS), len(codes), figsize=(6.2 * len(codes), 8.2),
                             sharey=True)
    axes = np.atleast_2d(axes)
    for j, slug in enumerate(codes):
        allw = np.concatenate([D[f"{slug}__{ci}__N{N}__wall"] for ci, _ in CRITS for N in NS])
        grid = np.logspace(np.log10(allw.min() * 0.7), np.log10(allw.max() * 1.3), 140)
        for i, (ci, clab) in enumerate(CRITS):
            ax = axes[i, j]
            for N in NS:
                le = lat_eff(slug, ci, N)
                ler = np.array([(le > t).mean() for t in grid])
                ler = np.where(ler > 0, ler, np.nan)
                ax.plot(grid, ler, color=COLORS[N], lw=2.2,
                        label=f"N = {N}" + (" (RelayBP)" if N == 1 else ""))
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(disp(slug), fontsize=12.5, fontweight="bold")
            if i == len(CRITS) - 1:
                ax.set_xlabel("hard deadline (ms)")
            if j == 0:
                ax.set_ylabel("deadline-miss rate")
            ax.grid(True, which="major", color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    for i, (ci, clab) in enumerate(CRITS):           # criterion as a row label
        axes[i, 0].annotate(clab, xy=(-0.22, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=12, fontweight="bold", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    fig.savefig(out, dpi=150, facecolor=SURF)
    print("wrote", out.split("/")[-1])


make(["surf_d9_r9", SURF3], f"{HERE}/deadline_surface.png")
make(["bb72", "bb144"], f"{HERE}/deadline_bb.png")

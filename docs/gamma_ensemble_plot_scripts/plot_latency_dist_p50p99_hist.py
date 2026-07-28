# Histogram version of the latency-distribution figure (companion to the KDE
# version plot_latency_dist_p50p99.py, which is kept). Proper log-binned
# histograms (density per decade, so directly comparable to the KDE), with p50
# (solid) and p99 (dotted) marked per ensemble size N. Reads report_data.npz.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR = os.path.join(
    os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-test"), "docs_report")
Z = np.load(os.path.join(REPORT_DIR, "report_data.npz"), allow_pickle=True)
COLORS = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100"}
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
SURF3 = os.environ.get("SURF3", "surf_d9_r27")   # third surface code slug (overridable)
PANELS = ["surf_d9_r9", SURF3, "bb72", "bb144"]
NBINS = 90
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 10.5,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID})

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, slug in zip(axes.ravel(), PANELS):
    label, dem = Z[f"{slug}__meta"][0], Z[f"{slug}__meta"][1]
    NS = [int(n) for n in Z[f"{slug}__ens"]]
    walls = {N: Z[f"{slug}__0__N{N}__wall"] for N in NS}          # ci=0 = FirstConv
    lo = min(w.min() for w in walls.values()); hi = max(w.max() for w in walls.values())
    edges = np.logspace(np.log10(lo), np.log10(hi), NBINS + 1)    # log-spaced bin edges
    logedges = np.log10(edges)
    centers = 10 ** (0.5 * (logedges[:-1] + logedges[1:]))        # geometric bin centers
    for N in NS:                                                  # log-binned density (per decade)
        dens, _ = np.histogram(np.log10(walls[N]), bins=logedges, density=True)
        ax.fill_between(centers, dens, color=COLORS[N], alpha=0.10, zorder=2)
        ax.plot(centers, dens, color=COLORS[N], lw=2, zorder=3, label=f"N = {N}")
    for N in NS:                                                  # p50 solid, p99 dotted
        w = walls[N]
        ax.axvline(np.percentile(w, 50), color=COLORS[N], lw=1.3, ls="-", alpha=0.9)
        ax.axvline(np.percentile(w, 99), color=COLORS[N], lw=1.3, ls=(0, (1, 2)), alpha=0.9)
    ax.set_xscale("log")
    ax.set_title(f"{label}  (DEM {dem})", fontsize=11, fontweight="bold")
    ax.set_xlabel("per-syndrome decode latency (ms)")
    ax.set_ylabel("probability density")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right", title="ensemble size",
                  title_fontproperties={"weight": "bold", "size": 9})
fig.suptitle("Decode-latency distribution — "
             "solid = p50, dotted = p99", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(REPORT_DIR, "latency_dist_p50p99_hist.png"),
            dpi=150, facecolor=SURF)
print("wrote latency_dist_p50p99_hist.png")

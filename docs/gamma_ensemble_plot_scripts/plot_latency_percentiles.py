# Latency percentiles (p50 / p90 / p99) vs ensemble size N, one panel per code
# (FirstConv). Companion to the latency-distribution figure. Reads report_data.npz.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

Z = np.load("/workspaces/qec-ensemble-test/docs_report/report_data.npz", allow_pickle=True)
ORDER = ["surf_d5_r5", "surf_d9_r9", "surf_d9_r27", "bb72", "bb144"]
NS = [1, 2, 4, 8]
CI = 0                                   # FirstConv
QS = [("p50", 50), ("p90", 90), ("p99", 99)]
PC = {"p50": "#2a78d6", "p90": "#eb6834", "p99": "#1baf7a"}
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 10.5,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID})

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
axf = axes.ravel()
for ax, slug in zip(axf, ORDER):
    label, dem = Z[f"{slug}__meta"][0], Z[f"{slug}__meta"][1]
    for name, q in QS:
        ys = [np.percentile(Z[f"{slug}__{CI}__N{N}__wall"], q) for N in NS]
        ax.plot(NS, ys, "-o", color=PC[name], lw=2, ms=7, mec=SURF, mew=1.2, label=name)
        ax.annotate(f"{ys[-1] / ys[0]:.2f}×", (NS[-1], ys[-1]), textcoords="offset points",
                    xytext=(7, 0), color=PC[name], fontsize=9, fontweight="bold", va="center")
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xlim(0.8, 14)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_title(f"{label}  (DEM {dem})", fontsize=11, fontweight="bold")
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
# legend in the empty 6th slot (no overlap with data)
h, l = axf[0].get_legend_handles_labels()
axf[-1].axis("off")
axf[-1].legend(h, l, loc="center", frameon=False, fontsize=13, title="percentile",
               title_fontproperties={"weight": "bold", "size": 12})
fig.supxlabel("ensemble size N", fontsize=11, y=0.055)
fig.supylabel("latency (ms)", fontsize=11)
fig.text(0.5, 0.012, "Right-edge labels: each percentile at N=8 relative to N=1 "
         "(<1 = ensemble faster; smaller is better).", ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig("/workspaces/qec-ensemble-test/docs_report/latency_percentiles.png",
            dpi=150, facecolor=SURF)
print("wrote latency_percentiles.png")

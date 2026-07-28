# Report figure model_decomposition.png: the runtime decomposition
# T = I(N) x t_iter(N) shown as three panels (iterations to converge,
# per-iteration time, mean latency) vs ensemble size N, measured under the
# FirstConv stopping criterion.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPORT_DIR = os.path.join(
    os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-test"), "docs_report")
Z = np.load(os.path.join(REPORT_DIR, "report_data.npz"), allow_pickle=True)
NS = np.array([1, 2, 4, 8])
CI = 0                          # FirstConv
SURF3 = os.environ.get("SURF3", "surf_d9_r27")   # third surface code slug (overridable)
ORDER = ["surf_d5_r5", "surf_d9_r9", SURF3, "bb72", "bb144"]
COL = {"surf_d5_r5": "#9ecae1", "surf_d9_r9": "#2a78d6", SURF3: "#1baf7a",
       "bb72": "#eb6834", "bb144": "#e34948"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

# gather per-code quantities
P = {}
for s in ORDER:
    m = Z[f"{s}__meta"]
    per = Z[f"{s}__periter"] / 1000.0                       # t_iter, us
    I = np.array([np.median(Z[f"{s}__{CI}__N{N}__iter"]) for N in NS])
    Tm = np.array([np.mean(Z[f"{s}__{CI}__N{N}__wall"]) for N in NS])   # mean latency, ms
    P[s] = dict(label=str(m[0]), per=per, I=I, Tm=Tm)


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)


def logN(ax):
    ax.set_xscale("log", base=2); ax.set_xticks(NS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("ensemble size N")

# ---------------- Figure 1: T = I(N) x t_iter(N) ----------------------------- #
fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))
#fig.suptitle("Total runtime  T ≈ T0 + I(N) × t_iter(N,D)   —   "
#             "(A) fewer iterations × (B) costlier iterations = (C) net latency",
#             fontsize=12.5, fontweight="bold", y=1.0)


def edgelabels(a, items):        # (endpoint y, ratio, color); place at right edge, decluttered
    items = sorted(items, key=lambda e: e[0]); gap = 0.05; prev, ly = -1e9, []
    for yend, _, _ in items:
        v = max(np.log10(yend), prev + gap); ly.append(v); prev = v
    for (yend, r, col), v in zip(items, ly):
        a.text(NS[-1] * 1.13, 10 ** v, f"{r:.2f}×", color=col, fontsize=8.5,
               fontweight="bold", va="center")


labs = {0: [], 1: [], 2: []}
for s in ORDER:
    p = P[s]
    ax[0].plot(NS, p["I"], "-o", color=COL[s], lw=2, ms=7, mec=SURFACE, mew=1.2,
               label=p["label"])
    ax[1].plot(NS, p["per"], "-o", color=COL[s], ms=7, mec=SURFACE, mew=1.2)
    ax[2].plot(NS, p["Tm"], "-o", color=COL[s], lw=2, ms=7, mec=SURFACE, mew=1.2)
    labs[0].append((p["I"][-1], p["I"][-1] / p["I"][0], COL[s]))
    labs[1].append((p["per"][-1], p["per"][-1] / p["per"][0], COL[s]))
    labs[2].append((p["Tm"][-1], p["Tm"][-1] / p["Tm"][0], COL[s]))
for k in labs:
    edgelabels(ax[k], labs[k])
ax[0].set_title("(A) iterations to converge", loc="left", fontweight="bold")
ax[0].set_ylabel("median num_iter"); ax[0].set_yscale("log"); logN(ax[0]); despine(ax[0])
ax[0].legend(frameon=False, fontsize=8.5)
ax[1].set_title("(B) time per iteration", loc="left", fontweight="bold")
ax[1].set_ylabel("µs / iteration"); ax[1].set_yscale("log"); logN(ax[1]); despine(ax[1])
ax[2].set_title("(C) mean latency", loc="left", fontweight="bold")
ax[2].set_ylabel("mean decode latency (ms)"); ax[2].set_yscale("log"); logN(ax[2]); despine(ax[2])
for a in ax:                                                # room for right-edge labels
    a.set_xlim(0.85, NS[-1] * 1.6)
fig.text(0.5, 0.005, f"Right-edge labels: value at N={NS[-1]} relative to N=1 (×).",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(os.path.join(REPORT_DIR, "model_decomposition.png"),
            dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote model_decomposition.png")

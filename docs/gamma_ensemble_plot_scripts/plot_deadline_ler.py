# Standalone: for one run, plot BOTH the deadline-miss rate P(lat>t) and the
# LER(t)=P(lat>t OR logical error), 2 rows (FirstConv/NConv) x 4 codes, N lanes.
# Reads report_data.npz; writes deadline_miss.png and deadline_ler.png into docs_deadline/.
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-surf001-bb002")
REP = os.path.join(ROOT, "docs_report"); DLD = os.path.join(ROOT, "docs_deadline")
os.makedirs(DLD, exist_ok=True)
Z = np.load(os.path.join(REP, "report_data.npz"), allow_pickle=True)
ORDER = [s for s in ["surf_d9_r9", "bb72", "bb144", "bb288"] if f"{s}__meta" in Z.files]
NS = [int(x) for x in os.environ.get("ENS", "1,2,4,8,16").split(",")]
NCOL = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100", 16: "#7b3294"}
CRITS = [(ci, lab) for ci, lab in [(0, "FirstConv"), (1, "NConv(5)")]
         if f"{ORDER[0]}__{ci}__N1__wall" in Z.files]      # only criteria present in the data
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 10.5,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID})


def make(kind):   # kind = "miss" (P(lat>t)) or "ler" (P(lat>t OR err))
    share = (kind == "miss")          # LER: per-panel y so each code scales to its own floor
    fig, axes = plt.subplots(len(CRITS), len(ORDER), figsize=(5 * len(ORDER), 8.2), sharey=share)
    axes = np.atleast_2d(axes)
    floor = 0.5 / max(len(Z[f"{ORDER[0]}__0__N1__wall"]), 1)   # ~ 1/(2*shots)
    for j, slug in enumerate(ORDER):
        allw = np.concatenate([Z[f"{slug}__{ci}__N{N}__wall"] for ci, _ in CRITS for N in NS])
        grid = np.logspace(np.log10(allw.min() * 0.7), np.log10(allw.max() * 1.3), 160)
        for i, (ci, clab) in enumerate(CRITS):
            ax = axes[i, j]
            for N in NS:
                w = Z[f"{slug}__{ci}__N{N}__wall"]; le = Z[f"{slug}__{ci}__N{N}__lerr"]
                if kind == "miss":
                    y = np.array([(w > t).mean() for t in grid])
                else:
                    y = np.array([((w > t) | le).mean() for t in grid])
                ax.plot(grid, np.where(y > 0, y, np.nan), color=NCOL[N], lw=2.2,
                        label=f"N = {N}" + (" (RelayBP)" if N == 1 else ""))
            ax.set_xscale("log"); ax.set_yscale("log")
            if share:
                ax.set_ylim(bottom=floor)        # shared axis (miss): common floor
            if i == 0:
                ax.set_title(str(Z[f"{slug}__meta"][0]), fontsize=12.5, fontweight="bold")
            if i == len(CRITS) - 1:
                ax.set_xlabel("hard deadline (ms)")
            if j == 0:
                ax.set_ylabel("deadline-miss rate  P(lat>t)" if kind == "miss"
                              else "logical error rate  LER(t)")
            ax.grid(True, which="major", color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    for i, (ci, clab) in enumerate(CRITS):
        axes[i, 0].annotate(clab, xy=(-0.28, 0.5), xycoords="axes fraction", rotation=90,
                            va="center", ha="center", fontsize=12, fontweight="bold", color=INK)
    if kind == "miss":
        axes[0, 0].legend(frameon=False, fontsize=9.0, loc="lower left")
        title = "Deadline-miss rate vs hard deadline  —  P(latency > t)"
        out = "deadline_miss.png"
    else:
        axes[0, 0].legend(frameon=False, fontsize=9.0, loc="lower left")
        title = "Logical error rate vs hard deadline  —  LER(t) = P(latency > t or logical error)"
        out = "deadline_ler.png"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    fig.savefig(os.path.join(DLD, out), dpi=150, facecolor=SURF); plt.close(fig)
    print("wrote", os.path.join(DLD, out))


make("miss"); make("ler")

# Report figures for the surface d=9,r=9 + BB codes (Z-component), N in
# {1,2,4,8,16}. Produces four figures:
#   model_decomposition.png, latency_dist_p50p99_hist.png,
#   latency_percentiles.png, deadline_bb.png  (LER(t) under a hard deadline)
# All read report_data.npz, which records per-shot wall latency + logical error.
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.environ.get("QEC_DATA_ROOT", "/workspaces/qec-ensemble-bb")
REP = os.path.join(ROOT, "docs_report"); DLD = os.path.join(ROOT, "docs_deadline")
os.makedirs(DLD, exist_ok=True)
Zr = np.load(os.path.join(REP, "report_data.npz"), allow_pickle=True)
ORDER = [s for s in ["surf_d9_r9", "bb72", "bb144", "bb288"] if f"{s}__meta" in Zr.files]
CODECOL = {"surf_d9_r9": "#2a78d6", "bb72": "#eb6834", "bb144": "#e34948", "bb288": "#7b3294"}
NCOL = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a", 8: "#eda100", 16: "#7b3294"}
NS = [int(x) for x in os.environ.get("ENS", "1,2,4,8,16").split(",")]; CI = 0   # FirstConv for the report figs
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 10.5,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": GRID})


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)


def logN(ax):
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xlim(0.8, NS[-1] * 1.7)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("ensemble size N")


# ---------------- Figure 1: model_decomposition (A/B/C vs N) ----------------- #
P = {}
for s in ORDER:
    m = Zr[f"{s}__meta"]
    P[s] = dict(label=str(m[0]), dem=str(m[1]),
                I=np.array([np.median(Zr[f"{s}__{CI}__N{N}__iter"]) for N in NS]),
                per=(Zr[f"{s}__periter"] / 1000.0)[[list(Zr[f"{s}__ens"]).index(N) for N in NS]],
                Tm=np.array([np.mean(Zr[f"{s}__{CI}__N{N}__wall"]) for N in NS]))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))


def edgelabels(a, items):
    items = sorted(items, key=lambda e: e[0]); gap = 0.05; prev, ly = -1e9, []
    for yend, _, _ in items:
        v = max(np.log10(yend), prev + gap); ly.append(v); prev = v
    for (yend, r, col), v in zip(items, ly):
        a.text(NS[-1] * 1.13, 10 ** v, f"{r:.2f}×", color=col, fontsize=8.5, fontweight="bold", va="center")


labs = {0: [], 1: [], 2: []}
for s in ORDER:
    p = P[s]
    ax[0].plot(NS, p["I"], "-o", color=CODECOL[s], lw=2, ms=7, mec=SURF, mew=1.2, label=p["label"])
    ax[1].plot(NS, p["per"], "-o", color=CODECOL[s], ms=7, mec=SURF, mew=1.2)
    ax[2].plot(NS, p["Tm"], "-o", color=CODECOL[s], lw=2, ms=7, mec=SURF, mew=1.2)
    labs[0].append((p["I"][-1], p["I"][-1] / p["I"][0], CODECOL[s]))
    labs[1].append((p["per"][-1], p["per"][-1] / p["per"][0], CODECOL[s]))
    labs[2].append((p["Tm"][-1], p["Tm"][-1] / p["Tm"][0], CODECOL[s]))
for k in labs:
    edgelabels(ax[k], labs[k])
ax[0].set_title("(A) iterations to converge", loc="left", fontweight="bold")
ax[0].set_ylabel("median num_iter"); ax[0].set_yscale("log"); logN(ax[0]); despine(ax[0])
ax[0].legend(frameon=False, fontsize=8.5)
ax[1].set_title("(B) time per iteration", loc="left", fontweight="bold")
ax[1].set_ylabel("µs / iteration"); ax[1].set_yscale("log"); logN(ax[1]); despine(ax[1])
ax[2].set_title("(C) mean latency", loc="left", fontweight="bold")
ax[2].set_ylabel("mean decode latency (ms)"); ax[2].set_yscale("log"); logN(ax[2]); despine(ax[2])
fig.text(0.5, 0.005, f"Right-edge labels: value at N={NS[-1]} relative to N=1 (×).",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(os.path.join(REP, "model_decomposition.png"), dpi=150, bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# ---------------- Figure 2: latency distribution (1 x 4) --------------------- #
NBINS = 90
fig, axes = plt.subplots(1, len(ORDER), figsize=(5 * len(ORDER), 4.6))
for ax, slug in zip(np.atleast_1d(axes), ORDER):
    label, dem = Zr[f"{slug}__meta"][0], Zr[f"{slug}__meta"][1]
    walls = {N: Zr[f"{slug}__0__N{N}__wall"] for N in NS}
    lo = min(w.min() for w in walls.values()); hi = max(w.max() for w in walls.values())
    logedges = np.linspace(np.log10(lo), np.log10(hi), NBINS + 1)
    centers = 10 ** (0.5 * (logedges[:-1] + logedges[1:]))
    for N in NS:
        dens, _ = np.histogram(np.log10(walls[N]), bins=logedges, density=True)
        ax.fill_between(centers, dens, color=NCOL[N], alpha=0.10, zorder=2)
        ax.plot(centers, dens, color=NCOL[N], lw=2, zorder=3, label=f"N = {N}")
    for N in NS:
        ax.axvline(np.percentile(walls[N], 50), color=NCOL[N], lw=1.3, ls="-", alpha=0.9)
        ax.axvline(np.percentile(walls[N], 99), color=NCOL[N], lw=1.3, ls=(0, (1, 2)), alpha=0.9)
    ax.set_xscale("log"); ax.set_title(f"{label}  (Z-DEM {dem})", fontsize=11, fontweight="bold")
    ax.set_xlabel("per-syndrome decode latency (ms)"); ax.set_ylabel("probability density")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, fontsize=9, loc="upper right", title="ensemble size",
               title_fontproperties={"weight": "bold", "size": 9})
fig.suptitle("Decode-latency distribution — solid = p50, dotted = p99", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(REP, "latency_dist_p50p99_hist.png"), dpi=150, facecolor=SURF)
plt.close(fig)

# ---------------- Figure 3: latency percentiles (1 x 4) --------------------- #
QS = [("p50", 50), ("p90", 90), ("p99", 99)]
PC = {"p50": "#2a78d6", "p90": "#eb6834", "p99": "#1baf7a"}
fig, axes = plt.subplots(1, len(ORDER), figsize=(5 * len(ORDER), 4.6), sharex=True)
for ax, slug in zip(np.atleast_1d(axes), ORDER):
    label = Zr[f"{slug}__meta"][0]
    for name, q in QS:
        ys = [np.percentile(Zr[f"{slug}__{CI}__N{N}__wall"], q) for N in NS]
        ax.plot(NS, ys, "-o", color=PC[name], lw=2, ms=7, mec=SURF, mew=1.2, label=name)
        ax.annotate(f"{ys[-1] / ys[0]:.2f}×", (NS[-1], ys[-1]), textcoords="offset points",
                    xytext=(7, 0), color=PC[name], fontsize=9, fontweight="bold", va="center")
    ax.set_yscale("log"); ax.set_title(f"{label}", fontsize=11, fontweight="bold")
    logN(ax); despine(ax)
axes[0].legend(frameon=False, fontsize=11, title="percentile", title_fontproperties={"weight": "bold", "size": 10})
fig.supylabel("latency (ms)", fontsize=11)
fig.text(0.5, 0.012, f"Right-edge labels: each percentile at N={NS[-1]} relative to N=1",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(os.path.join(REP, "relaybp_gamma_ensemble_perf.png"), dpi=150, facecolor=SURF)
plt.close(fig)

# ---------------- Figure 4: LER(t) under a hard deadline (2 x 4) ------------- #
# LER(t) = P(latency > t or logical error). Dotted = intrinsic LER floor.
CRITS = [(ci, lab) for ci, lab in [(0, "FirstConv"), (1, "NConv(5)")]
         if f"{ORDER[0]}__{ci}__N1__wall" in Zr.files]
fig, axes = plt.subplots(len(CRITS), len(ORDER), figsize=(5 * len(ORDER), 4.1 * len(CRITS)), sharey=True)
axes = np.atleast_2d(axes)
for j, slug in enumerate(ORDER):
    allw = np.concatenate([Zr[f"{slug}__{ci}__N{N}__wall"] for ci, _ in CRITS for N in NS])
    grid = np.logspace(np.log10(allw.min() * 0.7), np.log10(allw.max() * 1.3), 160)
    for i, (ci, clab) in enumerate(CRITS):
        ax = axes[i, j]
        for N in NS:
            w = Zr[f"{slug}__{ci}__N{N}__wall"]; le = Zr[f"{slug}__{ci}__N{N}__lerr"]
            ler = np.array([((w > t) | le).mean() for t in grid])
            ax.plot(grid, np.where(ler > 0, ler, np.nan), color=NCOL[N], lw=2.2,
                    label=f"N = {N}" + (" (RelayBP)" if N == 1 else ""))
            if le.mean() > 0:                                 # LER floor (skip if error-free)
                ax.axhline(le.mean(), color=NCOL[N], ls=(0, (1, 2)), lw=1.0, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(bottom=5e-5)
        if i == 0:
            ax.set_title(str(Zr[f"{slug}__meta"][0]), fontsize=12.5, fontweight="bold")
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
axes[0, 0].legend(frameon=False, fontsize=9.0, loc="lower left",
                  title="dotted = intrinsic LER floor", title_fontproperties={"size": 8.5})
fig.suptitle("Logical error rate vs hard deadline  —  LER(t) = P(latency > t or logical error)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0.02, 0, 1, 0.97])
fig.savefig(os.path.join(DLD, "deadline_bb.png"), dpi=150, facecolor=SURF)
plt.close(fig)
print("wrote model_decomposition.png, latency_dist_p50p99_hist.png, relaybp_gamma_ensemble_perf.png, deadline_bb.png (LER)")

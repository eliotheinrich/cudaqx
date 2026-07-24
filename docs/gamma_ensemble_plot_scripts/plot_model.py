# Visualize the runtime model  T = T0 + I(N) * [ b + c*max(1, N*D/C) ]
# against the measured data (FirstConv). Produces two figures:
#   model_decomposition.png : T = I(N) x t_iter(N)  (3 panels)
#   model_law.png           : universal saturation collapse + help/hurt outcome
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

Z = np.load("/workspaces/qec-ensemble-test/docs_report/report_data.npz", allow_pickle=True)
C = 322.0                       # measured GPU co-resident block capacity
NS = np.array([1, 2, 4, 8])
CI = 0                          # FirstConv
ORDER = ["surf_d5_r5", "surf_d9_r9", "surf_d9_r27", "bb72", "bb144"]
COL = {"surf_d5_r5": "#9ecae1", "surf_d9_r9": "#2a78d6", "surf_d9_r27": "#1baf7a",
       "bb72": "#eb6834", "bb144": "#e34948"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

# gather per-code quantities
P = {}
for s in ORDER:
    m = Z[f"{s}__meta"]; D = int(str(m[1]).split("x")[0])
    per = Z[f"{s}__periter"] / 1000.0                       # t_iter, us
    I = np.array([np.median(Z[f"{s}__{CI}__N{N}__iter"]) for N in NS])
    T = np.array([np.median(Z[f"{s}__{CI}__N{N}__wall"]) for N in NS])  # ms
    Tm = np.array([np.mean(Z[f"{s}__{CI}__N{N}__wall"]) for N in NS])   # mean latency, ms
    x = np.maximum(1.0, NS * D / C)                         # max(1, N D / C)
    c = (per[3] - per[0]) / (x[3] - x[0]); b = per[0] - c * x[0]        # fit b,c
    P[s] = dict(label=str(m[0]), D=D, per=per, I=I, T=T, Tm=Tm, x=x, b=b, c=c)


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
Ncont = np.linspace(1, 8, 100)


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
    tmodel = p["b"] + p["c"] * np.maximum(1.0, Ncont * p["D"] / C)
#    ax[1].plot(Ncont, tmodel, "-", color=COL[s], lw=1.8)
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
    a.set_xlim(0.85, 11)
fig.text(0.5, 0.005, "Right-edge labels: value at N=8 relative to N=1 (×).",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig("/workspaces/qec-ensemble-test/docs_report/model_decomposition.png",
            dpi=150, bbox_inches="tight", facecolor=SURFACE)

# ---------------- Figure 2: universal law + help/hurt ------------------------ #
fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
fig.suptitle("The saturation knee at N·D = C, and its consequence for latency",
             fontsize=12.5, fontweight="bold", y=1.0)
# (A) measured rows-per-block (= D/blocks_x) vs N·D/C -> exactly max(1, N·D/C).
# Uses directly-measured blocks_x (blocks_raw.txt); small codes (d=3,d=5) sit
# BELOW the knee on the flat part.
import re
D_INFO = {12: ("surface d=3", "#c6dbef"), 60: ("surface d=5", "#6baed6"),
          360: ("surface d=9,r=9", "#2a78d6"), 1080: ("surface d=9,r=27", "#1baf7a"),
          432: ("BB [[72,12,6]]", "#eb6834"), 1728: ("BB [[144,12,12]]", "#e34948")}
pts = {}
for line in open("/workspaces/qec-ensemble-test/docs_report/blocks_raw.txt"):
    mm = re.search(r"N=(\d+) syndrome_size=(\d+) blocks_x=(\d+).*capacity=(\d+)", line)
    if mm:
        N, D, bx, cap = map(int, mm.groups())
        pts.setdefault(D, []).append((N * D / cap, D / bx))
xs = np.logspace(np.log10(0.03), np.log10(60), 200)
ax[0].plot(xs, np.maximum(1.0, xs), color=INK, lw=2.5, zorder=2,
           label="max(1, N·D/C)")
ax[0].axvline(1.0, color=MUTED, ls=(0, (4, 3)), lw=1.2)
ax[0].annotate("knee: N·D = C", (1.0, 40), color=MUTED, fontsize=9, ha="center")
ax[0].annotate("flat: lanes fit\n(free)", (0.1, 2.2), color=MUTED, fontsize=9, ha="center")
ax[0].annotate("∝ N·D: lanes\nserialize", (18, 3.0), color=MUTED, fontsize=9, ha="center")
for D in sorted(pts):
    lab, col = D_INFO[D]
    xy = sorted(pts[D]); xx = [a for a, b in xy]; yy = [b for a, b in xy]
    ax[0].plot(xx, yy, "o", color=col, ms=8, mec=SURFACE, mew=1.3, zorder=3,
               label=f"{lab}  (D={D})")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("N·D / C   (ensemble × detectors / GPU block capacity)")
ax[0].set_ylabel("rows per block  =  D / blocks_x   (measured)")
ax[0].set_title("(A) measured blocks_x geometry: rows/block = max(1, N·D/C)",
                loc="left", fontweight="bold")
despine(ax[0]); ax[0].legend(frameon=False, fontsize=8.0, loc="upper left")
# (B) T(N)/T(1): help (<1) vs hurt (>1)
ax[1].axhline(1.0, color=MUTED, ls=(0, (4, 3)), lw=1.2)
ax[1].annotate("break-even", (1, 1.0), textcoords="offset points", xytext=(2, 5),
               color=MUTED, fontsize=9)
for s in ORDER:
    p = P[s]
    ax[1].plot(NS, p["T"] / p["T"][0], "-o", color=COL[s], lw=2, ms=7,
               mec=SURFACE, mew=1.2, label=p["label"])
    ax[1].annotate(f"{p['T'][-1]/p['T'][0]:.2f}×", (8, p["T"][-1] / p["T"][0]),
                   textcoords="offset points", xytext=(7, 0), color=COL[s],
                   fontsize=9, fontweight="bold", va="center")
logN(ax[1]); ax[1].set_ylabel("T(N) / T(N=1)   (<1 faster, >1 slower)")
ax[1].set_title("(B) net outcome: surface speeds up, BB slows down",
                loc="left", fontweight="bold")
ax[1].set_yscale("log"); ax[1].set_yticks([0.5, 0.7, 1, 2, 3, 4])
ax[1].get_yaxis().set_major_formatter(mticker.ScalarFormatter())
despine(ax[1]); ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/workspaces/qec-ensemble-test/docs_report/model_law.png",
            dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote model_decomposition.png, model_law.png")

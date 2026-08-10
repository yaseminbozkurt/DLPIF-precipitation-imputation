"""
Figure 11 - Graded context-loss ablation (RQ4a), three panels: local-loss,
neighbour-loss, joint-loss. Rebuilt from results/rq4_context_availability/
graded/graded_{local,neighbour,joint}_loss.csv.

PRECIP mask fixed at MCAR-10 (identical 416 masked positions at every
level in every panel); error bars are +/- 1 SD over 3 context-mask seeds
x 3 model seeds (n=9 per point). Adjacency uses k=2 neighbours per
station (verified against adjacency.pkl, not the originally assumed k=3).
Local-loss and neighbour-loss were varied independently BEFORE joint-loss.

The message this figure carries: neighbour information is the dominant
driver of occurrence robustness (panel B drops sharply), local
information alone is nearly inert (panel A is nearly flat), but complete
collapse (F1=0, matching NetBlock-30d) only appears in panel C when BOTH
vanish together -- an interaction effect a single scalar availability
score cannot represent, which is why RQ4b builds an interaction-aware
gate rather than a scalar-threshold gate.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRADED = os.path.join(REPO, "results", "rq4_context_availability", "graded")

COL_LOCAL = "#0072B2"     # blue
COL_NBR = "#D55E00"       # vermillion
COL_JOINT = "#CC79A7"     # reddish purple

local_df = pd.read_csv(os.path.join(GRADED, "graded_local_loss.csv"))
nbr_df = pd.read_csv(os.path.join(GRADED, "graded_neighbour_loss.csv"))
joint_df = pd.read_csv(os.path.join(GRADED, "graded_joint_loss.csv"))

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=150, sharey=True)

# ── Panel A: local-loss ─────────────────────────────────────────────────
ax = axes[0]
order = [0, 2, 3, 4, 6]
labels = ["0/6", "2/6", "3/6", "4/6", "6/6"]
g = local_df.groupby("keep_count")["f1"].agg(["mean", "std", "count"]).reindex(order)
x = np.arange(len(order))
ax.errorbar(x, g["mean"], yerr=g["std"], marker="o", color=COL_LOCAL, lw=2.2,
           markersize=8, capsize=4, label="Local-loss")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_xlabel("Local variables available (of 6)", fontsize=10.5)
ax.set_ylabel("Occurrence F1", fontsize=11)
ax.set_title("(A) Local-loss\n(neighbours held at full)", fontsize=11.5, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.05, "nearly flat", transform=ax.transAxes, ha="center",
       fontsize=9.5, style="italic", color="#444444")

# ── Panel B: neighbour-loss ──────────────────────────────────────────────
ax = axes[1]
order_n = [0, 1, 2]
labels_n = ["0/2", "1/2", "2/2"]
g = nbr_df.groupby("keep_count")["f1"].agg(["mean", "std", "count"]).reindex(order_n)
x = np.arange(len(order_n))
ax.errorbar(x, g["mean"], yerr=g["std"], marker="D", color=COL_NBR, lw=2.2,
           markersize=8, capsize=4, label="Neighbour-loss")
ax.set_xticks(x); ax.set_xticklabels(labels_n, fontsize=10)
ax.set_xlabel("Neighbours available (of 2, k=2 adjacency)", fontsize=10.5)
ax.set_title("(B) Neighbour-loss\n(local held at full)", fontsize=11.5, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.annotate("dominant\ndriver", xy=(0, g["mean"].iloc[0]), xytext=(0.55, 0.35),
           fontsize=9.5, style="italic", color="#444444",
           arrowprops=dict(arrowstyle="->", color="#888888", lw=1.1))

# ── Panel C: joint-loss ──────────────────────────────────────────────────
ax = axes[2]
order_j = ["0L+0N", "3L+1N", "6L+2N"]
labels_j = ["none\n(0L+0N)", "partial\n(3L+1N)", "full\n(6L+2N)"]
g = joint_df.groupby("level")["f1"].agg(["mean", "std", "count"]).reindex(order_j)
x = np.arange(len(order_j))
ax.errorbar(x, g["mean"], yerr=g["std"], marker="^", color=COL_JOINT, lw=2.2,
           markersize=9, capsize=4, label="Joint-loss")
ax.set_xticks(x); ax.set_xticklabels(labels_j, fontsize=10)
ax.set_xlabel("Combined local + neighbour context", fontsize=10.5)
ax.set_title("(C) Joint-loss\n(both degraded together)", fontsize=11.5, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.annotate("total collapse\n(matches NetBlock-30d)", xy=(0, 0.0), xytext=(0.9, 0.28),
           fontsize=9.5, style="italic", color="#444444",
           arrowprops=dict(arrowstyle="->", color="#888888", lw=1.1))

for ax in axes:
    ax.set_ylim(-0.05, 1.0)

fig.suptitle("Graded context-loss ablation: neighbour information dominates, but collapse\n"
             "requires BOTH local and neighbour context to fail together (interaction effect)",
             fontsize=12.5, fontweight="bold")
fig.text(0.5, 0.005,
        "PRECIP mask fixed at MCAR-10 (416 identical masked positions in every panel); error bars = "
        "±1 SD over 3 context-mask seeds × 3 model seeds (n=9/point).\n"
        "Adjacency uses k=2 neighbours per station. Local-loss and neighbour-loss varied independently "
        "before joint-loss.",
        ha="center", fontsize=8.5, style="italic", color="#444444")
fig.tight_layout(rect=[0, 0.05, 1, 0.90])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_11_ContextLossAblation.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 11 (png/pdf/svg)")

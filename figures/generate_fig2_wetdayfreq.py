"""
Figure 2 - Predicted vs. observed wet-day frequency, per method per scenario.
Rebuilt from results/canonical/metrics/canonical_metrics_summary.csv (mean
across seeds for trainable methods; single deterministic run for
Mean/Linear/KNN/MICE), replacing the stale pre-canonical static image.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(REPO, "results", "canonical")
summary = pd.read_csv(os.path.join(CANON, "metrics", "canonical_metrics_summary.csv"))

METHODS = [
    ("Mean", "Mean"), ("Linear", "Linear"), ("KNN", "KNN"), ("MICE", "MICE"),
    ("SAITS", "SAITS"),
    ("WGANGP_raw", "WGAN-GP (raw)"), ("WGANGP_PrecipFix", "WGAN-GP (calibrated)"),
    ("AmountRF_DLPIF", "DLPIF"),
]
SCEN_ORDER = [("10pct", "10% Missing"), ("20pct", "20% Missing"),
             ("block7d", "7-Day Block"), ("block30d", "30-Day Block")]

COLORS = ["#1f77b4", "#c9922a", "#2ca07f", "#b5501e", "#c07bb0", "#a67a5b",
         "#e8a4c8", "#8c8c8c", "#c9c227"]

fig, axes = plt.subplots(1, 4, figsize=(16, 6), dpi=150, sharey=True)

for ax, (scen_key, scen_label) in zip(axes, SCEN_ORDER):
    labels, vals, gt = [], [], None
    for i, (key, label) in enumerate(METHODS):
        row = summary[(summary["method"] == key) & (summary["scenario"] == scen_key)]
        if row.empty:
            continue
        r = row.iloc[0]
        labels.append(label)
        vals.append(r["freq_pred_mean"])
        gt = r["freq_gt_mean"]
    y = range(len(labels))
    ax.barh(y, vals, color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
           edgecolor="black", linewidth=0.6, height=0.7)
    if gt is not None:
        ax.axvline(gt, color="black", ls="--", lw=2, zorder=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Wet-day Frequency", fontsize=11)
    ax.set_title(scen_label, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

from matplotlib.lines import Line2D
fig.legend([Line2D([0], [0], color="black", ls="--", lw=2)], ["Ground Truth"],
          loc="lower center", ncol=1, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.04, 1, 1])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_2_WetDayFrequency.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 2 (png/pdf/svg)")

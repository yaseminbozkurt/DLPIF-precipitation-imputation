"""
Figure 5 - Extreme precipitation (>=95th percentile) MAE and RMSE per
scenario per method. Rebuilt from
results/canonical/metrics/canonical_metrics_summary.csv. The p95 threshold
value in the subtitle is read from canonical_metrics.P95_THRESH directly
(not hand-typed) so it can never drift from the value actually used to
compute mae_p95/rmse_p95.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, "src")
sys.path.insert(0, SRC_DIR)
import canonical_metrics as cm

CANON = os.path.join(REPO, "results", "canonical")
summary = pd.read_csv(os.path.join(CANON, "metrics", "canonical_metrics_summary.csv"))

METHODS = [
    ("Linear", "Linear", "#1f77b4"),
    ("SAITS", "SAITS", "#2ca07f"),
    ("WGANGP_PrecipFix", "WGAN-GP", "#8c564b"),
    ("AmountRF_DLPIF", "DLPIF", "#c77bb5"),
]
SCEN_ORDER = ["10pct", "20pct", "block7d", "block30d"]
SCEN_LABELS = ["10%", "20%", "7d Block", "30d Block"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
x = np.arange(len(SCEN_ORDER))
width = 0.8 / len(METHODS)

for ax, col, title in zip(axes, ["mae_p95_mean", "rmse_p95_mean"],
                           ["(a) Extreme Precipitation MAE", "(b) Extreme Precipitation RMSE"]):
    for i, (key, label, color) in enumerate(METHODS):
        vals = []
        for scen in SCEN_ORDER:
            row = summary[(summary["method"] == key) & (summary["scenario"] == scen)]
            vals.append(row.iloc[0][col] if not row.empty else 0.0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=label,
              color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SCEN_LABELS, fontsize=11)
    ax.set_title(f"{title}\n(Events ≥ 95th percentile ({cm.P95_THRESH:.2f} mm/day))",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("MAE (mm/day)" if "mae" in col else "RMSE (mm/day)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

axes[1].legend(title="Method", fontsize=10, title_fontsize=10, loc="upper left",
              bbox_to_anchor=(1.02, 1.0))
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_5_ExtremeEvents.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 5 (png/pdf/svg)")

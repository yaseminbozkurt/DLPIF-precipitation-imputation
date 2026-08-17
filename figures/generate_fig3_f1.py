"""
Figure 3 - Wet-day F1 across missingness scenarios, per method (mean +/- std
across seeds for trainable methods). Rebuilt from
results/canonical/metrics/canonical_metrics_summary.csv.
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
    ("Linear", "Linear", "#1f77b4", "o", "-"),
    ("SAITS", "SAITS", "#2ca07f", "^", "-"),
    ("WGANGP_PrecipFix", "WGAN-GP", "#8c564b", "s", "-"),
    ("AmountRF_DLPIF", "DLPIF", "#c9422a", "D", "--"),
]
SCEN_ORDER = ["10pct", "20pct", "block7d", "block30d"]
SCEN_LABELS = ["10%", "20%", "7d Block", "30d Block"]

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
x = range(len(SCEN_ORDER))
for key, label, color, marker, ls in METHODS:
    means, stds = [], []
    for scen in SCEN_ORDER:
        row = summary[(summary["method"] == key) & (summary["scenario"] == scen)]
        means.append(row.iloc[0]["f1_mean"] if not row.empty else float("nan"))
        stds.append(row.iloc[0]["f1_std"] if not row.empty else 0.0)
    ax.errorbar(x, means, yerr=stds, label=label, color=color, marker=marker,
               ls=ls, lw=2.2, markersize=9, capsize=4)

ax.set_xticks(list(x))
ax.set_xticklabels(SCEN_LABELS, fontsize=12)
ax.set_xlabel("Missingness Scenario", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_ylim(0.3, 1.0)
ax.legend(title="Method", fontsize=11, title_fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_3_F1_Performance.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 3 (png/pdf/svg)")

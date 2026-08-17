"""
Figure 4 - RMSE-F1 trade-off: continuous accuracy vs. event-level
reconstruction, one panel per scenario, error bars = +/-1 std across seeds
(42, 123, 456) for trainable methods. Rebuilt from
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

# Light/white background to match every other figure in the manuscript and
# standard academic print conventions (this panel previously used
# dark_background, which looked inconsistent alongside Figures 2/3/5/6/7).
METHODS = [
    ("Mean", "Mean", "#9a9a9a", "s"),
    ("Linear", "Linear", "#7f9fbf", "^"),
    ("KNN", "KNN", "#9a9a9a", "D"),
    ("MICE", "MICE", "#9a9a9a", "P"),
    ("SAITS", "SAITS", "#3fae7a", "h"),
    ("WGANGP_raw", "WGAN-GP (raw)", "#c0392b", "X"),
    ("WGANGP_PrecipFix", "WGAN-GP (calib.)", "#8c564b", "v"),
    ("AmountRF_DLPIF", "DLPIF", "#2ecc71", "*"),
]
SCEN_PANELS = [("10pct", "(a)  Random 10%"), ("20pct", "(b)  Random 20%"),
              ("block7d", "(c)  Block 7d"), ("block30d", "(d)  Block 30d")]

fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=150)

for ax, (scen, title) in zip(axes.flat, SCEN_PANELS):
    xs, ys = [], []
    for key, label, color, marker in METHODS:
        row = summary[(summary["method"] == key) & (summary["scenario"] == scen)]
        if row.empty:
            continue
        r = row.iloc[0]
        x, xerr = r["rmse_wet_mean"], r["rmse_wet_std"]
        y, yerr = r["f1_mean"], r["f1_std"]
        xs.append(x); ys.append(y)
        is_dlpif = key == "AmountRF_DLPIF"
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker=marker, color=color,
                   markersize=16 if is_dlpif else 10,
                   markeredgecolor="black" if is_dlpif else "none",
                   markeredgewidth=1.3 if is_dlpif else 0,
                   capsize=3, lw=1.3, ls="none")
        ax.annotate(label, (x, y), textcoords="offset points",
                   xytext=(8, 6), fontsize=9,
                   color="#1a7a3f" if is_dlpif else "black",
                   fontweight="bold" if is_dlpif else "normal")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel("Wet-day RMSE (mm)", fontsize=11)
    ax.set_ylabel("F1 (wet-day occurrence)", fontsize=11)
    # Auto-scaled with a fixed margin (not hardcoded to the old dataset's
    # value range, which silently clipped points when the underlying data
    # changed) -- margin leaves room for the offset annotation labels.
    x_margin = 0.15 * (max(xs) - min(xs) + 1e-9)
    y_margin = 0.15 * (max(ys) - min(ys) + 1e-9)
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
    ax.grid(True, alpha=0.25)

fig.suptitle("RMSE-F1 Trade-off: Continuous Accuracy vs. Event-Level Reconstruction",
            fontsize=15, fontweight="bold")
fig.text(0.5, 0.955,
        "Error bars show ±1 std across seeds 42, 123, 456  ·  "
        "Methods in upper-left quadrant achieve both low RMSE and high F1",
        ha="center", fontsize=10, color="#555555")
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_4_RMSE_F1_Tradeoff.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None,
               facecolor=fig.get_facecolor())
plt.close(fig)
print("Saved Figure 4 (png/pdf/svg)")

"""
Figure 8 - Reliability diagrams (Raw RF vs selected Platt calibration) for
four dose-response scenarios: MCAR-10, MNAR-Wet, MNAR-Intensity-Moderate,
MNAR-Intensity-Severe. Rebuilt from results/rq2_calibration/reliability_bins.csv
(10 fixed equal-width probability bins, count-weighted mean across the 3
occurrence-RF seeds within each bin).

Companion to Figure 9 (F1-ECE divergence). Raw/Platt/Isotonic three-way
comparison is Figure 8b (supplementary) -- kept out of the main figure to
avoid clutter, per the two-curve-per-panel convention used elsewhere in
this figure set.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RQ2 = os.path.join(REPO, "results", "rq2_calibration")
bins_df = pd.read_csv(os.path.join(RQ2, "reliability_bins.csv"))
metrics_df = pd.read_csv(os.path.join(RQ2, "test_calibration_metrics.csv"))

# Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008) -- no palette
# validator script was runnable in this environment (no node.js), so this
# well-established reference set is used directly rather than eyeballing a
# custom one.
COL_RAW = "#56B4E9"     # sky blue
COL_PLATT = "#D55E00"   # vermillion
COL_ISO = "#009E73"     # bluish green
COL_DIAG = "#333333"

PANELS = [
    ("TEST:10pct", "MCAR-10"),
    ("TEST:mnar_wet", "MNAR-Wet"),
    ("TEST:mnar_intensity_moderate", "MNAR-Intensity-Moderate"),
    ("TEST:mnar_intensity_severe", "MNAR-Intensity-Severe"),
]


def weighted_curve(split, variant):
    sub = bins_df[(bins_df["split"] == split) & (bins_df["variant"] == variant)]
    sub = sub.dropna(subset=["mean_pred", "obs_freq"])
    agg = sub.groupby(["bin_lo", "bin_hi"]).apply(
        lambda g: pd.Series({
            "mean_pred": np.average(g["mean_pred"], weights=g["count"]),
            "obs_freq": np.average(g["obs_freq"], weights=g["count"]),
            "count": g["count"].sum(),
        }),
        include_groups=False,
    ).reset_index().sort_values("bin_lo")
    return agg


def ece_for(scenario, variant):
    row = metrics_df[(metrics_df["scenario"] == scenario) & (metrics_df["variant"] == variant)]
    return float(row["ece"].mean())


fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=150)
for ax, (split, title) in zip(axes.flat, PANELS):
    scenario = split.replace("TEST:", "")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=COL_DIAG, label="Perfect calibration")

    for variant, color, marker in [("raw", COL_RAW, "o"), ("platt", COL_PLATT, "D")]:
        curve = weighted_curve(split, variant)
        ece = ece_for(scenario, variant)
        label = f"{'Raw RF' if variant == 'raw' else 'Platt (selected)'}  (ECE={ece:.3f})"
        ax.plot(curve["mean_pred"], curve["obs_freq"], marker=marker, color=color,
                lw=2.0, markersize=7, label=label)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean predicted wet probability", fontsize=10)
    ax.set_ylabel("Observed wet frequency", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=1.0)

fig.suptitle("Stage-1 occurrence probability reliability under increasing MNAR severity",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_8_ReliabilityDiagram.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 8 (png/pdf/svg)")

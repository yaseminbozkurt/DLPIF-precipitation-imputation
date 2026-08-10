"""
Figure 9 - F1 vs ECE divergence across all 8 test scenarios, selected
(Platt) calibration variant, mean over 3 occurrence-RF seeds. Rebuilt from
results/rq2_calibration/test_calibration_metrics.csv.

The headline plot for RQ1+RQ2 combined: apparent occurrence skill (F1, x)
does not track probabilistic reliability (ECE, y). The MCAR -> MNAR-
Intensity-Moderate -> MNAR-Intensity-Severe dose-response triplet is
connected with a thin guide line to make the joint F1-up/ECE-up trend
visible as a single trajectory rather than three unrelated points.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RQ2 = os.path.join(REPO, "results", "rq2_calibration")
df = pd.read_csv(os.path.join(RQ2, "test_calibration_metrics.csv"))
sel = df[df["selected_calibration"]]
agg = sel.groupby("scenario")[["f1", "ece"]].mean().reset_index()

# Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008); fixed hue per
# scenario family (MCAR / MAR / MNAR-Wet / MNAR-Intensity dose pair /
# context-loss block scenarios), not per individual scenario, so the
# family grouping reads directly off color.
SCENARIOS = [
    # (key, display label, color, marker)
    ("10pct",                    "MCAR-10",                 "#0072B2", "o"),
    ("mar_meteo",                "MAR-Meteo",                "#009E73", "o"),
    ("mnar_wet",                 "MNAR-Wet",                 "#D55E00", "o"),
    ("mnar_intensity_moderate",  "MNAR-Intensity-Moderate",  "#CC79A7", "D"),
    ("mnar_intensity_severe",    "MNAR-Intensity-Severe",    "#CC79A7", "^"),
    ("block7d",                  "Block-7d",                 "#56B4E9", "s"),
    ("block30d",                 "Block-30d",                "#56B4E9", "P"),
    ("netblock30d",              "NetBlock-30d (collapse)",  "#000000", "X"),
]

fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

# Dose-response guide line: MCAR -> MNAR-Intensity-Moderate -> MNAR-Intensity-Severe
dose_keys = ["10pct", "mnar_intensity_moderate", "mnar_intensity_severe"]
dose_pts = agg.set_index("scenario").loc[dose_keys]
ax.plot(dose_pts["f1"], dose_pts["ece"], color="#999999", lw=1.6, ls="-", zorder=1)

# Custom per-point label offsets (points) -- MCAR-10/MAR-Meteo/Block-7d/
# Block-30d sit almost on top of each other in the low-F1-error corner, so
# a uniform offset would collide; fan them out instead.
LABEL_OFFSETS = {
    "10pct": (10, 20),
    "mar_meteo": (75, 8),
    "block30d": (-95, -6),
    "block7d": (-70, -14),
    "mnar_wet": (10, 8),
    "mnar_intensity_moderate": (12, 6),
    "mnar_intensity_severe": (-14, 12),
    "netblock30d": (12, 8),
}

for key, label, color, marker in SCENARIOS:
    row = agg[agg["scenario"] == key]
    if row.empty:
        continue
    x, y = float(row["f1"].iloc[0]), float(row["ece"].iloc[0])
    ax.scatter(x, y, s=140, color=color, marker=marker, edgecolor="white",
              linewidth=1.2, zorder=3, label=label)
    ax.annotate(label, (x, y), textcoords="offset points",
               xytext=LABEL_OFFSETS.get(key, (8, 6)), fontsize=8.5)

ax.set_xlabel("Occurrence F1 (higher = better classification skill)", fontsize=12)
ax.set_ylabel("Expected Calibration Error (lower = better reliability)", fontsize=12)
ax.set_title("Apparent classification skill vs probabilistic reliability\n"
             "(Platt-calibrated Stage-1 occurrence probabilities, mean over 3 seeds)",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.annotate("", xy=(0.97, 0.03), xytext=(0.75, 0.25), xycoords="axes fraction",
           arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))
ax.text(0.97, 0.02, "ideal region", transform=ax.transAxes, fontsize=9,
       color="#888888", ha="right", style="italic")
ax.margins(y=0.10)
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_9_F1_ECE_Divergence.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 9 (png/pdf/svg)")

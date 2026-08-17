"""
Figure 7 - Zoomed empirical near-zero precipitation distribution (0-3 mm/day)
over artificially masked PRECIP positions under the 10% random missingness
scenario (seed 42, the primary realization used throughout the manuscript).
Rebuilt directly from results/canonical/predictions/*.csv.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(REPO, "results", "canonical")
PRED_DIR = os.path.join(CANON, "predictions")

SCEN = "10pct"
SEED = "42"
BINS = np.arange(0, 3.01, 0.2)

METHODS = [
    ("SAITS", "SAITS", "#2ca07f", "deterministic-or-seed"),
    ("WGANGP_raw", "WGAN-GP (raw)", "#c0392b", "deterministic-or-seed"),
    ("WGANGP_PrecipFix", "WGAN-GP (calib.)", "#8c564b", "deterministic-or-seed"),
    ("AmountRF_DLPIF", "DLPIF (Ours)", "#1f77b4", "deterministic-or-seed"),
]

def load_col(method, col):
    df = pd.read_csv(os.path.join(PRED_DIR, f"{method}.csv"), dtype={"seed": str})
    sub = df[df["scenario"] == SCEN]
    if sub["seed"].iloc[0] != "deterministic":
        sub = sub[sub["seed"] == SEED]
    return sub[col].to_numpy()

gt = load_col("AmountRF_DLPIF", "y_true")

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
# range=(0,3) (not np.clip) so values >3mm are simply excluded rather than
# piling up into the final bin as a false edge spike
ax.hist(gt, bins=BINS, range=(0, 3), density=True, histtype="step",
       color="black", lw=2.2, label="Ground Truth")

for key, label, color, _ in METHODS:
    pred = load_col(key, "y_pred")
    ax.hist(pred, bins=BINS, range=(0, 3), density=True, histtype="step",
           color=color, lw=2.0, label=label)

ax.set_xlim(0, 3)
ax.set_xlabel("Precipitation (mm/day)", fontsize=12)
ax.set_ylabel("Probability", fontsize=12)
ax.set_title("Precipitation Distribution Comparison (Near-Zero Zoom)",
            fontsize=13, fontweight="bold")
ax.legend(title="Method", fontsize=11, title_fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(out_dir, f"Figure_7_DistributionComparison.{ext}"),
               bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("Saved Figure 7 (png/pdf/svg)")

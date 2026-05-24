import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "figures/EMS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PREFIX = "Figure_6_TimeSeries"

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["font.size"] = 11

def inverse_scale(scaler_path, array_norm):
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    sc = scaler_data['scaler']
    mvars = scaler_data['meteo_vars']
    pidx = list(mvars).index('PRECIP')
    
    arr_scaled = np.nan_to_num(array_norm, nan=0.0)
    arr_mm = sc.inverse_transform(arr_scaled)[:, pidx]
    return np.clip(arr_mm, 0, None)

# Load data
test_npz = np.load('src/preprocessed_test.npz')
gt_norm = test_npz['data']
mask_10pct = test_npz['art_mask_10pct']

try:
    with open('src/scaler.pkl', 'rb') as f:
        scaler_data = pickle.load(f)
    mvars = scaler_data['meteo_vars']
    pidx = list(mvars).index('PRECIP')
except:
    pidx = 0

gt_mm = inverse_scale('src/scaler.pkl', gt_norm)
mask_precip = mask_10pct[:, pidx] > 0.5

wgan_raw_norm = np.load('src/gan_imputed_test_modeB_seed42.npy')
wgan_raw_mm = inverse_scale('src/scaler.pkl', wgan_raw_norm)

saits_norm = np.load('src/results_dl/saits_v2/saits_imputed_test_10pct_v2_seed42_flat.npy')
# SAITS was trained with log1p normalization!
# We need to apply expm1 to the scaled back value.
try:
    with open('src/results_dl/saits_v2/scaler_v2.pkl', 'rb') as f:
        sc_saits = pickle.load(f)
    saits_unscaled = sc_saits.inverse_transform(np.nan_to_num(saits_norm, nan=0.0))[:, pidx]
    saits_mm = np.expm1(saits_unscaled)
    saits_mm = np.clip(saits_mm, 0, None)
except:
    # fallback
    saits_mm = inverse_scale('src/scaler.pkl', saits_norm)

dlpif_norm = np.load('src/gan_imputed_test_modeB_seed42_msclean_amountrf.npy')
dlpif_mm = inverse_scale('src/scaler.pkl', dlpif_norm)

min_len = min(len(gt_mm), len(wgan_raw_mm), len(saits_mm), len(dlpif_mm))
gt_mm = gt_mm[:min_len]
wgan_raw_mm = wgan_raw_mm[:min_len]
saits_mm = saits_mm[:min_len]
dlpif_mm = dlpif_mm[:min_len]
mask_precip = mask_precip[:min_len]

# Window index 10785 was selected from candidates produced by
# `select_diagnostic_window.py` for qualitative diagnostic visualisation only.
# Criteria: sufficient dry/wet contrast, presence of high-intensity events,
# and visible continuous-baseline drizzle artefacts in the raw WGAN-GP output.
# This window is NOT used for any quantitative result; all reported metrics
# are computed over the full masked test set.
start = 10785
W = 50
print(f"Selected Window Start: {start}")

time_idx = np.arange(W)
fig, ax = plt.subplots(figsize=(10, 4.5))

# Plot lines
# Masked GT = GT, but only plotted where mask_precip == False (not masked)
# Wait, mask_precip == True means it is artificially masked (missing).
# So "Masked GT" in the plot usually means the observed GT that is given to the model,
# and we also want to show the true GT values that were hidden.
# The user said: GT observed, masked GT. 
# Usually, GT observed = the true sequence. Masked GT = the points that are given as input.
# Or vice versa: "Observed" = the ones given, "Masked GT" = the hidden ones we try to predict.
# Let's plot GT as a solid black line, and add markers for Masked GT (points that were artificial).

ax.plot(time_idx, gt_mm[start:start+W], color="black", linestyle="-", linewidth=2.5, label="GT Observed")

# Masked GT (the missing points we evaluate on)
missing_idx = time_idx[mask_precip[start:start+W]]
ax.scatter(missing_idx, gt_mm[start+missing_idx], color="gray", marker="o", s=40, zorder=5, label="Masked GT")

ax.plot(time_idx, wgan_raw_mm[start:start+W], color="#d55e00", linestyle="--", linewidth=2, label="WGAN-GP (raw)")
ax.plot(time_idx, saits_mm[start:start+W], color="#009e73", linestyle="-.", linewidth=2, label="SAITS")
ax.plot(time_idx, dlpif_mm[start:start+W], color="#0072b2", linestyle="-", linewidth=2.5, label="DLPIF (Ours)")

ax.set_ylabel("Precipitation (mm/day)")
ax.set_xlabel("Time Steps (Days)")
ax.set_xlim(0, W-1)

# Clean legend
ax.legend(loc='upper right', frameon=False, ncol=3, fontsize=10)
sns.despine(ax=ax)

plt.tight_layout()

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.{ext}")
    plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()

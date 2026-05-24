import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "figures/EMS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PREFIX = "Figure_7_DistributionComparison"

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
try:
    with open('src/results_dl/saits_v2/scaler_v2.pkl', 'rb') as f:
        sc_saits = pickle.load(f)
    saits_unscaled = sc_saits.inverse_transform(np.nan_to_num(saits_norm, nan=0.0))[:, pidx]
    saits_mm = np.expm1(saits_unscaled)
    saits_mm = np.clip(saits_mm, 0, None)
except:
    saits_mm = inverse_scale('src/scaler.pkl', saits_norm)

dlpif_norm = np.load('src/gan_imputed_test_modeB_seed42_msclean_amountrf.npy')
dlpif_mm = inverse_scale('src/scaler.pkl', dlpif_norm)

min_len = min(len(gt_mm), len(wgan_raw_mm), len(saits_mm), len(dlpif_mm))

gt_mm = gt_mm[:min_len]
wgan_raw_mm = wgan_raw_mm[:min_len]
saits_mm = saits_mm[:min_len]
dlpif_mm = dlpif_mm[:min_len]
mask_precip = mask_precip[:min_len]

# Filter by mask
gt_masked = gt_mm[mask_precip]
wgan_masked = wgan_raw_mm[mask_precip]
saits_masked = saits_mm[mask_precip]
dlpif_masked = dlpif_mm[mask_precip]

# Prepare DataFrame for seaborn
df_list = []
df_list.append(pd.DataFrame({'Precipitation (mm/day)': gt_masked, 'Method': 'Ground Truth'}))
df_list.append(pd.DataFrame({'Precipitation (mm/day)': wgan_masked, 'Method': 'WGAN-GP (raw)'}))
df_list.append(pd.DataFrame({'Precipitation (mm/day)': saits_masked, 'Method': 'SAITS'}))
df_list.append(pd.DataFrame({'Precipitation (mm/day)': dlpif_masked, 'Method': 'DLPIF (Ours)'}))

df = pd.concat(df_list, ignore_index=True)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

palette = {"Ground Truth": "black", "WGAN-GP (raw)": "#d55e00", "SAITS": "#009e73", "DLPIF (Ours)": "#0072b2"}

# We will use a zoomed histogram for the near-zero region
sns.histplot(
    data=df, x="Precipitation (mm/day)", hue="Method",
    element="step", fill=False, stat="probability", common_norm=False,
    bins=30, binrange=(0, 3), palette=palette, ax=ax,
    linewidth=2.5, alpha=0.9
)

ax.set_xlim(-0.1, 3.0)
# Dynamic ylim based on data, linear scale
ax.set_xlabel("Precipitation (mm/day)")
ax.set_ylabel("Probability")
ax.set_title("Precipitation Distribution Comparison (Near-Zero Zoom)", fontweight='bold')

sns.despine(ax=ax)
plt.tight_layout()

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.{ext}")
    plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()

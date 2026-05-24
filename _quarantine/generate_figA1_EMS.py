"""
_quarantine/generate_figA1_EMS.py
==================================
NON-SUBMITTED EXPLORATORY SCRIPT — NOT PART OF THE OFFICIAL PIPELINE.

This script produces Figure A1 (threshold sensitivity analysis), which is an
exploratory/supplementary figure that was NOT submitted as part of the
manuscript.  It is retained here for reproducibility of the exploratory
analysis only.

Output files (Figure_A1_ThresholdSensitivity.{png,pdf,svg}) are stored in
figures/not_submitted/ and must NOT be cited or referenced as submitted figures.

Do NOT list this script in the official figure-generation instructions in
README.md.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

OUTPUT_DIR = "figures/EMS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PREFIX = "Figure_A1_ThresholdSensitivity"

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
tr = np.load('src/preprocessed_train.npz')
te = np.load('src/preprocessed_test.npz')

try:
    with open('src/scaler.pkl', 'rb') as f:
        scaler_data = pickle.load(f)
    mvars = scaler_data['meteo_vars']
    pidx = list(mvars).index('PRECIP')
except:
    pidx = 0

tr_corrupt = tr['corrupted_10pct']
te_corrupt = te['corrupted_10pct']
te_gt = te['data']
mask_10pct = te['art_mask_10pct'][:, pidx] > 0.5

# Impute baselines
print("Running mean imputation...")
imp_mean = SimpleImputer(strategy='mean').fit(tr_corrupt)
pred_mean_norm = imp_mean.transform(te_corrupt)

print("Running linear interpolation...")
pred_linear_norm = pd.DataFrame(te_corrupt).interpolate(method='linear', limit_direction='both').values

print("Running KNN imputation...")
# To speed up, we fit KNN on a subset of train (2000 rows)
idx = np.random.choice(len(tr_corrupt), min(2000, len(tr_corrupt)), replace=False)
imp_knn = KNNImputer(n_neighbors=5).fit(tr_corrupt[idx])
pred_knn_norm = imp_knn.transform(te_corrupt)

print("Running MICE imputation...")
imp_mice = IterativeImputer(max_iter=2, random_state=42, tol=0.01).fit(tr_corrupt[idx])
pred_mice_norm = imp_mice.transform(te_corrupt)

# DL models
wgan_raw_norm = np.load('src/gan_imputed_test_modeB_seed42.npy')
saits_norm = np.load('src/results_dl/saits_v2/saits_imputed_test_10pct_v2_seed42_flat.npy')
dlpif_norm = np.load('src/gan_imputed_test_modeB_seed42_msclean_amountrf.npy')

min_len = min(len(te_gt), len(wgan_raw_norm), len(saits_norm), len(dlpif_norm))

preds_mm = {
    'Mean': inverse_scale('src/scaler.pkl', pred_mean_norm)[:min_len],
    'Linear': inverse_scale('src/scaler.pkl', pred_linear_norm)[:min_len],
    'KNN': inverse_scale('src/scaler.pkl', pred_knn_norm)[:min_len],
    'MICE': inverse_scale('src/scaler.pkl', pred_mice_norm)[:min_len],
    'WGAN-GP raw': inverse_scale('src/scaler.pkl', wgan_raw_norm)[:min_len],
    'DLPIF (Ours)': inverse_scale('src/scaler.pkl', dlpif_norm)[:min_len]
}

# SAITS special scale
try:
    with open('src/results_dl/saits_v2/scaler_v2.pkl', 'rb') as f:
        sc_saits = pickle.load(f)
    saits_unscaled = sc_saits.inverse_transform(np.nan_to_num(saits_norm, nan=0.0))[:, pidx]
    saits_mm = np.expm1(saits_unscaled)
    preds_mm['SAITS'] = np.clip(saits_mm, 0, None)[:min_len]
except:
    preds_mm['SAITS'] = inverse_scale('src/scaler.pkl', saits_norm)[:min_len]

gt_mm = inverse_scale('src/scaler.pkl', te_gt)[:min_len]
mask = mask_10pct[:min_len]

# Filter to mask only
gt_m = gt_mm[mask]
methods_m = {k: v[mask] for k, v in preds_mm.items()}

# Thresholds
thresholds = [0.1, 0.2, 0.5, 1.0]

records = []
for tau in thresholds:
    gt_wet = gt_m >= tau
    for m, pred in methods_m.items():
        pred_wet = pred >= tau
        
        tp = (gt_wet & pred_wet).sum()
        fp = (~gt_wet & pred_wet).sum()
        fn = (gt_wet & ~pred_wet).sum()
        
        f1 = 2 * tp / (2 * tp + fp + fn) if (2*tp+fp+fn) > 0 else 0
        csi = tp / (tp + fp + fn) if (tp+fp+fn) > 0 else 0
        
        freq_gt = gt_wet.mean()
        freq_pred = pred_wet.mean()
        bias = abs(freq_pred - freq_gt)
        
        records.append({'Method': m, 'Threshold': tau, 'F1': f1, 'CSI': csi, 'Bias': bias})

df = pd.DataFrame(records)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
methods = ['Mean', 'Linear', 'KNN', 'MICE', 'WGAN-GP raw', 'SAITS', 'DLPIF (Ours)']
palette = sns.color_palette("colorblind", len(methods))
markers = ["o", "s", "^", "D", "P", "X", "*"]

sns.lineplot(data=df, x='Threshold', y='F1', hue='Method', style='Method', markers=markers, dashes=False, palette=palette, ax=axes[0], linewidth=2.5, markersize=8)
axes[0].set_title("(a) F1 Score", fontweight='bold')
axes[0].set_ylabel("F1 Score")
axes[0].set_xlabel(r"Threshold $\tau$ (mm/day)")
axes[0].legend_.remove()

sns.lineplot(data=df, x='Threshold', y='CSI', hue='Method', style='Method', markers=markers, dashes=False, palette=palette, ax=axes[1], linewidth=2.5, markersize=8)
axes[1].set_title("(b) Critical Success Index (CSI)", fontweight='bold')
axes[1].set_ylabel("CSI")
axes[1].set_xlabel(r"Threshold $\tau$ (mm/day)")
axes[1].legend_.remove()

sns.lineplot(data=df, x='Threshold', y='Bias', hue='Method', style='Method', markers=markers, dashes=False, palette=palette, ax=axes[2], linewidth=2.5, markersize=8)
axes[2].set_title("(c) Wet-Day Frequency Bias", fontweight='bold')
axes[2].set_ylabel("Absolute Wet-Day Frequency Bias")
axes[2].set_xlabel(r"Threshold $\tau$ (mm/day)")
axes[2].axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=0)

handles, labels = axes[2].get_legend_handles_labels()
axes[2].legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, title="Imputation Method", fontsize=9, title_fontsize=10)

for ax in axes:
    ax.set_xticks(thresholds)
    sns.despine(ax=ax)

plt.tight_layout()

for ext in ["png", "pdf", "svg"]:
    out_path = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.{ext}")
    plt.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()

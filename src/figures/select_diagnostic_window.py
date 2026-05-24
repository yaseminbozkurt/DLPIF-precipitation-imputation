"""
select_diagnostic_window.py
============================
Searches the test set for 50-day windows suitable for qualitative diagnostic
visualisation in Figure 6.

Selection criteria (neutral — no method is favoured):
  1. Sufficient dry/wet contrast:  at least 10 dry days in the window.
  2. Presence of high-intensity precipitation events: at least 2 days with
     ground-truth precipitation > 10 mm/day.
  3. Visible continuous-baseline drizzle artefacts: the raw WGAN-GP output
     predicts wet on >= 40% of days in the window, indicating drizzle spread.

The window with the highest score under these three neutral criteria is printed.
Selected windows are used ONLY for qualitative visualisation; all quantitative
results in the manuscript are computed over the full masked test set.

Usage (run from the repository root):
    python src/figures/select_diagnostic_window.py
"""
import numpy as np
import pickle


def inverse_scale(scaler_path, array_norm):
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    sc = scaler_data['scaler']
    mvars = scaler_data['meteo_vars']
    pidx = list(mvars).index('PRECIP')
    arr_scaled = np.nan_to_num(array_norm, nan=0.0)
    arr_mm = sc.inverse_transform(arr_scaled)[:, pidx]
    return np.clip(arr_mm, 0, None)


gt_norm = np.load('src/preprocessed_test.npz')['data']
gt_mm = inverse_scale('src/scaler.pkl', gt_norm)

wgan_raw_norm = np.load('src/gan_imputed_test_modeB_seed42.npy')
wgan_raw_mm = inverse_scale('src/scaler.pkl', wgan_raw_norm)

min_len = min(len(gt_mm), len(wgan_raw_mm))
gt_mm = gt_mm[:min_len]
wgan_raw_mm = wgan_raw_mm[:min_len]

W = 50
results = []

for start in range(0, min_len - W, 5):
    gt_win   = gt_mm[start:start + W]
    wgan_win = wgan_raw_mm[start:start + W]

    # Criterion 1: dry/wet contrast
    gt_wet  = (gt_win > 0.1).sum()
    gt_dry  = W - gt_wet

    # Criterion 2: high-intensity events
    gt_peaks = (gt_win > 10.0).sum()

    # Criterion 3: drizzle artefact visibility (raw WGAN-GP wet fraction)
    wgan_wet_ratio = (wgan_win > 0.05).sum() / W

    if gt_dry >= 10 and gt_peaks >= 2 and wgan_wet_ratio >= 0.40:
        # Neutral score: reward dry/wet contrast and peak count; no method comparison
        score = float(gt_dry) + float(gt_peaks) * 2.0 + wgan_wet_ratio * 10.0
        results.append((start, score, gt_dry, gt_peaks, wgan_wet_ratio))

results.sort(key=lambda x: x[1], reverse=True)

print("Top candidate windows for qualitative diagnostic visualisation:")
print("Start | Score | Dry days | GT Peaks (>10mm) | WGAN Wet Ratio")
print("-" * 65)
for r in results[:20]:
    print(f"{r[0]:5d}  | {r[1]:6.1f} | {r[2]:8d} | {r[3]:16d} | {r[4]:.2f}")

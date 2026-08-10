"""
evaluate_baseline_precip_occurrence.py
========================================
PRECIP-only wet/dry occurrence + magnitude metrics for the classical
baselines (Mean, Linear, KNN, MICE) at every test scenario, computed
through the SAME canonical_metrics.metrics_row() formula already used for
DirectTwoStageRF (results/direct_two_stage_rf_evaluation.csv), Single-Stage
RF (results/single_stage_rf_evaluation.csv), and SAITS
(results_dl/saits_v2/evaluation_v2_seed*.csv) -- so all methods are directly
comparable, row for row, in the RQ1 mechanism-sensitivity table.

03_baseline_imputation.py only reports a macro RMSE/MAE averaged across all
7 meteorological variables (baseline_rmse.csv); it never isolates PRECIP nor
computes wet-day classification metrics (F1/CSI/precision/recall/bias) or
the extreme (p95) RMSE. This script fills that gap without re-running or
modifying 03_baseline_imputation.py -- it only reads its already-saved
baseline_results.pkl (all_scenarios: (method, scenario) -> imputed (N,7)
normalised array) plus preprocessed_test.npz for ground truth and masks.

Also adds false_wet_rate = FP / (FP + TN): the fraction of truly-dry masked
positions the method predicts wet, matching the concept already tracked in
results/analysis_false_wet_rate.csv for the original scenarios.

Output:
  results/baseline_precip_occurrence_metrics.csv
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
import canonical_metrics as cm
import direct_two_stage_rf as d2s

RESULTS_DIR = os.path.join(SRC_DIR, '..', 'results')
METHODS = ['mean', 'linear', 'knn', 'mice']


def main():
    print('=' * 70)
    print('  BASELINE (Mean/Linear/KNN/MICE) PRECIP OCCURRENCE + MAGNITUDE METRICS')
    print(f'  (canonical_metrics.metrics_row(), WET_THRESH={cm.WET_THRESH}, '
          f'P95_THRESH={cm.P95_THRESH} -- same formula as DLPIF/SingleStageRF/SAITS)')
    print('=' * 70)

    with open(os.path.join(SRC_DIR, 'scaler.pkl'), 'rb') as f:
        sc_data = pickle.load(f)
    scaler = sc_data['scaler']
    meteo_vars = list(sc_data['meteo_vars'])
    precip_idx = meteo_vars.index('PRECIP')

    te = np.load(os.path.join(SRC_DIR, 'preprocessed_test.npz'), allow_pickle=True)
    gt_norm = te['data'].astype(np.float32)
    gt_mm_full = scaler.inverse_transform(gt_norm)[:, precip_idx]

    with open(os.path.join(SRC_DIR, 'baseline_results.pkl'), 'rb') as f:
        bl_data = pickle.load(f)
    all_scenarios = bl_data['all_scenarios']

    scenario_labels = [label for label, _, _ in d2s.SCENARIOS]

    rows = []
    for method in METHODS:
        for scen in scenario_labels:
            key = (method, scen)
            if key not in all_scenarios:
                print(f'  [SKIP] {method} x {scen}: not found in baseline_results.pkl')
                continue
            mask_key = f'art_mask_{scen}'
            if mask_key not in te.files:
                print(f'  [SKIP] {method} x {scen}: {mask_key} not found in preprocessed_test.npz')
                continue

            pred_norm = np.clip(all_scenarios[key].astype(np.float32), 0, 1)
            pred_mm_full = scaler.inverse_transform(pred_norm)[:, precip_idx]

            mask = te[mask_key].astype(np.float32)[:, precip_idx] > 0.5
            gt_m = gt_mm_full[mask]
            pred_m = pred_mm_full[mask]

            row = cm.metrics_row(gt_m, pred_m, method=method, scenario=scen,
                                  seed='deterministic', n_evaluated=int(mask.sum()),
                                  source_prediction_file='baseline_results.pkl')
            fp, tn = row['fp'], row['tn']
            row['false_wet_rate'] = round(fp / (fp + tn), 4) if (fp + tn) > 0 else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'baseline_precip_occurrence_metrics.csv')
    df.to_csv(out_path, index=False)
    print(f'\n  -> {out_path}  ({len(df)} rows: {len(METHODS)} methods x '
          f'{len(scenario_labels)} scenarios)')

    show_cols = ['method', 'scenario', 'n_evaluated', 'signed_frequency_bias',
                 'precision', 'recall', 'f1', 'csi', 'false_wet_rate',
                 'rmse_wet', 'rmse_p95']
    print('\n  RQ1-relevant scenarios (10pct / mar_meteo / mnar_wet / mnar_intensity_*):')
    rq1_scens = ['10pct', 'mar_meteo', 'mnar_wet',
                 'mnar_intensity_moderate', 'mnar_intensity_severe']
    show = df[df['scenario'].isin(rq1_scens)].copy()
    show['scenario'] = pd.Categorical(show['scenario'], categories=rq1_scens, ordered=True)
    show = show.sort_values(['scenario', 'method'])
    print(show[show_cols].to_string(index=False))
    print('=' * 70)


if __name__ == '__main__':
    main()

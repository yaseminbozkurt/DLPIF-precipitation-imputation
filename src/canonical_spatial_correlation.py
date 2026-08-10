# -*- coding: utf-8 -*-
"""
canonical_spatial_correlation.py
==================================
Inter-station spatial correlation preservation analysis (manuscript
Table 8), computed exclusively from preprocessed_test.npz +
results/canonical/predictions/*.csv via
build_canonical_outputs.build_filled_series() -- the canonical successor
to 11_spatial_correlation.py.

Metrics (mean absolute error between the reconstructed and observed
inter-station correlation matrices, upper triangle, across the 4
stations):
  pearson_amount_corr_mae   -- precipitation amount, Pearson
  spearman_amount_corr_mae  -- precipitation amount, Spearman
  wet_occurrence_corr_mae   -- binary wet/dry occurrence, Pearson

Outputs:
  results/canonical/analysis/spatial_correlation.csv
  results/canonical/analysis/spatial_correlation_summary.md
"""
import os
import warnings
import numpy as np
import pandas as pd

import canonical_metrics as cm
import build_canonical_outputs as bco
import direct_two_stage_rf as d2s

warnings.filterwarnings('ignore')

ANALYSIS_DIR = os.path.join(bco.CANON_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

WET_THRESH = cm.WET_THRESH
METHOD_ORDER = ['Mean', 'Linear', 'KNN', 'MICE', 'SAITS', 'AmountRF_DLPIF', 'DirectTwoStageRF']
SCENARIO_ORDER = ['10pct', '20pct', 'block7d', 'block30d']


def compute_corr_mae(series_df, val_col, obs_col, method='pearson', is_binary=False):
    pivot_pred = series_df.pivot(index='date', columns='station_id', values=val_col)
    pivot_obs  = series_df.pivot(index='date', columns='station_id', values=obs_col)
    if is_binary:
        pivot_pred = (pivot_pred > WET_THRESH).astype(int)
        pivot_obs  = (pivot_obs > WET_THRESH).astype(int)
    corr_pred = pivot_pred.corr(method=method).values
    corr_obs  = pivot_obs.corr(method=method).values
    idx = np.triu_indices_from(corr_obs, k=1)
    return float(np.mean(np.abs(corr_obs[idx] - corr_pred[idx])))


def main():
    print('=' * 65)
    print('  canonical_spatial_correlation.py -- Table 8 source')
    print('=' * 65)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    te = d2s.load_npz('test')
    dates = pd.to_datetime(te['dates'])
    sids = te['station_ids']
    y_true_full = sc.inverse_transform(
        np.nan_to_num(te['data'].astype(np.float64), nan=0.0))[:, pidx]

    records = []
    for method, (seeds, _loader, method_scenarios) in bco.METHODS.items():
        for seed in seeds:
            for scen_label, _c, _m in method_scenarios:
                series = bco.build_filled_series(method, seed, scen_label, sc, pidx,
                                                 te, dates, sids, y_true_full)
                pearson_mae  = compute_corr_mae(series, 'y_filled', 'y_true_full', 'pearson', False)
                spearman_mae = compute_corr_mae(series, 'y_filled', 'y_true_full', 'spearman', False)
                occ_mae      = compute_corr_mae(series, 'y_filled', 'y_true_full', 'pearson', True)
                records.append(dict(
                    method=method, scenario=scen_label, seed=seed,
                    pearson_amount_corr_mae=pearson_mae,
                    spearman_amount_corr_mae=spearman_mae,
                    wet_occurrence_corr_mae=occ_mae,
                ))
            print(f'  [OK] {method} seed={seed}')

    result_df = pd.DataFrame(records)
    out_csv = os.path.join(ANALYSIS_DIR, 'spatial_correlation.csv')
    result_df.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}  ({len(result_df)} rows)')

    numeric_cols = ['pearson_amount_corr_mae', 'spearman_amount_corr_mae', 'wet_occurrence_corr_mae']
    final_summary = result_df.groupby(['method', 'scenario'])[numeric_cols].mean().reset_index()
    final_summary['_m'] = final_summary['method'].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    final_summary['_s'] = final_summary['scenario'].map(lambda s: SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 99)
    final_summary = final_summary.sort_values(['_s', '_m'])

    md_lines = ['# Spatial Correlation Analysis Summary (Table 8 source)', '',
               'Inter-station correlation matrix MAE vs. observed, mean across seeds.', '',
               '| Scenario | Method | Pearson Amount MAE | Spearman Amount MAE | Wet Occurrence MAE |',
               '|---|---|---|---|---|']
    for _, r in final_summary.iterrows():
        md_lines.append(
            f"| {r['scenario']} | {r['method']} | {r['pearson_amount_corr_mae']:.4f} | "
            f"{r['spearman_amount_corr_mae']:.4f} | {r['wet_occurrence_corr_mae']:.4f} |")
    md_path = os.path.join(ANALYSIS_DIR, 'spatial_correlation_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')
    print('=' * 65)


if __name__ == '__main__':
    main()

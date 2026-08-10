"""
provisional_dry_spell_diagnostics.py
======================================
PROVISIONAL RQ1 DIAGNOSTIC -- NOT part of the canonical, audited pipeline.

Computes the same CDD (max consecutive dry days) / dry-spell hydrological
indices as canonical_dry_spell.py (Table 7), but sourced directly from each
method's raw prediction artifact instead of results/canonical/predictions/
(which is produced by build_canonical_outputs.py and currently requires
WGAN-GP outputs -- gan_imputed_test_modeB_*.npy / *_precipfix.npy -- that
only exist for the original {10pct, 20pct, block7d, block30d} scenarios;
generating them for mar_meteo/mnar_wet/mnar_intensity_{moderate,severe}
needs a GPU WGAN-GP run this environment cannot do).

This script answers a narrower, cheaper question first: does the
MNAR-Intensity F1-up/RMSE-up divergence (see results/direct_two_stage_rf_
evaluation.csv, results/single_stage_rf_evaluation.csv,
results/baseline_precip_occurrence_metrics.csv, results_dl/saits_v2/
evaluation_v2_seed*.csv) also show up as degraded HYDROLOGICAL structure
(dry-spell length, CDD), or is it confined to pointwise RMSE? It covers
every method that does NOT require WGAN-GP: Mean, Linear, KNN, MICE,
SAITS, Single-Stage RF, DirectTwoStageRF (== AmountRF_DLPIF at masked
positions, see results/canonical/README.md's dlpif_vs_direct_equivalence
finding).

Method (identical definitions to canonical_dry_spell.py, only the series-
reconstruction step differs):
  1. y_true_full: the full observed PRECIP series (mm), one row per
     (date, station), read straight from preprocessed_test.npz + scaler.pkl.
  2. y_filled: y_true_full, with each method's own prediction substituted
     in AT ITS OWN ARTIFICIALLY-MASKED POSITIONS ONLY (mirrors
     build_canonical_outputs.build_filled_series()). Sources:
       Mean/Linear/KNN/MICE   -- baseline_results.pkl['all_scenarios']
       Single-Stage RF        -- single_stage_rf_test_seed{s}_{scen}.npy
       DirectTwoStageRF       -- direct_two_stage_rf_test_seed{s}_{scen}.npy
       SAITS                  -- results_dl/saits_v2/saits_imputed_test_
                                   {scen}_v2_seed{s}_flat.npy (only covers
                                   the first 4080/4168 rows due to
                                   non-overlapping 30-day windowing; the
                                   dropped tail falls back to y_true_full,
                                   same as evaluate_saits.py's own N-trim)
  3. Dry-spell run lengths (PRECIP <= 0.1mm) computed per station on both
     series; CDD = max run length.

Output:
  results/rq1_mechanism_diagnostics/provisional_dry_spell_diagnostics.csv
  results/rq1_mechanism_diagnostics/provisional_dry_spell_summary.md
"""
import os
import pickle
import numpy as np
import pandas as pd

import canonical_metrics as cm
import direct_two_stage_rf as d2s
from canonical_dry_spell import get_dry_spell_lengths, compute_spell_metrics

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq1_mechanism_diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

WET_THRESH = cm.WET_THRESH
SEEDS = [42, 123, 456]
SCENARIO_ORDER = ['10pct', '20pct', 'block7d', 'block30d',
                   'mar_meteo', 'mnar_wet',
                   'mnar_intensity_moderate', 'mnar_intensity_severe']
METHOD_ORDER = ['Mean', 'Linear', 'KNN', 'MICE', 'SAITS',
                 'SingleStageRF', 'DirectTwoStageRF']


def build_filled_series_baseline(method_key, scenario, pidx, scaler, te, y_true_full, bl_all):
    key = (method_key, scenario)
    if key not in bl_all:
        return None
    pred_norm = np.clip(bl_all[key].astype(np.float32), 0, 1)
    pred_mm = scaler.inverse_transform(pred_norm)[:, pidx]
    mask = te[f'art_mask_{scenario}'].astype(np.float32)[:, pidx] > 0.5
    y_filled = y_true_full.copy()
    y_filled[mask] = pred_mm[mask]
    return y_filled


def build_filled_series_npy(prefix, seed, scenario, pidx, scaler, te, y_true_full):
    path = os.path.join(SRC_DIR, f'{prefix}_test_seed{seed}_{scenario}.npy')
    if not os.path.exists(path):
        return None
    pred_norm = np.load(path).astype(np.float32)
    pred_mm = scaler.inverse_transform(np.clip(pred_norm, 0, 1))[:, pidx]
    mask = te[f'art_mask_{scenario}'].astype(np.float32)[:, pidx] > 0.5
    y_filled = y_true_full.copy()
    y_filled[mask] = pred_mm[mask]
    return y_filled


def build_filled_series_saits(seed, scenario, pidx, scaler, te, y_true_full):
    path = os.path.join(SRC_DIR, 'results_dl', 'saits_v2',
                         f'saits_imputed_test_{scenario}_v2_seed{seed}_flat.npy')
    if not os.path.exists(path):
        return None
    flat_norm = np.load(path).astype(np.float32)
    N = flat_norm.shape[0]
    flat_mm = scaler.inverse_transform(np.clip(flat_norm, 0, 1))[:, pidx]
    mask = te[f'art_mask_{scenario}'].astype(np.float32)[:, pidx] > 0.5
    mask_covered = mask.copy()
    mask_covered[N:] = False   # SAITS windowing drops the tail; fall back to y_true there
    y_filled = y_true_full.copy()
    covered_idx = np.where(mask_covered)[0]
    y_filled[covered_idx] = flat_mm[covered_idx]
    return y_filled


def main():
    print('=' * 70)
    print('  PROVISIONAL RQ1 DIAGNOSTIC -- dry-spell / CDD, NOT canonical pipeline')
    print('  (Mean/Linear/KNN/MICE/SAITS/SingleStageRF/DirectTwoStageRF only;')
    print('   WGAN-GP not included -- see module docstring)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    te = d2s.load_npz('test')
    dates = pd.to_datetime(te['dates'])
    sids = te['station_ids']
    y_true_full = sc.inverse_transform(
        np.nan_to_num(te['data'].astype(np.float64), nan=0.0))[:, pidx]

    with open(os.path.join(SRC_DIR, 'baseline_results.pkl'), 'rb') as f:
        bl_all = pickle.load(f)['all_scenarios']

    scenario_labels = [label for label, _, _ in d2s.SCENARIOS if label in SCENARIO_ORDER]

    records = []
    for scen in scenario_labels:
        if f'art_mask_{scen}' not in te.files:
            continue

        # -- Deterministic baselines (seed='deterministic') --
        for method_disp, method_key in [('Mean', 'mean'), ('Linear', 'linear'),
                                         ('KNN', 'knn'), ('MICE', 'mice')]:
            y_filled = build_filled_series_baseline(method_key, scen, pidx, sc, te, y_true_full, bl_all)
            if y_filled is None:
                continue
            for sid in pd.unique(sids):
                rows = np.where(sids == sid)[0]
                obs_m = compute_spell_metrics(get_dry_spell_lengths(y_true_full[rows]))
                pred_m = compute_spell_metrics(get_dry_spell_lengths(y_filled[rows]))
                records.append(dict(
                    method=method_disp, scenario=scen, seed='deterministic', station_id=sid,
                    cdd_obs=obs_m['cdd'], cdd_pred=pred_m['cdd'],
                    cdd_abs_error=abs(obs_m['cdd'] - pred_m['cdd']),
                    mean_dry_obs=obs_m['mean_dry'], mean_dry_pred=pred_m['mean_dry'],
                    mean_dry_abs_error=abs(obs_m['mean_dry'] - pred_m['mean_dry']),
                    p95_dry_obs=obs_m['p95_dry'], p95_dry_pred=pred_m['p95_dry'],
                    p95_dry_abs_error=abs(obs_m['p95_dry'] - pred_m['p95_dry']),
                ))

        # -- Seeded RF methods + SAITS --
        for seed in SEEDS:
            for method_disp, y_filled in [
                ('SingleStageRF', build_filled_series_npy('single_stage_rf', seed, scen, pidx, sc, te, y_true_full)),
                ('DirectTwoStageRF', build_filled_series_npy('direct_two_stage_rf', seed, scen, pidx, sc, te, y_true_full)),
                ('SAITS', build_filled_series_saits(seed, scen, pidx, sc, te, y_true_full)),
            ]:
                if y_filled is None:
                    continue
                for sid in pd.unique(sids):
                    rows = np.where(sids == sid)[0]
                    obs_m = compute_spell_metrics(get_dry_spell_lengths(y_true_full[rows]))
                    pred_m = compute_spell_metrics(get_dry_spell_lengths(y_filled[rows]))
                    records.append(dict(
                        method=method_disp, scenario=scen, seed=seed, station_id=sid,
                        cdd_obs=obs_m['cdd'], cdd_pred=pred_m['cdd'],
                        cdd_abs_error=abs(obs_m['cdd'] - pred_m['cdd']),
                        mean_dry_obs=obs_m['mean_dry'], mean_dry_pred=pred_m['mean_dry'],
                        mean_dry_abs_error=abs(obs_m['mean_dry'] - pred_m['mean_dry']),
                        p95_dry_obs=obs_m['p95_dry'], p95_dry_pred=pred_m['p95_dry'],
                        p95_dry_abs_error=abs(obs_m['p95_dry'] - pred_m['p95_dry']),
                    ))
        print(f'  [OK] {scen}')

    df = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, 'provisional_dry_spell_diagnostics.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}  ({len(df)} rows)')

    numeric_cols = ['cdd_obs', 'cdd_pred', 'cdd_abs_error',
                    'mean_dry_obs', 'mean_dry_pred', 'mean_dry_abs_error',
                    'p95_dry_obs', 'p95_dry_pred', 'p95_dry_abs_error']
    per_seed = df.groupby(['method', 'scenario', 'seed'])[numeric_cols].mean().reset_index()
    summary = per_seed.groupby(['method', 'scenario'])[numeric_cols].mean().reset_index()
    summary['_m'] = summary['method'].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    summary['_s'] = summary['scenario'].map(lambda s: SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 99)
    summary = summary.sort_values(['_s', '_m'])

    md_lines = ['# PROVISIONAL RQ1 Dry-Spell / CDD Diagnostic', '',
               '**NOT part of the canonical, audited pipeline** (results/canonical/) --',
               'WGAN-GP has no GPU-generated output for the new mechanism scenarios, so',
               'WGANGP_raw/WGANGP_PrecipFix are excluded here. See module docstring.', '',
               'Averages across 4 stations and all available seeds. A dry day is PRECIP <= 0.1mm.', '',
               '| Scenario | Method | Obs CDD | Pred CDD | CDD Error | Obs Mean | '
               'Pred Mean | Mean Error | Obs p95 | Pred p95 | p95 Error |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in summary.iterrows():
        md_lines.append(
            f"| {r['scenario']} | {r['method']} | {r['cdd_obs']:.1f} | {r['cdd_pred']:.1f} | "
            f"{r['cdd_abs_error']:.2f} | {r['mean_dry_obs']:.2f} | {r['mean_dry_pred']:.2f} | "
            f"{r['mean_dry_abs_error']:.2f} | {r['p95_dry_obs']:.1f} | {r['p95_dry_pred']:.1f} | "
            f"{r['p95_dry_abs_error']:.2f} |")
    md_path = os.path.join(OUT_DIR, 'provisional_dry_spell_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')

    print('\n  RQ1-relevant subset (DirectTwoStageRF, dose-response scenarios):')
    show = summary[(summary.method == 'DirectTwoStageRF') &
                    (summary.scenario.isin(['10pct', 'mar_meteo', 'mnar_wet',
                                            'mnar_intensity_moderate', 'mnar_intensity_severe']))]
    print(show[['scenario', 'cdd_obs', 'cdd_pred', 'cdd_abs_error',
                'mean_dry_abs_error', 'p95_dry_abs_error']].to_string(index=False))
    print('=' * 70)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
canonical_dry_spell.py
========================
Dry-spell / CDD (max consecutive dry days) hydrological-index analysis
(manuscript Table 7), computed exclusively from
preprocessed_test.npz + results/canonical/predictions/*.csv via
build_canonical_outputs.build_filled_series() -- the canonical successor
to 10_dry_spell_cdd.py, which read a separate results/filled_datasets/
directory this session's pipeline no longer produces.

Dry-spell length requires day-to-day continuity, which a masked-positions-
only table cannot provide -- build_filled_series() reconstructs the FULL
per-station daily series (ground truth at observed positions, this
method's own prediction at its masked positions) before spell lengths are
computed. A dry day is PRECIP <= 0.1 mm.

Output columns per (method, scenario, seed, station_id) row:
  cdd_obs, cdd_pred, cdd_abs_error,
  mean_dry_obs, mean_dry_pred, mean_dry_abs_error,
  p95_dry_obs, p95_dry_pred, p95_dry_abs_error,
  n_dry_spells_obs, n_dry_spells_pred

Outputs:
  results/canonical/analysis/dry_spells.csv          (station-level detail)
  results/canonical/analysis/dry_spells_summary.md    (method x scenario)
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
SCENARIO_ORDER = ['10pct', '20pct', 'block7d', 'block30d',
                   'mar_meteo', 'mnar_wet',
                   'mnar_intensity_moderate', 'mnar_intensity_severe']


def get_dry_spell_lengths(precip_array):
    is_dry = (precip_array <= WET_THRESH).astype(int)
    padded = np.pad(is_dry, (1, 1), mode='constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return ends - starts


def compute_spell_metrics(lengths):
    if len(lengths) == 0:
        return dict(cdd=0.0, mean_dry=0.0, p95_dry=0.0, n_spells=0)
    return dict(cdd=float(np.max(lengths)), mean_dry=float(np.mean(lengths)),
               p95_dry=float(np.percentile(lengths, 95)), n_spells=len(lengths))


def main():
    print('=' * 65)
    print('  canonical_dry_spell.py -- Table 7 source')
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
                series = series.sort_values(['station_id', 'date'])

                for station_id, grp in series.groupby('station_id'):
                    obs_m = compute_spell_metrics(get_dry_spell_lengths(grp['y_true_full'].to_numpy()))
                    pred_m = compute_spell_metrics(get_dry_spell_lengths(grp['y_filled'].to_numpy()))
                    records.append(dict(
                        method=method, scenario=scen_label, seed=seed, station_id=station_id,
                        cdd_obs=obs_m['cdd'], cdd_pred=pred_m['cdd'],
                        cdd_abs_error=abs(obs_m['cdd'] - pred_m['cdd']),
                        mean_dry_obs=obs_m['mean_dry'], mean_dry_pred=pred_m['mean_dry'],
                        mean_dry_abs_error=abs(obs_m['mean_dry'] - pred_m['mean_dry']),
                        p95_dry_obs=obs_m['p95_dry'], p95_dry_pred=pred_m['p95_dry'],
                        p95_dry_abs_error=abs(obs_m['p95_dry'] - pred_m['p95_dry']),
                        n_dry_spells_obs=obs_m['n_spells'], n_dry_spells_pred=pred_m['n_spells'],
                    ))
            print(f'  [OK] {method} seed={seed}')

    result_df = pd.DataFrame(records)
    out_csv = os.path.join(ANALYSIS_DIR, 'dry_spells.csv')
    result_df.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}  ({len(result_df)} rows)')

    numeric_cols = ['cdd_obs', 'cdd_pred', 'cdd_abs_error',
                    'mean_dry_obs', 'mean_dry_pred', 'mean_dry_abs_error',
                    'p95_dry_obs', 'p95_dry_pred', 'p95_dry_abs_error',
                    'n_dry_spells_obs', 'n_dry_spells_pred']
    # average across stations, then across seeds
    per_seed = result_df.groupby(['method', 'scenario', 'seed'])[numeric_cols].mean().reset_index()
    final_summary = per_seed.groupby(['method', 'scenario'])[numeric_cols].mean().reset_index()

    final_summary['_m'] = final_summary['method'].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    final_summary['_s'] = final_summary['scenario'].map(lambda s: SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 99)
    final_summary = final_summary.sort_values(['_s', '_m'])

    md_lines = ['# Dry-Spell and CDD Analysis Summary (Table 7 source)', '',
               'Averages across 4 stations and all available seeds.', '',
               '| Scenario | Method | Obs CDD | Pred CDD | CDD Error | Obs Mean | '
               'Pred Mean | Mean Error | Obs p95 | Pred p95 | p95 Error |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in final_summary.iterrows():
        md_lines.append(
            f"| {r['scenario']} | {r['method']} | {r['cdd_obs']:.1f} | {r['cdd_pred']:.1f} | "
            f"{r['cdd_abs_error']:.2f} | {r['mean_dry_obs']:.2f} | {r['mean_dry_pred']:.2f} | "
            f"{r['mean_dry_abs_error']:.2f} | {r['p95_dry_obs']:.1f} | {r['p95_dry_pred']:.1f} | "
            f"{r['p95_dry_abs_error']:.2f} |")
    md_path = os.path.join(ANALYSIS_DIR, 'dry_spells_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')
    print('=' * 65)


if __name__ == '__main__':
    main()

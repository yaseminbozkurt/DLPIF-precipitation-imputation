# -*- coding: utf-8 -*-
"""
canonical_stage_ablation.py
=============================
Occurrence-only vs. full DLPIF (Stage 1 + Stage 2) ablation -- sources for
manuscript Table 3 ("seed-robustness summary for the occurrence-amount
ablation") and Table 5 ("ablation analysis of precipitation-specific
correction stages"), rebuilt 2026-07-23 to no longer depend on WGAN-GP.

Both tables previously compared "Precip2Stage" (Stage-1 occurrence RF
applied on top of WGAN-GP's raw continuous amount, via quantile mapping)
against full DLPIF (Stage-1 occurrence RF + Stage-2 amount RF). WGAN-GP was
dropped 2026-07-23 (no GPU available to retrain it on the post-2005
dataset), so that specific comparison no longer exists.

Rebuilt as a fully backbone-free ablation, consistent with DLPIF's actual
design (the whole point of this paper's occurrence-amount decomposition is
that it never needs a continuous backbone for PRECIP):
  "Occurrence-only" -- same Stage-1 occurrence RF decision (wet/dry) as
    DLPIF, but wet-predicted cells are filled with that station's TRAINING-
    SET wet-day climatological mean amount instead of a learned regression.
  "DLPIF" -- Stage-1 occurrence RF + Stage-2 amount RF (unchanged).
This isolates the marginal contribution of Stage 2 (learned amount
regression) over a naive climatological-mean fill, given the identical
Stage-1 occurrence decision -- the same ablation question the removed
Precip2Stage comparison was asking, now answered without any backbone
imputer at all.

The wet/dry decision is read directly from the canonical AmountRF_DLPIF
predictions (y_pred > 0 <=> Stage 1 predicted wet, since dry-predicted
cells are always written as exactly 0.0 -- see multiseed_clean_rerun.py),
so this script does not retrain anything; it only recombines already-
computed canonical predictions with a directly-computed training climatology.

Outputs:
  results/canonical/analysis/stage_ablation.csv       (per seed x scenario)
  results/canonical/analysis/stage_ablation_summary.md (mean +/- std across seeds)
"""
import os
import numpy as np
import pandas as pd

import canonical_metrics as cm
import build_canonical_outputs as bco
import direct_two_stage_rf as d2s

ANALYSIS_DIR = os.path.join(bco.CANON_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

SCENARIO_ORDER = ['10pct', '20pct', 'block7d', 'block30d']


def station_wet_climatology(sc, pidx, tr):
    """Per-station mean of TRUE wet-day (>WET_THRESH) training precipitation."""
    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = tr_obs & (tr_gt > d2s.WET_THRESH)
    sids = tr['station_ids']
    clim = {}
    for sid in np.unique(sids):
        vals = tr_gt[tr_wet & (sids == sid)]
        clim[sid] = float(vals.mean()) if len(vals) else float(tr_gt[tr_wet].mean())
    return clim


def main():
    print('=' * 70)
    print('  canonical_stage_ablation.py -- Occurrence-only vs DLPIF (Tables 3/5)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    tr = d2s.load_npz('train')
    clim = station_wet_climatology(sc, pidx, tr)
    print('  Station wet-day climatological means (mm):',
         {int(k): round(v, 2) for k, v in clim.items()})

    dlpif = pd.read_csv(os.path.join(bco.PRED_DIR, 'AmountRF_DLPIF.csv'), dtype={'seed': str})

    records = []
    for seed in d2s.SEEDS:
        for scen in SCENARIO_ORDER:
            grp = dlpif[(dlpif['scenario'] == scen) & (dlpif['seed'] == str(seed))]
            assert len(grp) > 0, f'no rows for seed={seed} scenario={scen}'
            y_true = grp['y_true'].to_numpy()
            y_pred_dlpif = grp['y_pred'].to_numpy()
            wet_pred = y_pred_dlpif > 0.0
            station_id = grp['station_id'].to_numpy()

            y_pred_occ_only = np.zeros_like(y_true)
            for sid, mean_amt in clim.items():
                sel = wet_pred & (station_id == sid)
                y_pred_occ_only[sel] = mean_amt

            m_dlpif = cm.precip_metrics(y_true, y_pred_dlpif)
            m_occ = cm.precip_metrics(y_true, y_pred_occ_only)
            records.append(dict(
                seed=seed, scenario=scen, n=len(y_true),
                dlpif_rmse_wet=m_dlpif['rmse_wet'], occ_only_rmse_wet=m_occ['rmse_wet'],
                dlpif_rmse_p95=m_dlpif['rmse_p95'], occ_only_rmse_p95=m_occ['rmse_p95'],
                dlpif_f1=m_dlpif['f1'], occ_only_f1=m_occ['f1'],
                dlpif_bias=m_dlpif['signed_frequency_bias'],
                occ_only_bias=m_occ['signed_frequency_bias'],
            ))
            print(f'  [{scen} seed={seed}] occ-only RMSE_wet={m_occ["rmse_wet"]:.2f} '
                 f'DLPIF RMSE_wet={m_dlpif["rmse_wet"]:.2f}  '
                 f'occ-only p95={m_occ["rmse_p95"]:.2f} DLPIF p95={m_dlpif["rmse_p95"]:.2f}')

    df = pd.DataFrame(records)
    out_csv = os.path.join(ANALYSIS_DIR, 'stage_ablation.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}')

    summary = df.groupby('scenario').agg(
        dlpif_rmse_wet_mean=('dlpif_rmse_wet', 'mean'), dlpif_rmse_wet_std=('dlpif_rmse_wet', 'std'),
        occ_only_rmse_wet_mean=('occ_only_rmse_wet', 'mean'), occ_only_rmse_wet_std=('occ_only_rmse_wet', 'std'),
        dlpif_rmse_p95_mean=('dlpif_rmse_p95', 'mean'), dlpif_rmse_p95_std=('dlpif_rmse_p95', 'std'),
        occ_only_rmse_p95_mean=('occ_only_rmse_p95', 'mean'), occ_only_rmse_p95_std=('occ_only_rmse_p95', 'std'),
    ).round(4).reindex(SCENARIO_ORDER)
    print('\n', summary.to_string())

    md_lines = ['# Occurrence-only vs. DLPIF Stage Ablation (Tables 3/5 source)', '',
               'Both variants share the identical Stage-1 occurrence-RF wet/dry '
               'decision. "Occurrence-only" fills wet-predicted cells with that '
               'station\'s training-set wet-day climatological mean; "DLPIF" '
               'fills them with the Stage-2 amount-RF regression. Mean +/- std '
               'across seeds 42, 123, 456.', '',
               '| Scenario | Occ-only RMSE_wet | DLPIF RMSE_wet | Occ-only Extreme RMSE | DLPIF Extreme RMSE |',
               '|---|---|---|---|---|']
    for scen in SCENARIO_ORDER:
        r = summary.loc[scen]
        md_lines.append(
            f"| {scen} | {r['occ_only_rmse_wet_mean']:.2f} ± {r['occ_only_rmse_wet_std']:.3f} "
            f"| {r['dlpif_rmse_wet_mean']:.2f} ± {r['dlpif_rmse_wet_std']:.3f} "
            f"| {r['occ_only_rmse_p95_mean']:.2f} ± {r['occ_only_rmse_p95_std']:.3f} "
            f"| {r['dlpif_rmse_p95_mean']:.2f} ± {r['dlpif_rmse_p95_std']:.3f} |")
    md_path = os.path.join(ANALYSIS_DIR, 'stage_ablation_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()

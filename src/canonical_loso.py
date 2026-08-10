# -*- coding: utf-8 -*-
"""
canonical_loso.py
===================
Leave-One-Station-Out (LOSO) spatial cross-validation (manuscript
Section 5.10): for each of the 4 stations, hold it out of training
entirely, train Stage 1/Stage 2 on the remaining 3 stations' 10%
random-missingness partition, and evaluate purely out-of-sample on the
held-out station's masked test rows, across all four test scenarios.

Confirms that the correction layer learns generalisable meteorological
relationships rather than memorising the training stations' climatology.

Leakage fix (2026-07-23)
------------------------
The previous version excluded the held-out station's own rows from
tr_keep/va_keep (as training TARGETS), but reused the GLOBAL scaler and
GLOBAL neighbor_avg/neighbor_mask features -- both fit/computed once across
all 4 stations in 01_data_preprocessing.py, before any fold-specific
exclusion. This meant the OTHER 3 stations' neighbor-average predictor
features could already encode the held-out station's true observed values
(if it was one of their k=2 nearest neighbors), and the global scaler's
min/max range was influenced by the held-out station's training-period
values -- both real leak paths into a supposedly held-out evaluation.

Fixed by, per fold:
  1. Recovering true raw (unscaled) values via the GLOBAL scaler's inverse
     transform -- an exact, deterministic recovery (the scaler is an
     invertible affine map), not a re-estimation, so this step alone
     introduces no leakage.
  2. Refitting a NEW MinMaxScaler on only the non-held-out stations'
     training rows, and re-normalising all partitions with it.
  3. Zeroing the held-out station's COLUMN in the (geography-only, static,
     non-leaking) adjacency matrix, so no OTHER station may use it as a
     neighbour-DATA source -- while the held-out station's own ROW keeps
     its original 2-nearest-neighbour weights (already guaranteed
     non-held-out by construction), which is legitimate: using OTHER,
     already-trained-on stations' real observations to help predict the
     held-out station is genuine spatial generalisation, not leakage.
  4. Recomputing neighbor_avg/neighbor_mask from the re-scaled,
     leakage-safe adjacency for every scenario actually used.
Station geography (lat/lon/elev) itself is not a leak -- it is static
metadata, always known a priori in any real deployment, unlike a station's
actual meteorological readings.

Reuses direct_two_stage_rf.py's exact feature-construction and training
code (build_occ_X, train_occ) and 01_data_preprocessing.py's exact
compute_neighbor_avg implementation -- the only new code is the per-fold
scaler/adjacency refit above.
Scored with the same canonical_metrics.precip_metrics implementation used
throughout the pipeline. Seed 42 only (matches the manuscript's existing
LOSO framing).

Outputs:
  results/canonical/analysis/loso.csv           (per-station-fold detail)
  results/canonical/analysis/loso_summary.md     (averaged over 4 folds)
"""
import os
import sys
import pickle
import warnings
import importlib.util
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import canonical_metrics as cm
import build_canonical_outputs as bco

sys.stdout.flush()
import direct_two_stage_rf as d2s  # reuse exact feature/RF code

warnings.filterwarnings('ignore')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# -- 01_data_preprocessing.py unconditionally does
# `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)` at module level
# (written to run standalone, never imported). Keeping an explicit
# reference to every intermediate wrapper prevents CPython's refcounting GC
# from closing the shared underlying buffer and raising "I/O operation on
# closed file" on the next print() (see canonical_multimask_variance.py,
# which established this same pattern first).
_stdout_keepalive = [sys.stdout]
_spec = importlib.util.spec_from_file_location(
    'preprocessing01', os.path.join(SRC_DIR, '01_data_preprocessing.py'))
preprocessing01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocessing01)
_stdout_keepalive.append(sys.stdout)

ANALYSIS_DIR = os.path.join(bco.CANON_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

SEED = 42
TRAIN_SCENARIO = d2s.TRAIN_SCENARIO       # '10pct'
TRAIN_COR_KEY = d2s.TRAIN_COR_KEY         # 'corrupted_10pct'
SCENARIO_ORDER = ['10pct', '20pct', 'block7d', 'block30d']


def refit_fold_arrays(held_out_station, sc, station_list, A_knn, tr, va, te):
    """Rebuild every scaler-dependent and neighbor-dependent array for one
    LOSO fold, excluding held_out_station from the scaler fit AND from
    being usable as any other station's neighbour-data source. Returns
    sc_fold, {split: raw_mm_array}, and {(split, scenario): dict(corrupted,
    neighbor_avg, neighbor_mask)} (all normalised [0,1] except raw_mm)."""
    tr_keep = tr['station_ids'] != held_out_station

    raw = {
        'train': sc.inverse_transform(tr['data'].astype(np.float64)),
        'val':   sc.inverse_transform(va['data'].astype(np.float64)),
        'test':  sc.inverse_transform(te['data'].astype(np.float64)),
    }

    sc_fold = MinMaxScaler(feature_range=(0, 1))
    sc_fold.fit(raw['train'][tr_keep])  # NaN ignored by sklearn's MinMaxScaler.fit

    data_fold = {
        'train': sc_fold.transform(raw['train']).astype(np.float32),
        'val':   sc_fold.transform(raw['val']).astype(np.float32),
        'test':  sc_fold.transform(raw['test']).astype(np.float32),
    }

    held_idx = station_list.index(held_out_station)
    A_knn_fold = A_knn.copy()
    A_knn_fold[:, held_idx] = 0.0

    npzs = {'train': tr, 'val': va, 'test': te}
    scenario_arrays = {}
    for split, z in npzs.items():
        sids = z['station_ids']
        for scen_label, cor_key, mask_key in d2s.SCENARIOS:
            if cor_key not in z.files:
                continue
            art_mask = z[mask_key].astype(bool)
            corrupted_fold = data_fold[split].copy()
            corrupted_fold[art_mask] = np.nan
            nbr_avg_fold, nbr_mask_fold = preprocessing01.compute_neighbor_avg(
                corrupted_fold, sids, A_knn_fold, station_list)
            scenario_arrays[(split, scen_label)] = dict(
                corrupted=corrupted_fold, neighbor_avg=nbr_avg_fold,
                neighbor_mask=nbr_mask_fold)

    return sc_fold, raw, scenario_arrays


def build_amt_X_fold(sc_fold, corrupted_norm, temporal, neighbor_avg_norm,
                     neighbor_mask_norm, pidx):
    """Fold-local equivalent of direct_two_stage_rf.build_amt_X, operating
    on already-computed fold arrays instead of an npz dict."""
    cor = d2s.inv(sc_fold, corrupted_norm)
    cor[:, pidx] = 0.0
    nbr_avg_mm = d2s.inv(sc_fold, neighbor_avg_norm)
    return np.concatenate(
        [cor, temporal.astype(np.float32), nbr_avg_mm,
         neighbor_mask_norm.astype(np.float32)], axis=1).astype(np.float32)


def run_fold(held_out_station, sc, station_list, A_knn, pidx, tr, va, te):
    tr_keep = tr['station_ids'] != held_out_station
    va_keep = va['station_ids'] != held_out_station

    sc_fold, raw, scen = refit_fold_arrays(held_out_station, sc, station_list, A_knn, tr, va, te)

    tr10 = scen[('train', TRAIN_SCENARIO)]
    va10 = scen[('val', TRAIN_SCENARIO)]

    X_tr_full = d2s.build_occ_X(tr10['corrupted'], tr['temporal'].astype(np.float32),
                                tr10['neighbor_avg'], tr10['neighbor_mask'], pidx)
    X_va_full = d2s.build_occ_X(va10['corrupted'], va['temporal'].astype(np.float32),
                                va10['neighbor_avg'], va10['neighbor_mask'], pidx)
    X_tr_amt  = build_amt_X_fold(sc_fold, tr10['corrupted'], tr['temporal'],
                                 tr10['neighbor_avg'], tr10['neighbor_mask'], pidx)

    tr_gt = raw['train'][:, pidx]
    va_gt = raw['val'][:, pidx]

    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool) & tr_keep
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool) & va_keep
    tr_wet = (tr_gt > d2s.WET_THRESH) & tr_obs

    X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs] > d2s.WET_THRESH).astype(int)
    X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs] > d2s.WET_THRESH).astype(int)

    occ_rf, occ_cut, vm = d2s.train_occ(X_tr_occ, y_tr, X_va_occ, y_va, SEED)

    amt_rf = RandomForestRegressor(n_estimators=400, random_state=SEED,
                                   min_samples_leaf=2, n_jobs=-1)
    amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

    te_gt = raw['test'][:, pidx]
    te_is_held_out = te['station_ids'] == held_out_station

    fold_metrics = {}
    for scen_label, cor_key, mask_key in d2s.SCENARIOS:
        te_s = scen[('test', scen_label)]
        X_occ = d2s.build_occ_X(te_s['corrupted'], te['temporal'].astype(np.float32),
                                te_s['neighbor_avg'], te_s['neighbor_mask'], pidx)
        X_amt = build_amt_X_fold(sc_fold, te_s['corrupted'], te['temporal'],
                                 te_s['neighbor_avg'], te_s['neighbor_mask'], pidx)

        m = (te[mask_key].astype(np.float32)[:, pidx] > 0.5) & te_is_held_out
        gt_m = te_gt[m]

        te_proba = occ_rf.predict_proba(X_occ)[:, 1]
        te_wet_pred = (te_proba >= occ_cut).astype(bool)

        pred_scen = np.zeros_like(te_gt)
        apply_sel = m & te_wet_pred
        if apply_sel.sum() > 0:
            pred_scen[apply_sel] = np.maximum(amt_rf.predict(X_amt[apply_sel]), 0.0)

        fold_metrics[scen_label] = cm.precip_metrics(gt_m, pred_scen[m])
        fold_metrics[scen_label]['n_masked'] = int(m.sum())
    return occ_cut, vm, fold_metrics, sc_fold


def main():
    print('=' * 70)
    print('  canonical_loso.py -- Leave-One-Station-Out (Section 5.10 source)')
    print(f'  seed={SEED}, training scenario={TRAIN_SCENARIO}')
    print('  Per-fold leakage-safe scaler + adjacency refit (2026-07-23 fix)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')

    with open(os.path.join(SRC_DIR, 'adjacency.pkl'), 'rb') as f:
        adj = pickle.load(f)
    A_knn = adj['A_knn']
    station_list = adj['stations']

    stations = sorted(np.unique(te['station_ids']).tolist())
    assert stations == sorted(station_list), \
        f'station list mismatch: test={stations} vs adjacency={station_list}'

    records = []
    fold_scaler_deltas = []
    for st in stations:
        print(f'\n  --- held-out station {st} ---')
        occ_cut, vm, fold_metrics, sc_fold = run_fold(st, sc, station_list, A_knn, pidx, tr, va, te)
        print(f'  occ RF cutoff={occ_cut}  val_F1={vm["val_f1"]}')

        # Verify the fold scaler genuinely differs from the global one (proof
        # the held-out station's values were actually excluded from the fit).
        delta = float(np.max(np.abs(sc_fold.data_min_ - sc.data_min_)) +
                      np.max(np.abs(sc_fold.data_max_ - sc.data_max_)))
        fold_scaler_deltas.append(delta)
        print(f'  fold-scaler vs global-scaler max |delta| (min+max) = {delta:.6f}')

        for scen in SCENARIO_ORDER:
            m = fold_metrics[scen]
            records.append(dict(
                held_out_station=int(st), scenario=scen,
                n_masked=m['n_masked'], f1=m['f1'], rmse_wet=m['rmse_wet'],
                signed_frequency_bias=m['signed_frequency_bias'],
                precision=m['precision'], recall=m['recall'],
            ))
            print(f'    [{scen}] n={m["n_masked"]}  F1={m["f1"]:.4f}  '
                  f'RMSE_wet={m["rmse_wet"]}  bias={m["signed_frequency_bias"]:+.4f}')

    assert all(d > 1e-6 for d in fold_scaler_deltas), (
        'a fold scaler was numerically identical to the global scaler -- '
        'the held-out station exclusion did not take effect')
    print(f'\n  [OK] All {len(fold_scaler_deltas)} fold scalers verified distinct '
          f'from the global scaler (leakage-safe refit confirmed).')

    df = pd.DataFrame(records)
    out_csv = os.path.join(ANALYSIS_DIR, 'loso.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}')

    summary = df.groupby('scenario')[['f1', 'rmse_wet']].mean().round(4)
    summary = summary.reindex(SCENARIO_ORDER)
    print('\n  Averaged over 4 station-folds:')
    print(summary.to_string())

    md_lines = ['# Leave-One-Station-Out (LOSO) Spatial Cross-Validation Summary '
               '(Section 5.10 source)', '',
               f'Seed {SEED}. Each fold trains Stage 1/Stage 2 on 3 stations '
               '(10% random-missingness partition) and evaluates purely '
               'out-of-sample on the 4th held-out station, across all four test '
               'scenarios. The scaler and neighbour-averaging adjacency are '
               'refit per fold, excluding the held-out station, so no '
               'information about it can leak into the other 3 stations\' '
               'training features. Values below are averaged across the 4 '
               'station-folds.', '',
               '| Scenario | Mean F1 | Mean Wet RMSE (mm) |', '|---|---|---|']
    for scen in SCENARIO_ORDER:
        r = summary.loc[scen]
        md_lines.append(f"| {scen} | {r['f1']:.4f} | {r['rmse_wet']:.2f} |")
    md_path = os.path.join(ANALYSIS_DIR, 'loso_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()

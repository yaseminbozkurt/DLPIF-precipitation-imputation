# -*- coding: utf-8 -*-
"""
canonical_multimask_variance.py
==================================
Multi-mask variance experiment (manuscript Section 5.9), extended
2026-08-03 to strengthen the extreme-event (p95) analysis specifically:
in every single-mask evaluation elsewhere in this pipeline (Table 2,
Table S1, Table S2), the ground-truth extreme (>=19.20 mm) subset is tiny
-- 6-12 positions per scenario per seed (Section 5.3 seed-robustness
discussion) -- so a single mask realization's RMSE_p95 is estimated from
very few points and is correspondingly noisy. This experiment pools
MULTIPLE independent mask realizations ACROSS ALL THREE model seeds
(42, 123, 456) -- not seed 42 alone, as the original version of this
experiment did -- to give RMSE_p95 / MAE_p95 estimates backed by an order
of magnitude more extreme-event observations, alongside the original
occurrence-F1 variance check.

Methods compared, all applied to fresh mask realizations WITHOUT
retraining -- identical protocol to how each is applied across the four
named production test scenarios elsewhere in this pipeline:
  - Linear (station-wise interpolation): deterministic, no seed.
  - AmountRF_DLPIF: Stage 1 + Stage 2 trained once per seed on the 10pct
    production partition (direct_two_stage_rf.py's train_occ + a fresh
    RandomForestRegressor), then applied unchanged to each new mask.
  - SAITS: each seed's already-trained, checkpoint-verified model
    (baselines/train_saits_v2.py, loaded via its minimum-validation-MSE
    checkpoint -- see that script's 2026-08-03 bugfix) applied via
    .impute() to each new mask's corrupted array, including the
    overlapping tail-window reconstruction used everywhere else in this
    pipeline (baselines/repackage_saits_outputs.py) so no station's final
    rows are silently dropped from the extreme-event pool.

10 independent mask realizations per scenario (10pct, 20pct), mask seeds
disjoint from every seed used elsewhere in this pipeline (9000-9009, same
range as the original version of this experiment). Reusing exactly the
same functions used everywhere else in the canonical pipeline:
  - random_missingness / compute_neighbor_avg  (01_data_preprocessing.py)
  - linear_interpolation (station-wise)         (03_baseline_imputation.py)
  - build_occ_X / build_amt_X / train_occ       (direct_two_stage_rf.py)
  - to_log1p / log1p_to_mm / mm_to_norm /
    build_3d / reconstruct_flat                 (baselines/train_saits_v2.py)
  - precip_metrics                              (canonical_metrics.py)

Outputs:
  results/canonical/analysis/multimask_variance.csv
  results/canonical/analysis/multimask_variance_summary.md
  results/canonical/analysis/multimask_p95_variance_summary.md
"""
import importlib.util
import glob
import os
import re
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'baselines'))

import canonical_metrics as cm
import build_canonical_outputs as bco
import direct_two_stage_rf as d2s

# -- 01_data_preprocessing.py and 03_baseline_imputation.py both
# unconditionally do `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)`
# at module level (they're written to run standalone, never imported).
# Each reassignment drops the refcount of the previous TextIOWrapper to 0,
# and CPython's immediate refcounting GC then closes it -- which also
# closes the underlying buffer it wraps, orphaning the *next* wrapper and
# raising "I/O operation on closed file" on the next print(). Keeping an
# explicit reference to every intermediate wrapper prevents that.
_stdout_keepalive = [sys.stdout]

# -- dynamically load 01_data_preprocessing.py (filename starts with a
# digit, so it cannot be imported with a normal `import` statement) --
_spec = importlib.util.spec_from_file_location(
    'preprocessing01', os.path.join(SRC_DIR, '01_data_preprocessing.py'))
preprocessing01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocessing01)  # only defines functions; main() is __main__-guarded
_stdout_keepalive.append(sys.stdout)

_spec_bl = importlib.util.spec_from_file_location(
    'baseline03', os.path.join(SRC_DIR, '03_baseline_imputation.py'))
baseline03 = importlib.util.module_from_spec(_spec_bl)
_spec_bl.loader.exec_module(baseline03)
_stdout_keepalive.append(sys.stdout)

import train_saits_v2 as tsv2  # baselines/train_saits_v2.py -- helpers only, no top-level side effects

ANALYSIS_DIR = os.path.join(bco.CANON_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

N_REALIZATIONS = 10
MASK_SEED_BASE = 9000   # disjoint from all production seeds (42, 123, 456, ...)
SEEDS = [42, 123, 456]
SCENARIOS = [('10pct', 0.10), ('20pct', 0.20)]
P95_THRESH = cm.P95_THRESH


def find_best_saits_checkpoint(seed):
    """Minimum-validation-MSE checkpoint for this seed -- same selection
    rule as train_saits_v2.py's 2026-08-03 bugfix (PyPOTS's own best-model
    criterion is the lowest validation MSE across epochs, not the highest
    epoch number)."""
    pattern = os.path.join(SRC_DIR, 'experiments_dl', f'saits_v2_seed{seed}',
                           '**', 'SAITS_epoch*.pypots')
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f'no SAITS checkpoint found for seed={seed}: {pattern}')
    def mse_of(p):
        m = re.search(r'MSE([\d.]+)\.pypots', os.path.basename(p))
        return float(m.group(1)) if m else float('inf')
    return min(ckpts, key=mse_of)


def load_saits(seed):
    from pypots.imputation import SAITS
    saits = SAITS(n_steps=tsv2.N_STEPS, n_features=tsv2.N_FEATURES, **tsv2.SAITS_CFG,
                  epochs=1, patience=None, device='cpu',
                  saving_path=None, model_saving_strategy=None, verbose=False)
    ckpt = find_best_saits_checkpoint(seed)
    saits.load(ckpt)
    return saits


def saits_impute_full(saits, corrupted_norm, station_ids, station_list, sc, pidx):
    """Full-length (N,7) normalised SAITS reconstruction for an arbitrary
    corrupted array, including the overlapping tail-window fix
    (baselines/repackage_saits_outputs.py) so stations' final
    (per_station % N_STEPS) rows are not silently dropped from the
    extreme-event evaluation pool."""
    n_steps = tsv2.N_STEPS
    corr_log = tsv2.to_log1p(corrupted_norm.astype(np.float32), pidx, sc)
    X = tsv2.build_3d(corr_log, station_ids, n_steps=n_steps)
    imp3d = saits.impute({'X': X}).astype(np.float32)
    trunc_log = tsv2.reconstruct_flat(imp3d, station_ids, n_steps=n_steps)

    n_full = len(corrupted_norm)
    n_sta = len(station_list)
    per_station_full = n_full // n_sta
    n_drop = per_station_full % n_steps
    trunc_per_sta = per_station_full - n_drop

    if n_drop == 0:
        full_log = trunc_log
    else:
        tail_imp = {}
        for sid in station_list:
            sta = corr_log[station_ids == sid].astype(np.float32)
            tail = sta[-n_steps:][None, :, :]
            tail_imp[sid] = saits.impute({'X': tail}).astype(np.float32)[0]

        full_log = np.empty((n_full, tsv2.N_FEATURES), dtype=np.float32)
        for si, sid in enumerate(station_list):
            sta_rows_trunc = trunc_log[si::n_sta]
            assert len(sta_rows_trunc) == trunc_per_sta
            tail_rows = tail_imp[sid][-n_drop:]
            sta_rows_full = np.concatenate([sta_rows_trunc, tail_rows], axis=0)
            full_log[si::n_sta] = sta_rows_full

    full_mm = tsv2.log1p_to_mm(full_log, pidx)
    return full_mm[:, pidx]  # PRECIP in mm, full length, date-major/station-minor order


def main():
    print('=' * 70)
    print('  canonical_multimask_variance.py -- Section 5.9 source')
    print(f'  {N_REALIZATIONS} realizations x {len(SEEDS)} seeds per scenario, mask seeds '
         f'{MASK_SEED_BASE}..{MASK_SEED_BASE + N_REALIZATIONS - 1}')
    print('=' * 70)

    with open(os.path.join(SRC_DIR, 'adjacency.pkl'), 'rb') as f:
        adj_data = pickle.load(f)
    A_knn = adj_data['A_knn']
    station_list = adj_data['stations']
    print(f'  Loaded adjacency.pkl: stations={station_list}')

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    meteo_vars = mv
    tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')

    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)
    X_tr_full = d2s.build_occ_X(tr[d2s.TRAIN_COR_KEY].astype(np.float32),
                                tr['temporal'].astype(np.float32),
                                tr[tr_na_key].astype(np.float32),
                                tr[tr_nm_key].astype(np.float32), pidx)
    X_va_full = d2s.build_occ_X(va[d2s.TRAIN_COR_KEY].astype(np.float32),
                                va['temporal'].astype(np.float32),
                                va[tr_na_key].astype(np.float32),
                                va[tr_nm_key].astype(np.float32), pidx)
    X_tr_amt = d2s.build_amt_X(sc, tr, d2s.TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)

    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > d2s.WET_THRESH) & tr_obs
    X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs] > d2s.WET_THRESH).astype(int)
    X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs] > d2s.WET_THRESH).astype(int)

    from sklearn.ensemble import RandomForestRegressor

    dlpif_models = {}
    saits_models = {}
    for seed in SEEDS:
        occ_rf, occ_cut, vm = d2s.train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed)
        amt_rf = RandomForestRegressor(n_estimators=400, random_state=seed,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])
        dlpif_models[seed] = (occ_rf, occ_cut, amt_rf)
        print(f'  [seed={seed}] DLPIF occ RF trained: cutoff={occ_cut}  val_F1={vm["val_f1"]}')

        saits_models[seed] = load_saits(seed)
        print(f'  [seed={seed}] SAITS loaded: {os.path.basename(find_best_saits_checkpoint(seed))}')

    data_norm = te['data'].astype(np.float32)
    real_mask = te['real_mask'].astype(np.float32)
    station_ids = te['station_ids']
    temporal = te['temporal'].astype(np.float32)
    te_gt = d2s.inv(sc, data_norm)[:, pidx]

    records = []
    for scen_label, rate in SCENARIOS:
        for r in range(N_REALIZATIONS):
            seed_r = MASK_SEED_BASE + r
            corrupted_norm, art_mask = preprocessing01.random_missingness(
                data_norm, real_mask, rate, seed=seed_r)
            nbr_avg, nbr_mask = preprocessing01.compute_neighbor_avg(
                corrupted_norm, station_ids, A_knn, station_list)

            m = art_mask[:, pidx] > 0.5
            gt_m = te_gt[m]
            n_extreme = int((gt_m >= P95_THRESH).sum())

            # -- Linear (station-wise), deterministic, one row per realization --
            lin_norm = baseline03.linear_interpolation(
                tr['data'].astype(np.float32), corrupted_norm, meteo_vars, station_ids)
            lin_orig = sc.inverse_transform(np.clip(lin_norm.astype(np.float64), 0, 1))
            lin_pred_m = lin_orig[m, pidx]
            lin_metrics = cm.precip_metrics(gt_m, lin_pred_m)
            records.append(dict(scenario=scen_label, realization=r, mask_seed=seed_r,
                               method='Linear', seed='deterministic', n_masked=int(m.sum()),
                               n_extreme=n_extreme, f1=lin_metrics['f1'],
                               rmse_p95=lin_metrics['rmse_p95'], mae_p95=lin_metrics['mae_p95']))

            # -- shared feature matrices for this mask realization (seed-independent) --
            X_occ = d2s.build_occ_X(corrupted_norm, temporal, nbr_avg, nbr_mask, pidx)
            cor_orig = d2s.inv(sc, corrupted_norm).astype(np.float32)
            cor_orig[:, pidx] = 0.0
            X_amt = np.concatenate([cor_orig, temporal,
                                    d2s.inv(sc, nbr_avg).astype(np.float32),
                                    nbr_mask.astype(np.float32)], axis=1)

            for seed in SEEDS:
                occ_rf, occ_cut, amt_rf = dlpif_models[seed]
                proba = occ_rf.predict_proba(X_occ)[:, 1]
                wet_pred = (proba >= occ_cut).astype(bool)
                pred_scen = np.zeros_like(te_gt)
                apply_sel = m & wet_pred
                if apply_sel.sum() > 0:
                    pred_scen[apply_sel] = np.maximum(amt_rf.predict(X_amt[apply_sel]), 0.0)
                dlpif_metrics = cm.precip_metrics(gt_m, pred_scen[m])
                records.append(dict(scenario=scen_label, realization=r, mask_seed=seed_r,
                                   method='AmountRF_DLPIF', seed=seed, n_masked=int(m.sum()),
                                   n_extreme=n_extreme, f1=dlpif_metrics['f1'],
                                   rmse_p95=dlpif_metrics['rmse_p95'], mae_p95=dlpif_metrics['mae_p95']))

                saits_pred_full = saits_impute_full(
                    saits_models[seed], corrupted_norm, station_ids, station_list, sc, pidx)
                saits_metrics = cm.precip_metrics(gt_m, saits_pred_full[m])
                records.append(dict(scenario=scen_label, realization=r, mask_seed=seed_r,
                                   method='SAITS', seed=seed, n_masked=int(m.sum()),
                                   n_extreme=n_extreme, f1=saits_metrics['f1'],
                                   rmse_p95=saits_metrics['rmse_p95'], mae_p95=saits_metrics['mae_p95']))

            print(f'  [{scen_label} realization {r}] mask_seed={seed_r} n_masked={int(m.sum())} '
                 f'n_extreme={n_extreme}')
        print(f'  [{scen_label}] {N_REALIZATIONS} realizations x {len(SEEDS)} seeds done')

    df = pd.DataFrame(records)
    out_csv = os.path.join(ANALYSIS_DIR, 'multimask_variance.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}')

    # ── F1 summary (original Section 5.9 finding) ──────────────────────────
    f1_summary = df.groupby(['method', 'scenario'])['f1'].agg(['mean', 'std', 'count']).round(4)
    print('\n  F1 summary (mean +/- std, pooled across realizations x seeds where applicable):')
    print(f1_summary.to_string())

    md_lines = ['# Multi-Mask Variance Experiment (Section 5.9 source)', '',
               f'{N_REALIZATIONS} independent random mask realizations per scenario '
               f'(mask seeds {MASK_SEED_BASE}-{MASK_SEED_BASE + N_REALIZATIONS - 1}), '
               f'pooled across all {len(SEEDS)} model seeds ({", ".join(map(str, SEEDS))}) for '
               'AmountRF_DLPIF and SAITS (Linear is deterministic, seed-independent).',
               '', '| Method | Scenario | Mean F1 | Std F1 | N |', '|---|---|---|---|---|']
    for (method, scen), row in f1_summary.iterrows():
        md_lines.append(f"| {method} | {scen} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |")
    md_path = os.path.join(ANALYSIS_DIR, 'multimask_variance_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')

    # ── p95 (extreme-event) summary -- the strengthened analysis ───────────
    p95_summary = df.groupby(['method', 'scenario'])[['rmse_p95', 'mae_p95']].agg(['mean', 'std', 'count'])
    n_extreme_pooled = df.groupby(['method', 'scenario'])['n_extreme'].sum()
    print('\n  RMSE_p95 / MAE_p95 summary (pooled across realizations x seeds):')
    print(p95_summary.to_string())

    md2 = ['# Multi-Mask x Multi-Seed Extreme-Event (p95) Variance (Section 5.4/5.9 source)', '',
          f'{N_REALIZATIONS} independent random mask realizations per scenario x {len(SEEDS)} '
          f'model seeds ({", ".join(map(str, SEEDS))}) for AmountRF_DLPIF and SAITS -- {N_REALIZATIONS * len(SEEDS)} '
          'independent extreme-event evaluations pooled per method per scenario, versus 1 in the '
          'single-mask production tables (Table 2, Table S1). Linear is deterministic '
          f'({N_REALIZATIONS} realizations, no seed variation). Extreme threshold: '
          f'PRECIP >= {P95_THRESH} mm (canonical_metrics.P95_THRESH).',
          '',
          '| Method | Scenario | Mean RMSE_p95 | Std RMSE_p95 | Mean MAE_p95 | Std MAE_p95 | N (realizations x seeds) | Pooled n_extreme |',
          '|---|---|---|---|---|---|---|---|']
    for (method, scen) in p95_summary.index:
        rmse_mean = p95_summary.loc[(method, scen), ('rmse_p95', 'mean')]
        rmse_std = p95_summary.loc[(method, scen), ('rmse_p95', 'std')]
        mae_mean = p95_summary.loc[(method, scen), ('mae_p95', 'mean')]
        mae_std = p95_summary.loc[(method, scen), ('mae_p95', 'std')]
        n = int(p95_summary.loc[(method, scen), ('rmse_p95', 'count')])
        n_ext = int(n_extreme_pooled.loc[(method, scen)])
        md2.append(f"| {method} | {scen} | {rmse_mean:.2f} | {rmse_std:.2f} | "
                  f"{mae_mean:.2f} | {mae_std:.2f} | {n} | {n_ext} |")
    # ── Paired significance test: DLPIF vs SAITS / Linear on RMSE_p95 ──────
    # HIERARCHICAL aggregation (corrected 2026-08-04): the 3 model seeds
    # within one mask realization all evaluate the SAME masked positions
    # against the SAME ground truth (only the trained model differs), so
    # the 30 realization x seed rows are NOT 30 independent samples -- the
    # mask realization is the true independent unit, seeds are repeated
    # measures nested within it. Treating all 30 as independent (as an
    # earlier version of this script did) understates the p-values: e.g.
    # if a given mask happens to be "hard" for both DLPIF and SAITS, all
    # three of its seed-level differences point the same direction for a
    # shared reason (the mask), not three independent confirmations.
    # Fix: average the 3 seeds within each realization FIRST (one
    # DLPIF-vs-SAITS difference per realization), THEN run the paired
    # Wilcoxon signed-rank test across the resulting n=10 independent
    # realizations. Linear has no seed dimension, so its per-realization
    # value was already the correct unit and is unaffected by this fix.
    from scipy import stats
    sig_rows = []
    for scen in [s for s, _ in SCENARIOS]:
        sub = df[df.scenario == scen]
        dlpif = sub[sub.method == 'AmountRF_DLPIF'].set_index(['realization', 'seed'])['rmse_p95']
        saits = sub[sub.method == 'SAITS'].set_index(['realization', 'seed'])['rmse_p95']
        lin = sub[sub.method == 'Linear'].set_index('realization')['rmse_p95']

        both = pd.concat([dlpif, saits], axis=1, keys=['dlpif', 'saits']).dropna()
        per_seed_diff = (both['dlpif'] - both['saits']).reset_index()
        # average over seeds within each realization -> one value per mask
        per_mask_diff = per_seed_diff.groupby('realization')[0].mean()
        _, p_saits = stats.wilcoxon(per_mask_diff)
        sig_rows.append(dict(scenario=scen, comparison='AmountRF_DLPIF vs SAITS',
                             n=len(per_mask_diff),
                             mean_diff=round(float(per_mask_diff.mean()), 3), p_value=p_saits))

        dlpif_per_mask = dlpif.reset_index().groupby('realization')['rmse_p95'].mean()
        both2 = pd.concat([dlpif_per_mask, lin], axis=1, keys=['dlpif', 'lin']).dropna()
        diff2 = both2['dlpif'] - both2['lin']
        _, p_lin = stats.wilcoxon(diff2)
        sig_rows.append(dict(scenario=scen, comparison='AmountRF_DLPIF vs Linear', n=len(diff2),
                             mean_diff=round(float(diff2.mean()), 3), p_value=p_lin))

    sig_df = pd.DataFrame(sig_rows)
    sig_csv = os.path.join(ANALYSIS_DIR, 'multimask_p95_significance.csv')
    sig_df.to_csv(sig_csv, index=False)
    print('\n  Paired Wilcoxon signed-rank test, RMSE_p95, ONE value per mask realization '
         '(3 seeds averaged within each mask first -- n=10 independent masks, not 27-30):')
    print(sig_df.to_string(index=False))
    print(f'  -> {sig_csv}')

    md2.append('')
    md2.append('Paired Wilcoxon signed-rank test on RMSE_p95. The independent unit is the mask '
              'realization (n=10): for AmountRF_DLPIF vs SAITS, the 3 model seeds are averaged '
              'within each realization first (since seeds share the same masked positions and '
              'ground truth, they are repeated measures, not independent samples), then the test '
              'is run across the 10 resulting per-realization differences. Linear has no seed '
              'dimension and was already one value per realization.')
    md2.append('')
    md2.append('| Comparison | Scenario | N (independent masks) | Mean diff (A - B) | p-value |')
    md2.append('|---|---|---|---|---|')
    for _, row in sig_df.iterrows():
        md2.append(f"| {row['comparison']} | {row['scenario']} | {int(row['n'])} | "
                  f"{row['mean_diff']:+.3f} | {row['p_value']:.3e} |")

    md2_path = os.path.join(ANALYSIS_DIR, 'multimask_p95_variance_summary.md')
    with open(md2_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md2) + '\n')
    print(f'  -> {md2_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()

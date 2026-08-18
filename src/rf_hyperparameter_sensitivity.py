# -*- coding: utf-8 -*-
"""
rf_hyperparameter_sensitivity.py
====================================
R8 follow-up -- Sections 4.2/4.3/4.8 repeatedly note Stage 1 (300 trees,
min_samples_leaf=5) and Stage 2 (400 trees, min_samples_leaf=2) use
"fixed, manually-chosen defaults rather than a searched configuration;
sensitivity to alternative choices is untested." This script provides a
minimal, disclosed sensitivity check -- not a full grid search or
re-selection of hyperparameters -- varying n_estimators by roughly +/-50%
around each stage's default, holding every other hyperparameter (and the
25/26-feature schema, training data, threshold-selection procedure)
identical to Sections 4.2-4.3.

Design
------
- Stage 1 n_estimators in {150, 300 (default), 450}; min_samples_leaf
  fixed at 5 throughout (varying two hyperparameters jointly would
  conflate their individual effects, and tree count is the more commonly
  scrutinised RF regularisation choice for a fixed leaf size).
- Stage 2 n_estimators in {200, 400 (default), 600}; min_samples_leaf
  fixed at 2 throughout.
- Both stages are varied INDEPENDENTLY (Stage 1 sweep uses Stage 2's
  default regressor and vice versa) so any performance change is
  attributable to the specific stage being swept, not a confound between
  the two.
- Evaluated on the two representative scenarios already used in Table
  6.1 (10pct, block30d), across all 3 model seeds, using the identical
  validation-F1-maximising threshold-selection procedure (Section 4.2)
  re-run per (stage, n_estimators, seed) combination -- not the frozen
  default-configuration threshold reused unchanged, since a different
  tree count can shift the validation-optimal cutoff.

Output: results/rf_hyperparameter_sensitivity.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score

import direct_two_stage_rf as d2s

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SRC_DIR, '..', 'results', 'rf_hyperparameter_sensitivity.csv')

SEEDS = [42, 123, 456]
SCENARIOS = ['10pct', 'block30d']
STAGE1_DEFAULT_TREES = 300
STAGE2_DEFAULT_TREES = 400
STAGE1_SWEEP = [150, 300, 450]
STAGE2_SWEEP = [200, 400, 600]


def train_occ(X_tr, y_tr, X_va, y_va, seed, n_estimators):
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=5,
                                class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    va_p = rf.predict_proba(X_va)[:, 1]
    best_f1, best_cut = -1.0, 0.5
    for cut in np.arange(0.20, 0.82, 0.02):
        f = f1_score(y_va, (va_p >= cut).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_cut = f, float(cut)
    return rf, round(best_cut, 3)


def eval_scenario(occ_rf, cut, amt_rf, scen, sc, pidx, te, temporal_key='temporal'):
    cor_key, mask_key = {label: (ck, mk) for label, ck, mk in d2s.SCENARIOS}[scen]
    na_key, nm_key = d2s.neighbor_keys(scen)
    X_occ = d2s.build_occ_X(te[cor_key].astype(np.float32), te[temporal_key].astype(np.float32),
                            te[na_key].astype(np.float32), te[nm_key].astype(np.float32), pidx)
    X_amt = d2s.build_amt_X(sc, te, cor_key, pidx, na_key, nm_key)
    m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
    y_true = te_gt[m]
    raw_p = occ_rf.predict_proba(X_occ[m])[:, 1]
    wet_pred = raw_p >= cut
    yhat = np.zeros(len(y_true))
    if wet_pred.sum() > 0:
        yhat[wet_pred] = np.maximum(0.0, amt_rf.predict(X_amt[m][wet_pred]))
    gw = y_true > d2s.WET_THRESH
    f1 = f1_score(gw, wet_pred, zero_division=0)
    ws = gw
    rmse_wet = float(np.sqrt(np.mean((yhat[ws] - y_true[ws]) ** 2))) if ws.sum() else np.nan
    bias = float(wet_pred.mean() - gw.mean())
    return dict(f1=round(float(f1), 4), rmse_wet=round(rmse_wet, 4) if rmse_wet == rmse_wet else np.nan,
               bias=round(bias, 4), n=int(m.sum()))


def main():
    print('=' * 70)
    print('  R8 -- RANDOM FOREST HYPERPARAMETER (n_estimators) SENSITIVITY')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')

    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)
    X_tr_occ_full = d2s.build_occ_X(tr[d2s.TRAIN_COR_KEY].astype(np.float32), tr['temporal'].astype(np.float32),
                                    tr[tr_na_key].astype(np.float32), tr[tr_nm_key].astype(np.float32), pidx)
    X_va_occ_full = d2s.build_occ_X(va[d2s.TRAIN_COR_KEY].astype(np.float32), va['temporal'].astype(np.float32),
                                    va[tr_na_key].astype(np.float32), va[tr_nm_key].astype(np.float32), pidx)
    X_tr_amt = d2s.build_amt_X(sc, tr, d2s.TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)

    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > d2s.WET_THRESH) & tr_obs

    X_tr_occ = X_tr_occ_full[tr_obs]; y_tr = (tr_gt[tr_obs] > d2s.WET_THRESH).astype(int)
    X_va_occ = X_va_occ_full[va_obs]; y_va = (va_gt[va_obs] > d2s.WET_THRESH).astype(int)

    rows = []

    # -- Stage 1 sweep (Stage 2 held at its default 400 trees) --
    print('\n  --- Stage 1 (occurrence classifier) n_estimators sweep ---')
    for n_est in STAGE1_SWEEP:
        for seed in SEEDS:
            occ_rf, cut = train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed, n_est)
            amt_rf = RandomForestRegressor(n_estimators=STAGE2_DEFAULT_TREES, random_state=seed,
                                           min_samples_leaf=2, n_jobs=-1)
            amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])
            for scen in SCENARIOS:
                r = eval_scenario(occ_rf, cut, amt_rf, scen, sc, pidx, te)
                r.update(sweep='stage1_trees', stage1_trees=n_est, stage2_trees=STAGE2_DEFAULT_TREES,
                         seed=seed, scenario=scen, cutoff=cut)
                rows.append(r)
                print(f'    stage1_trees={n_est:4d}  seed={seed}  [{scen:10s}] F1={r["f1"]:.4f} '
                     f'RMSE_wet={r["rmse_wet"]}')

    # -- Stage 2 sweep (Stage 1 held at its default 300 trees) --
    print('\n  --- Stage 2 (amount regressor) n_estimators sweep ---')
    occ_by_seed = {}
    for seed in SEEDS:
        occ_rf, cut = train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed, STAGE1_DEFAULT_TREES)
        occ_by_seed[seed] = (occ_rf, cut)
    for n_est in STAGE2_SWEEP:
        for seed in SEEDS:
            occ_rf, cut = occ_by_seed[seed]
            amt_rf = RandomForestRegressor(n_estimators=n_est, random_state=seed,
                                           min_samples_leaf=2, n_jobs=-1)
            amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])
            for scen in SCENARIOS:
                r = eval_scenario(occ_rf, cut, amt_rf, scen, sc, pidx, te)
                r.update(sweep='stage2_trees', stage1_trees=STAGE1_DEFAULT_TREES, stage2_trees=n_est,
                         seed=seed, scenario=scen, cutoff=cut)
                rows.append(r)
                print(f'    stage2_trees={n_est:4d}  seed={seed}  [{scen:10s}] F1={r["f1"]:.4f} '
                     f'RMSE_wet={r["rmse_wet"]}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f'\n  -> {OUT_PATH}')

    print('\n  Stage 1 sweep (mean over 3 seeds):')
    print(df[df.sweep == 'stage1_trees'].groupby(['stage1_trees', 'scenario'])[['f1', 'rmse_wet']].mean().round(4))
    print('\n  Stage 2 sweep (mean over 3 seeds):')
    print(df[df.sweep == 'stage2_trees'].groupby(['stage2_trees', 'scenario'])[['f1', 'rmse_wet']].mean().round(4))
    print('=' * 70)


if __name__ == '__main__':
    main()

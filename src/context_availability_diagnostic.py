"""
context_availability_diagnostic.py
=====================================
RQ4 (diagnostic phase, before any gate/fallback is written) -- does
observational context availability actually predict where the in-situ
DLPIF occurrence/amount pathway becomes unreliable? Companion to RQ1
(tag rq1-freeze), RQ2 (tag rq2-freeze), RQ3 (tag rq3-freeze).

Per-masked-position availability scores (no model involved in computing
these -- pure data-availability bookkeeping):
  A_local    = fraction of the 6 non-PRECIP local meteorological variables
               (TMIN, TMEAN, TMAX, RH_MEAN, P_MEAN, WIND_MEAN) actually
               present in that scenario's corrupted array at that row
               (i.e. not NaN -- combines natural missingness AND that
               scenario's artificial mask, exactly what build_occ_X /
               build_amt_X actually receive before their own
               nan_to_num(nan=0) fill).
  A_neighbor = fraction of the 7 neighbor_mask_<scenario> columns that are
               1 at that row (>=1 neighbouring station has that variable
               available) -- exactly what Stage-1/Stage-2's neighbour
               features already encode.
  A_total    = 0.5*A_local + 0.5*A_neighbor (simple, unweighted composite;
               NOT a gate threshold decision -- this script only asks
               whether availability predicts reliability at all, before
               any gate design commits to a specific weighting).

Pooled across all 8 test scenarios x 3 seeds (one row per masked PRECIP
position x seed), binned into 5 fixed-width availability bins
[0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1], reporting n / F1 / Brier / ECE /
RMSE_wet / wet-frequency-bias per bin -- using the RQ2-frozen Platt-
calibrated occurrence probability + frozen threshold (never refit) and a
freshly-trained Stage-2 amount regressor (same deterministic procedure as
RQ3, never persisted anywhere in this repo).

This script does NOT build a gate or fallback -- it only characterizes
whether/where a "reliability cliff" exists, so that if a gate is built
next, its threshold is empirically motivated rather than assumed.

Outputs (results/rq4_context_availability/):
  position_level_availability.csv     one row per (seed, scenario, masked
                                       position): A_local, A_neighbor,
                                       A_total, y_true, p_calibrated,
                                       wet_pred, y_hat_amount
  bins_by_a_total.csv                 n/F1/Brier/ECE/RMSE_wet/bias per
                                       A_total bin (primary table)
  bins_by_a_local.csv                 n/F1 per A_local bin
  bins_by_a_neighbor.csv              n/F1 per A_neighbor bin
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import brier_score_loss, f1_score

import direct_two_stage_rf as d2s
from calibrate_occurrence_probability import apply_platt, expected_calibration_error

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq4_context_availability')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
WET_THRESH = d2s.WET_THRESH
TEST_SCENARIOS = ['10pct', 'mar_meteo', 'mnar_wet',
                   'mnar_intensity_moderate', 'mnar_intensity_severe',
                   'block7d', 'block30d', 'netblock30d']
BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]  # +epsilon so A_total==1.0 falls in the last bin
BIN_LABELS = ['[0,.2)', '[.2,.4)', '[.4,.6)', '[.6,.8)', '[.8,1]']


def main():
    print('=' * 70)
    print('  RQ4 DIAGNOSTIC -- does context availability predict reliability?')
    print('  (no gate/fallback built here -- characterization only)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    tr = d2s.load_npz('train')
    te = d2s.load_npz('test')
    precip_local_cols = [i for i in range(len(mv)) if i != pidx]  # 6 non-PRECIP local vars

    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)
    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs
    X_tr_amt = d2s.build_amt_X(sc, tr, d2s.TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)

    all_rows = []

    for seed in SEEDS:
        print(f'\n  seed={seed} ...')
        with open(os.path.join(SRC_DIR, f'direct_two_stage_occurrence_seed{seed}.pkl'), 'rb') as f:
            occ_rf = pickle.load(f)['rf']
        with open(os.path.join(RQ2_DIR, 'calibration_models', f'platt_seed{seed}.pkl'), 'rb') as f:
            platt = pickle.load(f)
        thresholds = pd.read_csv(os.path.join(RQ2_DIR, 'thresholds.csv'))
        tau_platt = float(thresholds[(thresholds.seed == seed) & (thresholds.variant == 'platt')]['threshold'].iloc[0])

        amt_rf = RandomForestRegressor(n_estimators=400, random_state=seed,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

        for scen_label, cor_key, mask_key in d2s.SCENARIOS:
            if scen_label not in TEST_SCENARIOS:
                continue
            na_key, nm_key = d2s.neighbor_keys(scen_label)
            corrupted = te[cor_key].astype(np.float32)
            neighbor_mask = te[nm_key].astype(np.float32)
            X_occ = d2s.build_occ_X(corrupted, te['temporal'].astype(np.float32),
                                    te[na_key].astype(np.float32), neighbor_mask, pidx)
            X_amt = d2s.build_amt_X(sc, te, cor_key, pidx, na_key, nm_key)
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]

            a_local = np.mean(~np.isnan(corrupted[:, precip_local_cols]), axis=1)
            a_neighbor = np.mean(neighbor_mask, axis=1)
            a_total = 0.5 * a_local + 0.5 * a_neighbor

            y_true = te_gt[m]
            raw_p = occ_rf.predict_proba(X_occ[m])[:, 1]
            p_cal = apply_platt(platt, raw_p)
            wet_pred = p_cal >= tau_platt
            yhat = np.zeros_like(y_true)
            if wet_pred.sum() > 0:
                yhat[wet_pred] = amt_rf.predict(X_amt[m][wet_pred])

            all_rows.append(pd.DataFrame(dict(
                seed=seed, scenario=scen_label,
                a_local=a_local[m], a_neighbor=a_neighbor[m], a_total=a_total[m],
                y_true=y_true, p_calibrated=p_cal, wet_pred=wet_pred, y_hat_amount=yhat,
            )))

    df = pd.concat(all_rows, ignore_index=True)
    df['is_wet_true'] = df['y_true'] > WET_THRESH
    pos_path = os.path.join(OUT_DIR, 'position_level_availability.csv')
    df.to_csv(pos_path, index=False)
    print(f'\n  -> {pos_path}  ({len(df)} rows, pooled across {len(SEEDS)} seeds x {len(TEST_SCENARIOS)} scenarios)')

    def bin_metrics(sub):
        n = len(sub)
        if n == 0:
            return pd.Series(dict(n=0, f1=np.nan, brier=np.nan, ece=np.nan,
                                  rmse_wet=np.nan, wet_freq_bias=np.nan))
        f1 = f1_score(sub['is_wet_true'], sub['wet_pred'], zero_division=0)
        brier = brier_score_loss(sub['is_wet_true'], sub['p_calibrated'])
        ece, _ = expected_calibration_error(sub['p_calibrated'].to_numpy(), sub['is_wet_true'].to_numpy())
        wet_sel = sub['is_wet_true']
        rmse_wet = (np.sqrt(np.mean((sub.loc[wet_sel, 'y_hat_amount'] - sub.loc[wet_sel, 'y_true']) ** 2))
                   if wet_sel.sum() > 0 else np.nan)
        bias = sub['wet_pred'].mean() - sub['is_wet_true'].mean()
        return pd.Series(dict(n=n, f1=round(f1, 4), brier=round(brier, 4),
                              ece=round(float(ece), 4),
                              rmse_wet=round(float(rmse_wet), 4) if rmse_wet == rmse_wet else np.nan,
                              wet_freq_bias=round(float(bias), 4)))

    df['bin_total'] = pd.cut(df['a_total'], bins=BIN_EDGES, labels=BIN_LABELS, right=False, include_lowest=True)
    df['bin_local'] = pd.cut(df['a_local'], bins=BIN_EDGES, labels=BIN_LABELS, right=False, include_lowest=True)
    df['bin_neighbor'] = pd.cut(df['a_neighbor'], bins=BIN_EDGES, labels=BIN_LABELS, right=False, include_lowest=True)

    bins_total = df.groupby('bin_total', observed=False).apply(bin_metrics, include_groups=False).reindex(BIN_LABELS)
    bins_total.to_csv(os.path.join(OUT_DIR, 'bins_by_a_total.csv'))
    print('\n  A_total bins (primary table):')
    print(bins_total)

    def f1_only(sub):
        n = len(sub)
        if n == 0:
            return pd.Series(dict(n=0, f1=np.nan))
        return pd.Series(dict(n=n, f1=round(f1_score(sub['is_wet_true'], sub['wet_pred'], zero_division=0), 4)))

    bins_local = df.groupby('bin_local', observed=False).apply(f1_only, include_groups=False).reindex(BIN_LABELS)
    bins_local.to_csv(os.path.join(OUT_DIR, 'bins_by_a_local.csv'))
    print('\n  A_local bins:')
    print(bins_local)

    bins_neighbor = df.groupby('bin_neighbor', observed=False).apply(f1_only, include_groups=False).reindex(BIN_LABELS)
    bins_neighbor.to_csv(os.path.join(OUT_DIR, 'bins_by_a_neighbor.csv'))
    print('\n  A_neighbor bins:')
    print(bins_neighbor)

    print(f'\n  -> {os.path.join(OUT_DIR, "bins_by_a_total.csv")}')
    print(f'  -> {os.path.join(OUT_DIR, "bins_by_a_local.csv")}')
    print(f'  -> {os.path.join(OUT_DIR, "bins_by_a_neighbor.csv")}')
    print('=' * 70)


if __name__ == '__main__':
    main()

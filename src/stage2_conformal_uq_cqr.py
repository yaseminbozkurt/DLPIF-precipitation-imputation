"""
stage2_conformal_uq_cqr.py
============================
RQ3 robustness companion -- Conformalized Quantile Regression (CQR) via a
Quantile Regression Forest (QRF), evaluated against the SAME populations,
scenarios, seeds, and frozen Stage-1 gate as stage2_conformal_uq.py's
symmetric split-conformal baseline (tag `rq3-freeze`). This script does not
modify or re-run that baseline; it is an independent, additive comparison
answering the "next step" flagged in the manuscript's own Limitations
(Sections 5.3, 7.6): does an adaptive-width interval recover the coverage
symmetric split-conformal loses under MNAR-Intensity stress and at extreme
(p95) magnitudes?

Method (Romano, Patterson & Candes, 2019, "Conformalized Quantile
Regression"; QRF as the base quantile estimator, Meinshausen 2006):
  1. Fit a RandomForestQuantileRegressor (400 trees, min_samples_leaf=2,
     identical hyperparameters/seed/training rows to the point-estimate
     Stage-2 regressor in stage2_conformal_uq.py) on TRAIN wet rows.
  2. On VAL-CAL wet rows (same VAL-CAL partition reused from RQ2/RQ3),
     predict the two-sided (alpha/2, 1-alpha/2) conditional quantiles and
     compute the CQR nonconformity score
         E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
  3. Conformal correction Q = the same finite-sample-corrected quantile of
     E_i used throughout this study (ceil((n+1)(1-alpha))/n).
  4. Test interval: [max(0, q_lo(x) - Q), q_hi(x) + Q].
Everything else (ORACLE vs END-TO-END populations, frozen Platt-calibrated
Stage-1 gate, 8 test scenarios, 3 model seeds, metrics) is unchanged from
stage2_conformal_uq.py, so the two scripts' outputs are directly comparable
cell-for-cell.

Outputs (results/rq3_conformal_cqr/ -- kept separate from results/rq3_conformal/
so the original frozen symmetric-conformal results are never overwritten):
  conformal_cqr_calibration_summary.csv
  test_conformal_cqr_oracle.csv
  test_conformal_cqr_endtoend.csv
"""
import os
import pickle
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

import direct_two_stage_rf as d2s
from calibrate_occurrence_probability import apply_platt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq3_conformal_cqr')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
ALPHA = 0.10  # nominal 90% coverage, identical to stage2_conformal_uq.py
Q_LO, Q_HI = ALPHA / 2, 1 - ALPHA / 2  # 0.05 / 0.95
WET_THRESH = d2s.WET_THRESH
P95_THRESH = d2s.P95_THRESH
TEST_SCENARIOS = ['10pct', 'mar_meteo', 'mnar_wet',
                   'mnar_intensity_moderate', 'mnar_intensity_severe',
                   'block7d', 'block30d', 'netblock30d']


def conformal_quantile(scores, alpha=ALPHA):
    """Identical finite-sample split-conformal quantile formula used
    throughout this study (stage2_conformal_uq.py, calibrate_occurrence_probability.py)."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))


def interval_score(y, lo, hi, alpha=ALPHA):
    width = hi - lo
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    return width + (2.0 / alpha) * below + (2.0 / alpha) * above


def picp(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def population_metrics(y, yhat, lo, hi, extreme_mask):
    out = dict(
        n=len(y),
        picp=picp(y, lo, hi),
        mpiw=float(np.mean(hi - lo)),
        interval_score=float(np.mean(interval_score(y, lo, hi))),
        rmse=float(np.sqrt(np.mean((yhat - y) ** 2))) if len(y) else np.nan,
    )
    if extreme_mask.sum() > 0:
        out['n_p95'] = int(extreme_mask.sum())
        out['picp_p95'] = picp(y[extreme_mask], lo[extreme_mask], hi[extreme_mask])
        out['mpiw_p95'] = float(np.mean((hi - lo)[extreme_mask]))
    else:
        out['n_p95'] = 0
        out['picp_p95'] = np.nan
        out['mpiw_p95'] = np.nan
    return out


def main():
    print('=' * 70)
    print('  RQ3 ROBUSTNESS -- CONFORMALIZED QUANTILE REGRESSION (QRF-based)')
    print(f'  (nominal coverage 1-alpha={1-ALPHA:.0%}, adaptive-width CQR;'
          f' companion to the symmetric split-conformal baseline)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    tr = d2s.load_npz('train')
    va = d2s.load_npz('val')
    te = d2s.load_npz('test')

    manifest = pd.read_csv(os.path.join(RQ2_DIR, 'validation_split_manifest.csv'),
                           parse_dates=['date'])
    cal_dates = set(manifest.loc[manifest['split'] == 'VAL-CAL', 'date'])
    va_dates = pd.to_datetime(va['dates'])
    is_cal_row = pd.Series(va_dates).isin(cal_dates).to_numpy()
    print(f'  Reusing RQ2/RQ3 VAL-CAL partition: {len(cal_dates)} dates')

    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)
    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs
    X_tr_amt = d2s.build_amt_X(sc, tr, d2s.TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)

    X_va_amt = d2s.build_amt_X(sc, va, d2s.TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)

    calib_rows, oracle_rows, e2e_rows = [], [], []

    for seed in SEEDS:
        print(f'\n{"="*70}\n  SEED = {seed}\n{"="*70}')

        with open(os.path.join(SRC_DIR, f'direct_two_stage_occurrence_seed{seed}.pkl'), 'rb') as f:
            occ_rf = pickle.load(f)['rf']
        with open(os.path.join(RQ2_DIR, 'calibration_models', f'platt_seed{seed}.pkl'), 'rb') as f:
            platt = pickle.load(f)
        thresholds = pd.read_csv(os.path.join(RQ2_DIR, 'thresholds.csv'))
        tau_platt = float(thresholds[(thresholds.seed == seed) & (thresholds.variant == 'platt')]['threshold'].iloc[0])

        # -- QRF Stage-2 amount regressor: identical hyperparameters, seed,
        #    and training rows (TRAIN wet only) as the point-estimate RF in
        #    stage2_conformal_uq.py / direct_two_stage_rf.py. Only the
        #    estimator class differs (quantile-capable vs point-only). --
        qrf = RandomForestQuantileRegressor(n_estimators=400, random_state=seed,
                                            min_samples_leaf=2, n_jobs=-1)
        qrf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

        # -- CQR calibration on VAL-CAL wet rows --
        mask_cal_wet = va_obs & is_cal_row & (va_gt > WET_THRESH)
        q_cal = qrf.predict(X_va_amt[mask_cal_wet], quantiles=[Q_LO, Q_HI])
        q_lo_cal, q_hi_cal = q_cal[:, 0], q_cal[:, 1]
        y_cal = va_gt[mask_cal_wet]
        scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
        Q = conformal_quantile(scores)
        print(f'  VAL-CAL wet n={mask_cal_wet.sum()}  CQR correction Q={Q:.4f} mm '
              f'(median raw QRF width={np.median(q_hi_cal - q_lo_cal):.4f} mm)')
        calib_rows.append(dict(seed=seed, n_cal_wet=int(mask_cal_wet.sum()), Q=Q,
                               nominal_coverage=1 - ALPHA))

        for scen_label, cor_key, mask_key in d2s.SCENARIOS:
            if scen_label not in TEST_SCENARIOS:
                continue
            na_key, nm_key = d2s.neighbor_keys(scen_label)
            X_te_occ = d2s.build_occ_X(te[cor_key].astype(np.float32), te['temporal'].astype(np.float32),
                                       te[na_key].astype(np.float32), te[nm_key].astype(np.float32), pidx)
            X_te_amt = d2s.build_amt_X(sc, te, cor_key, pidx, na_key, nm_key)
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]

            # ORACLE population
            oracle_mask = m & (te_gt > WET_THRESH)
            y_o = te_gt[oracle_mask]
            q_o = qrf.predict(X_te_amt[oracle_mask], quantiles=[Q_LO, 0.5, Q_HI])
            yhat_o = q_o[:, 1]  # median as the point estimate, for RMSE only
            lo_o = np.maximum(0.0, q_o[:, 0] - Q)
            hi_o = q_o[:, 2] + Q
            extreme_o = y_o >= P95_THRESH
            row_o = population_metrics(y_o, yhat_o, lo_o, hi_o, extreme_o)
            row_o.update(seed=seed, scenario=scen_label, population='oracle')
            oracle_rows.append(row_o)

            # END-TO-END population, gated by the SAME frozen Stage-1 decision
            y_e = te_gt[m]
            raw_p = occ_rf.predict_proba(X_te_occ[m])[:, 1]
            p_cal = apply_platt(platt, raw_p)
            wet_pred = p_cal >= tau_platt
            yhat_e = np.zeros_like(y_e)
            lo_e = np.zeros_like(y_e)
            hi_e = np.zeros_like(y_e)
            if wet_pred.sum() > 0:
                q_e = qrf.predict(X_te_amt[m][wet_pred], quantiles=[Q_LO, 0.5, Q_HI])
                yhat_e[wet_pred] = q_e[:, 1]
                lo_e[wet_pred] = np.maximum(0.0, q_e[:, 0] - Q)
                hi_e[wet_pred] = q_e[:, 2] + Q
            extreme_e = y_e >= P95_THRESH
            row_e = population_metrics(y_e, yhat_e, lo_e, hi_e, extreme_e)
            row_e.update(seed=seed, scenario=scen_label, population='end-to-end',
                         frac_predicted_wet=round(float(wet_pred.mean()), 4))
            e2e_rows.append(row_e)

            print(f'    [{scen_label:24s}] oracle: n={row_o["n"]:4d} PICP={row_o["picp"]:.3f} '
                  f'PICP_p95={row_o["picp_p95"]:.3f}  |  '
                  f'e2e: PICP={row_e["picp"]:.3f} PICP_p95={row_e["picp_p95"]:.3f}')

    calib_df = pd.DataFrame(calib_rows)
    oracle_df = pd.DataFrame(oracle_rows)
    e2e_df = pd.DataFrame(e2e_rows)
    calib_df.to_csv(os.path.join(OUT_DIR, 'conformal_cqr_calibration_summary.csv'), index=False)
    oracle_df.to_csv(os.path.join(OUT_DIR, 'test_conformal_cqr_oracle.csv'), index=False)
    e2e_df.to_csv(os.path.join(OUT_DIR, 'test_conformal_cqr_endtoend.csv'), index=False)

    print(f'\n  -> {OUT_DIR}\\conformal_cqr_calibration_summary.csv')
    print(f'  -> {OUT_DIR}\\test_conformal_cqr_oracle.csv')
    print(f'  -> {OUT_DIR}\\test_conformal_cqr_endtoend.csv')

    order = TEST_SCENARIOS
    for name, df in [('ORACLE', oracle_df), ('END-TO-END', e2e_df)]:
        print(f'\n  {name} (mean over 3 seeds):')
        agg = df.groupby('scenario')[['picp', 'picp_p95', 'mpiw', 'interval_score', 'rmse']].mean()
        agg = agg.reindex(order)
        print(agg.round(4))
    print('=' * 70)


if __name__ == '__main__':
    main()

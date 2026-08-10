"""
stage2_conformal_uq.py
========================
RQ3 -- Stage-2 (conditional amount regression) predictive uncertainty via
split-conformal prediction intervals. Companion to RQ1 (mechanism
sensitivity, tag `rq1-freeze`) and RQ2 (Stage-1 probability calibration,
tag `rq2-freeze`).

Design (agreed with user before implementation)
-------------------------------------------------
- Conformal calibration is fit on VAL-CAL WET observations only
  (Y>0.1mm) -- the same chronological unique-date VAL-CAL/VAL-SELECT
  partition frozen in results/rq2_calibration/validation_split_manifest.csv,
  reused here (not recomputed) so RQ2 and RQ3 share one validation split.
- Stage-2 amount regressor: direct_two_stage_rf.py never persists its
  RandomForestRegressor (only the Stage-1 occurrence classifier is
  pickled, to direct_two_stage_occurrence_seed{seed}.pkl -- reused here
  unmodified, exactly as RQ2 did). Stage-2 is retrained with the IDENTICAL
  procedure/data/seed direct_two_stage_rf.py already uses (n_estimators=400,
  min_samples_leaf=2, random_state=seed, fit on TRAIN wet rows only) --
  this is not a new/different model, just the same deterministic training
  call direct_two_stage_rf.py itself makes on every run.
- First-pass UQ method: plain SYMMETRIC split-conformal on absolute
  residuals (nominal 1-alpha=0.90). This is a transparent baseline, not a
  claim that a symmetric interval is ideal for skewed positive rainfall --
  if coverage/width prove inadequate, CQR/QRF is the documented next step.
- TWO separate coverage populations, kept apart on purpose (isolates
  Stage-2 conformal quality from Stage-1 occurrence-classification error):
    ORACLE  -- ground-truth-wet masked positions (Y_true>0.1mm), amt_rf +
               interval applied regardless of what Stage-1 would have
               predicted. Answers: "given Stage-2 is asked to do its job,
               how good is its uncertainty?"
    END-TO-END -- ALL masked positions, gated by the frozen RQ2 Stage-1
               decision (Platt-calibrated probability + frozen threshold,
               loaded from results/rq2_calibration/{calibration_models,
               thresholds.csv}). Positions Stage-1 calls dry get the
               two-stage architecture's actual output: a point prediction
               of exactly 0mm with a degenerate {0} interval (matching
               direct_two_stage_rf.py's own pred_scen = zeros_like(...)
               behaviour) -- so a true wet/extreme day Stage-1 misses
               contributes a real non-coverage, not an excluded case.
               Answers: "how much of the full pipeline's nominal 90%
               coverage promise survives Stage-1's own errors?"
- p95 PICP is reported separately from overall wet PICP in both
  populations (RQ1's motivating finding: MNAR-Intensity masks are
  disproportionately drawn from large events).

Outputs (results/rq3_conformal/, a new result family):
  conformal_calibration_summary.csv   q (per seed), n_cal, nominal coverage
  test_conformal_oracle.csv           PICP/PICP_p95/MPIW/Interval Score/
                                       RMSE per scenario x seed, oracle pop.
  test_conformal_endtoend.csv         same metrics, end-to-end population
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import direct_two_stage_rf as d2s
from calibrate_occurrence_probability import apply_platt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq3_conformal')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
ALPHA = 0.10  # nominal 90% coverage
WET_THRESH = d2s.WET_THRESH
P95_THRESH = d2s.P95_THRESH
TEST_SCENARIOS = ['10pct', 'mar_meteo', 'mnar_wet',
                   'mnar_intensity_moderate', 'mnar_intensity_severe',
                   'block7d', 'block30d', 'netblock30d']


def conformal_quantile(residuals, alpha=ALPHA):
    """Finite-sample split-conformal quantile: ceil((n+1)(1-alpha))/n."""
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level, method='higher'))


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
    print('  RQ3 -- STAGE-2 SPLIT-CONFORMAL UNCERTAINTY QUANTIFICATION')
    print(f'  (nominal coverage 1-alpha={1-ALPHA:.0%}, symmetric residual conformal)')
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
    print(f'  Reusing RQ2 VAL-CAL partition: {len(cal_dates)} dates')

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

        # -- Stage-1 occurrence: reuse the RQ2-frozen pretrained RF + Platt
        #    mapping + frozen threshold (never refit here). --
        with open(os.path.join(SRC_DIR, f'direct_two_stage_occurrence_seed{seed}.pkl'), 'rb') as f:
            occ_rf = pickle.load(f)['rf']
        with open(os.path.join(RQ2_DIR, 'calibration_models', f'platt_seed{seed}.pkl'), 'rb') as f:
            platt = pickle.load(f)
        thresholds = pd.read_csv(os.path.join(RQ2_DIR, 'thresholds.csv'))
        tau_platt = float(thresholds[(thresholds.seed == seed) & (thresholds.variant == 'platt')]['threshold'].iloc[0])

        # -- Stage-2 amount regressor: same training call direct_two_stage_rf.py
        #    itself makes (never persisted anywhere in this repo). --
        amt_rf = RandomForestRegressor(n_estimators=400, random_state=seed,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

        # -- Conformal calibration on VAL-CAL WET rows only --
        mask_cal_wet = va_obs & is_cal_row & (va_gt > WET_THRESH)
        yhat_cal = amt_rf.predict(X_va_amt[mask_cal_wet])
        residuals = np.abs(va_gt[mask_cal_wet] - yhat_cal)
        q = conformal_quantile(residuals)
        print(f'  VAL-CAL wet n={mask_cal_wet.sum()}  conformal q={q:.4f} mm')
        calib_rows.append(dict(seed=seed, n_cal_wet=int(mask_cal_wet.sum()), q=q,
                               nominal_coverage=1 - ALPHA))

        # -- Frozen application to TEST scenarios --
        for scen_label, cor_key, mask_key in d2s.SCENARIOS:
            if scen_label not in TEST_SCENARIOS:
                continue
            na_key, nm_key = d2s.neighbor_keys(scen_label)
            X_te_occ = d2s.build_occ_X(te[cor_key].astype(np.float32), te['temporal'].astype(np.float32),
                                       te[na_key].astype(np.float32), te[nm_key].astype(np.float32), pidx)
            X_te_amt = d2s.build_amt_X(sc, te, cor_key, pidx, na_key, nm_key)
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]

            # ORACLE population: ground-truth-wet masked positions.
            oracle_mask = m & (te_gt > WET_THRESH)
            y_o = te_gt[oracle_mask]
            yhat_o = amt_rf.predict(X_te_amt[oracle_mask])
            lo_o = np.maximum(0.0, yhat_o - q)
            hi_o = yhat_o + q
            extreme_o = y_o >= P95_THRESH
            row_o = population_metrics(y_o, yhat_o, lo_o, hi_o, extreme_o)
            row_o.update(seed=seed, scenario=scen_label, population='oracle')
            oracle_rows.append(row_o)

            # END-TO-END population: ALL masked positions, gated by frozen
            # Stage-1 decision. Predicted-dry -> degenerate {0} interval,
            # matching direct_two_stage_rf.py's actual zero-fill behaviour.
            y_e = te_gt[m]
            raw_p = occ_rf.predict_proba(X_te_occ[m])[:, 1]
            p_cal = apply_platt(platt, raw_p)
            wet_pred = p_cal >= tau_platt
            yhat_e = np.zeros_like(y_e)
            lo_e = np.zeros_like(y_e)
            hi_e = np.zeros_like(y_e)
            if wet_pred.sum() > 0:
                amt_pred = amt_rf.predict(X_te_amt[m][wet_pred])
                yhat_e[wet_pred] = amt_pred
                lo_e[wet_pred] = np.maximum(0.0, amt_pred - q)
                hi_e[wet_pred] = amt_pred + q
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
    calib_df.to_csv(os.path.join(OUT_DIR, 'conformal_calibration_summary.csv'), index=False)
    oracle_df.to_csv(os.path.join(OUT_DIR, 'test_conformal_oracle.csv'), index=False)
    e2e_df.to_csv(os.path.join(OUT_DIR, 'test_conformal_endtoend.csv'), index=False)

    print(f'\n  -> {OUT_DIR}\\conformal_calibration_summary.csv')
    print(f'  -> {OUT_DIR}\\test_conformal_oracle.csv')
    print(f'  -> {OUT_DIR}\\test_conformal_endtoend.csv')

    order = TEST_SCENARIOS
    for name, df in [('ORACLE', oracle_df), ('END-TO-END', e2e_df)]:
        print(f'\n  {name} (mean over 3 seeds):')
        agg = df.groupby('scenario')[['picp', 'picp_p95', 'mpiw', 'interval_score', 'rmse']].mean()
        agg = agg.reindex(order)
        print(agg.round(4))
    print('=' * 70)


if __name__ == '__main__':
    main()

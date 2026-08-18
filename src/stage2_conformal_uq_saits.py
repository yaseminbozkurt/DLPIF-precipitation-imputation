"""
stage2_conformal_uq_saits.py
==============================
RQ3 backbone-generalisation check -- applies the IDENTICAL symmetric
split-conformal construction used in stage2_conformal_uq.py (DLPIF's
Stage 2) to SAITS's own continuous point predictions, on the SAME
oracle population (ground-truth-wet masked test positions), across the
SAME 8 test scenarios and 3 seeds. This tests whether RQ3's central
finding -- coverage degradation under increasingly outcome-dependent
MNAR stress -- is specific to DLPIF's Random-Forest Stage 2, or also
appears in an architecturally unrelated backbone (a transformer-based
multivariate imputer with no occurrence/amount decomposition).

Scope note: SAITS has no separate occurrence classifier, so only the
ORACLE population (ground-truth-wet positions, regardless of any
wet/dry decision) is evaluated here -- there is no DLPIF-style
END-TO-END population to construct without inventing a synthetic
gating rule for SAITS, which this script deliberately does not do.

Data: SAITS's own point predictions, from
  saits_test_seed{seed}_{scenario}.npy       (test, 5 canonical scenarios,
                                               produced by
                                               baselines/repackage_saits_outputs.py)
  saits_test_seed{seed}_{mar/mnar scenario}.npy (test, 4 MAR/MNAR scenarios,
                                               produced by extending that
                                               same script's SCENARIOS dict
                                               with the mar_meteo/mnar_*
                                               corrupted_/art_mask_ keys
                                               already present in
                                               preprocessed_test.npz)
  saits_val_seed{seed}_10pct.npy             (VAL-CAL calibration,
                                               produced by
                                               baselines/repackage_saits_val_10pct.py,
                                               same corrupted_10pct scenario
                                               train_saits_v2.py already uses
                                               for its own validation loss)
All three families are inference-only reuses of the already-trained,
already-selected SAITS checkpoints -- no retraining.

Output: results/rq3_conformal_saits/
  conformal_saits_calibration_summary.csv
  test_conformal_saits_oracle.csv
"""
import os
import numpy as np
import pandas as pd

import direct_two_stage_rf as d2s

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq3_conformal_saits')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
ALPHA = 0.10
WET_THRESH = d2s.WET_THRESH
P95_THRESH = d2s.P95_THRESH
TEST_SCENARIOS = ['10pct', 'mar_meteo', 'mnar_wet',
                   'mnar_intensity_moderate', 'mnar_intensity_severe',
                   'block7d', 'block30d', 'netblock30d']
# scenario label -> corrupted_/art_mask_ key stem (matches d2s.SCENARIOS)
SCEN_KEYS = {label: key for label, key, _ in d2s.SCENARIOS if label in TEST_SCENARIOS}


def load_saits_precip(sc, pidx, npz, fname):
    path = os.path.join(SRC_DIR, fname)
    arr_norm = np.load(path).astype(np.float64)
    arr_orig = sc.inverse_transform(np.clip(arr_norm, 0, 1))
    return np.clip(arr_orig[:, pidx], 0, None)


def conformal_quantile(residuals, alpha=ALPHA):
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
        n=len(y), picp=picp(y, lo, hi), mpiw=float(np.mean(hi - lo)),
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
    print('  RQ3 BACKBONE-GENERALISATION CHECK -- SAITS symmetric split-conformal')
    print('  (same construction as stage2_conformal_uq.py, oracle population only)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    va = d2s.load_npz('val')
    te = d2s.load_npz('test')
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]

    manifest = pd.read_csv(os.path.join(RQ2_DIR, 'validation_split_manifest.csv'),
                           parse_dates=['date'])
    cal_dates = set(manifest.loc[manifest['split'] == 'VAL-CAL', 'date'])
    va_dates = pd.to_datetime(va['dates'])
    is_cal_row = pd.Series(va_dates).isin(cal_dates).to_numpy()
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)

    calib_rows, oracle_rows = [], []

    for seed in SEEDS:
        print(f'\n{"="*70}\n  SEED = {seed}\n{"="*70}')

        saits_va = load_saits_precip(sc, pidx, va, f'saits_val_seed{seed}_10pct.npy')
        mask_cal_wet = va_obs & is_cal_row & (va_gt > WET_THRESH)
        yhat_cal = saits_va[mask_cal_wet]
        residuals = np.abs(va_gt[mask_cal_wet] - yhat_cal)
        q = conformal_quantile(residuals)
        print(f'  VAL-CAL wet n={mask_cal_wet.sum()}  conformal q={q:.4f} mm')
        calib_rows.append(dict(seed=seed, n_cal_wet=int(mask_cal_wet.sum()), q=q,
                               nominal_coverage=1 - ALPHA))

        for scen_label in TEST_SCENARIOS:
            _, mask_key = SCEN_KEYS[scen_label], f'art_mask_{scen_label}'
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            saits_te = load_saits_precip(sc, pidx, te, f'saits_test_seed{seed}_{scen_label}.npy')

            oracle_mask = m & (te_gt > WET_THRESH)
            y_o = te_gt[oracle_mask]
            yhat_o = saits_te[oracle_mask]
            lo_o = np.maximum(0.0, yhat_o - q)
            hi_o = yhat_o + q
            extreme_o = y_o >= P95_THRESH
            row_o = population_metrics(y_o, yhat_o, lo_o, hi_o, extreme_o)
            row_o.update(seed=seed, scenario=scen_label, population='oracle', method='SAITS')
            oracle_rows.append(row_o)
            print(f'    [{scen_label:24s}] n={row_o["n"]:4d} PICP={row_o["picp"]:.3f} '
                  f'PICP_p95={row_o["picp_p95"]:.3f} IS={row_o["interval_score"]:.2f}')

    calib_df = pd.DataFrame(calib_rows)
    oracle_df = pd.DataFrame(oracle_rows)
    calib_df.to_csv(os.path.join(OUT_DIR, 'conformal_saits_calibration_summary.csv'), index=False)
    oracle_df.to_csv(os.path.join(OUT_DIR, 'test_conformal_saits_oracle.csv'), index=False)
    print(f'\n  -> {OUT_DIR}\\conformal_saits_calibration_summary.csv')
    print(f'  -> {OUT_DIR}\\test_conformal_saits_oracle.csv')

    print('\n  SAITS ORACLE (mean over 3 seeds):')
    agg = oracle_df.groupby('scenario')[['picp', 'picp_p95', 'mpiw', 'interval_score', 'rmse']].mean()
    agg = agg.reindex(TEST_SCENARIOS)
    print(agg.round(4))
    print('=' * 70)


if __name__ == '__main__':
    main()

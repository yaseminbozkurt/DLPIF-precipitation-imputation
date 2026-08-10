"""
missingness_indicator_rf.py
=============================
Missingness-indicator ablation for local (station-own) meteorological
variables: identical architecture, feature blocks, and hyperparameters to
direct_two_stage_rf.py (backbone-free Stage-1 occurrence classifier +
Stage-2 amount regressor, trained directly on raw corrupted records), with
one addition -- six extra binary features per stage marking whether each
LOCAL non-PRECIP variable (TMIN, TMEAN, TMAX, RH_MEAN, P_MEAN, WIND_MEAN)
was actually observed at that timestep, as opposed to zero-filled because
of a natural gap or an artificial mask.

Why this indicator is not already implicit
--------------------------------------------
build_occ_X() / build_amt_X() (direct_two_stage_rf.py) feed the RF the
*value* of the zero-filled corrupted array. A true zero (e.g. WIND_MEAN
genuinely calm) and a missing-and-zero-filled cell are numerically
indistinguishable to the model without an explicit flag -- exactly the
distinction the classical missing-data literature's "response indicator"
/ "M-matrix" features (Rubin's missingness mechanism framework; also used
routinely in MICE-adjacent tabular pipelines) are designed to restore.
Local PRECIP's own indicator is deliberately excluded: at every
artificially masked evaluation position that indicator is 0 by
construction (that is what "masked" means), so it would carry no
information at exactly the positions being scored and would only add
train/test distributional mismatch.

Indicator definition (per scenario, per local non-PRECIP column)
-------------------------------------------------------------------
combined_observed = real_mask & ~art_mask_<scenario>
i.e. TRUE only where the cell is both a genuine (non-missing) observation
AND not one of this scenario's artificially masked evaluation cells --
exactly the condition under which the corrupted array's zero-fill at that
cell reflects "missing" rather than a genuine value. This is recomputed
per scenario (train/val always use the 10% training partition's own
art_mask; each test scenario uses its own), never leaking the artificial
mask of one scenario into another.

Comparison baseline
--------------------
DirectTwoStageRF (direct_two_stage_rf.py) is the fairest comparison: same
data snapshot, same RF hyperparameters, same training scenario, same
metrics, features built the same way from the same raw corrupted arrays --
the ONLY difference is the addition of the six local missingness-indicator
features. Any performance gap therefore isolates the marginal value of
explicit local-variable missingness indicators, holding architecture and
every other design choice fixed.

Outputs
-------
  results/missingness_indicator_rf_evaluation.csv  -- per-seed x scenario metrics
  src/missingness_indicator_rf_test_seed{s}_{scenario}.npy
                                                      -- reconstructed
                                                         normalised (N,7)
                                                         array, PRECIP
                                                         column only
                                                         meaningful
"""
import sys, io, os, warnings
try:
    if getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import direct_two_stage_rf as d2s

SRC_DIR     = d2s.SRC_DIR
RESULTS_DIR = d2s.RESULTS_DIR
WET_THRESH  = d2s.WET_THRESH
SEEDS       = d2s.SEEDS
TRAIN_SCENARIO = d2s.TRAIN_SCENARIO
TRAIN_COR_KEY  = d2s.TRAIN_COR_KEY
TRAIN_MASK_KEY = 'art_mask_10pct'
SCENARIOS   = d2s.SCENARIOS


def local_indicator_cols(pidx, n_vars):
    """The 6 local non-PRECIP variable column indices."""
    return [i for i in range(n_vars) if i != pidx]


def combined_observed(z, art_mask_key, cols):
    """(N, len(cols)) float32 array: 1.0 where the local cell is a genuine
    observation AND not this scenario's artificial mask, 0.0 otherwise --
    matches exactly what a 0.0 in the corrupted array actually means at
    that cell for this scenario.
    """
    real_mask = z['real_mask'].astype(bool)
    art_mask = z[art_mask_key].astype(bool)
    obs = real_mask & ~art_mask
    return obs[:, cols].astype(np.float32)


def build_occ_X_indicator(z, cor_key, art_mask_key, pidx, na_key, nm_key):
    cor = z[cor_key].astype(np.float32)
    n_vars = cor.shape[1]
    cols = local_indicator_cols(pidx, n_vars)
    ind = combined_observed(z, art_mask_key, cols)
    parts = [
        np.nan_to_num(cor[:, cols], nan=0.0).astype(np.float32),
        np.nan_to_num(z['temporal'].astype(np.float32), nan=0.0),
        np.nan_to_num(z[na_key].astype(np.float32), nan=0.0),
        np.nan_to_num(z[nm_key].astype(np.float32), nan=0.0),
        ind,
    ]
    X = np.concatenate(parts, axis=1)
    assert X.shape[1] == 25 + len(cols), X.shape
    return X


def build_amt_X_indicator(sc, z, cor_key, art_mask_key, pidx, na_key, nm_key):
    cor = d2s.inv(sc, z[cor_key].astype(np.float32)).astype(np.float32)
    cor[:, pidx] = 0.0
    n_vars = cor.shape[1]
    cols = local_indicator_cols(pidx, n_vars)
    ind = combined_observed(z, art_mask_key, cols)
    X = np.concatenate([
        cor,
        z['temporal'].astype(np.float32),
        d2s.inv(sc, z[na_key].astype(np.float32)).astype(np.float32),
        z[nm_key].astype(np.float32),
        ind,
    ], axis=1)
    assert X.shape[1] == 26 + len(cols), X.shape
    return X


def main():
    print('=' * 70)
    print('  MISSINGNESS-INDICATOR RF -- LOCAL-VARIABLE ABLATION')
    print('  (DirectTwoStageRF architecture + 6 local non-PRECIP')
    print('   real-observed-vs-zero-filled indicator features)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')

    tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')
    tr_na_key, tr_nm_key = d2s.neighbor_keys(TRAIN_SCENARIO)
    d2s.require_keys(tr, [TRAIN_COR_KEY, TRAIN_MASK_KEY, tr_na_key, tr_nm_key], 'train')
    d2s.require_keys(va, [TRAIN_COR_KEY, TRAIN_MASK_KEY, tr_na_key, tr_nm_key], 'val')

    X_tr_full = build_occ_X_indicator(tr, TRAIN_COR_KEY, TRAIN_MASK_KEY, pidx, tr_na_key, tr_nm_key)
    X_va_full = build_occ_X_indicator(va, TRAIN_COR_KEY, TRAIN_MASK_KEY, pidx, tr_na_key, tr_nm_key)
    X_tr_amt  = build_amt_X_indicator(sc, tr, TRAIN_COR_KEY, TRAIN_MASK_KEY, pidx, tr_na_key, tr_nm_key)

    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]

    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs

    X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs] > WET_THRESH).astype(int)
    X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs] > WET_THRESH).astype(int)

    scen_features = {}
    for scen_label, cor_key, mask_key in SCENARIOS:
        na_key, nm_key = d2s.neighbor_keys(scen_label)
        d2s.require_keys(te, [cor_key, mask_key, na_key, nm_key], f'test/{scen_label}')
        scen_features[scen_label] = dict(
            X_occ=build_occ_X_indicator(te, cor_key, mask_key, pidx, na_key, nm_key),
            X_amt=build_amt_X_indicator(sc, te, cor_key, mask_key, pidx, na_key, nm_key),
            cor_norm=te[cor_key].astype(np.float32),
        )

    print(f'\n  Occ feature dim={X_tr_occ.shape[1]} (25 base + 6 local indicators)')
    print(f'  Amt feature dim={X_tr_amt.shape[1]} (26 base + 6 local indicators)')
    print(f'  Train obs={len(X_tr_occ):,}  Val obs={len(X_va_occ):,}')

    all_records = []

    for seed in SEEDS:
        print(f'\n{"="*60}')
        print(f'  SEED = {seed}')
        print('=' * 60)

        occ_rf, occ_cut, vm = d2s.train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed)
        print(f'  Occ RF  cutoff={occ_cut}  val_F1={vm["val_f1"]}  '
              f'P={vm["val_prec"]}  R={vm["val_rec"]}  bias={vm["val_bias"]:+.4f}')

        amt_rf = RandomForestRegressor(n_estimators=400, random_state=seed,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

        for scen_label, cor_key, mask_key in SCENARIOS:
            sf = scen_features[scen_label]
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            gt_m = te_gt[m]; n_m = int(m.sum())

            te_proba = occ_rf.predict_proba(sf['X_occ'])[:, 1]
            te_wet_pred = (te_proba >= occ_cut).astype(bool)

            pred_scen = np.zeros_like(te_gt)
            apply_sel = m & te_wet_pred
            if apply_sel.sum() > 0:
                pred_scen[apply_sel] = np.maximum(amt_rf.predict(sf['X_amt'][apply_sel]), 0.0)

            recon_norm = np.nan_to_num(sf['cor_norm'].copy(), nan=0.0)
            recon_norm[:, pidx] = d2s.to_norm(sc, pred_scen, pidx)
            recon_norm[m, pidx] = d2s.to_norm(sc, pred_scen[m], pidx)
            npy_path = os.path.join(SRC_DIR, f'missingness_indicator_rf_test_seed{seed}_{scen_label}.npy')
            np.save(npy_path, recon_norm.astype(np.float32))

            r = d2s.metrics(gt_m, pred_scen[m], f'MissingnessIndicatorRF_seed{seed}', scen_label, n_m)
            all_records.append(r)
            print(f'  [{scen_label} | {cor_key}] wet_pred={te_wet_pred[m].mean():.4f} '
                  f'F1={r["f1"]:.4f} bias={r["bias"]:+.4f} '
                  f'RMSE_wet={r["rmse_wet"]} RMSE_p95={r["rmse_p95"]}')

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(RESULTS_DIR, 'missingness_indicator_rf_evaluation.csv'), index=False)

    SCENS = ['10pct', '20pct', 'block7d', 'block30d']
    print('\n' + '=' * 70)
    print('  MISSINGNESS-INDICATOR-RF -- F1 / Bias / RMSE_wet / RMSE_p95 (mean +/- std)')
    print('=' * 70)
    g = df.groupby('scenario')
    for scen in SCENS:
        if scen not in g.groups: continue
        sub = g.get_group(scen)
        print(f'  {scen:<10} F1={sub.f1.mean():.4f}+/-{sub.f1.std():.4f}  '
              f'Bias={sub.bias.mean():+.4f}+/-{sub.bias.std():.4f}  '
              f'RMSE_wet={sub.rmse_wet.mean():.2f}+/-{sub.rmse_wet.std():.2f}  '
              f'RMSE_p95={sub.rmse_p95.mean():.2f}+/-{sub.rmse_p95.std():.2f}')

    print('\n  Outputs:')
    print('  -> results/missingness_indicator_rf_evaluation.csv')
    print('  -> src/missingness_indicator_rf_test_seed{seed}_{scenario}.npy')
    print('=' * 70)


if __name__ == '__main__':
    main()

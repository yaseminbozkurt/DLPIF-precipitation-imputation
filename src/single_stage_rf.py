"""
single_stage_rf.py
====================
Single-stage Random Forest regression baseline: the same 26-feature
amount matrix used by DLPIF's Stage 2 (build_amt_X -- raw corrupted
meteorological block with local PRECIP hard-zeroed, temporal encodings,
neighbouring-station average and mask), but with NO Stage 1 occurrence
classifier. One RandomForestRegressor is trained directly on ALL training
days (both observed-dry and observed-wet, not just the wet-day subset)
and applied to every masked test position, regardless of predicted wet/dry
state. Its own regression output is the only decision boundary between
"dry" and "wet" reconstruction.

Purpose
-------
Isolates the contribution of DLPIF's classify-then-regress decomposition
from the contribution of the feature set itself. DirectTwoStageRF and
AmountRF_DLPIF already show that raw-record features (no continuous
backbone) are sufficient for strong occurrence + amount reconstruction;
this script asks the complementary question -- given the IDENTICAL
feature matrix and the same RF family/hyperparameters as Stage 2, does
explicitly separating occurrence classification from amount regression
still help, or could a single regressor learn the zero-inflated
wet/dry-plus-amount target directly? Unlike Table 3's "Occurrence-only"
ablation (which removes Stage 2 and keeps Stage 1), this baseline removes
Stage 1 and keeps a single Stage-2-shaped regressor -- the complementary
half of the same ablation.

Held identical to direct_two_stage_rf.py on purpose (same data snapshot,
same feature construction, same RF regressor hyperparameters, same
training scenario, same metrics), so that any performance gap isolates
the single-stage-vs-two-stage design choice rather than a difference in
data, features, or RF configuration:
  - 26-feature amount matrix (build_amt_X), local PRECIP hard-zeroed.
  - RandomForestRegressor, 400 trees, min_leaf=2, identical to DLPIF's
    Stage 2 amount regressor -- but trained on ALL observed training-set
    days (tr_obs), not just the observed-wet subset (tr_wet), since there
    is no separate occurrence classifier to pre-select wet days here.
  - Trained once on the 10% random-missingness partition, applied
    unchanged to all four test scenarios.
  - Predictions clipped at 0 (physical non-negativity), matching every
    other method in the canonical pipeline.

Outputs
-------
  results/single_stage_rf_evaluation.csv   -- per-seed x scenario metrics
                                                (direct_two_stage_rf.py schema)
  src/single_stage_rf_test_seed{s}_{scenario}.npy
                                                -- reconstructed normalised
                                                   (N,7) array, PRECIP
                                                   column only meaningful
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
SCENARIOS   = d2s.SCENARIOS


def main():
    print('=' * 70)
    print('  SINGLE-STAGE RF -- DIRECT REGRESSION BASELINE (SAME FEATURES AS STAGE 2)')
    print('  (one RandomForestRegressor per seed, no occurrence classifier;')
    print('   trained on ALL observed training days using the identical')
    print('   26-feature amount matrix build_amt_X())')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')

    tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')
    tr_na_key, tr_nm_key = d2s.neighbor_keys(TRAIN_SCENARIO)
    d2s.require_keys(tr, [TRAIN_COR_KEY, tr_na_key, tr_nm_key], 'train')

    X_tr_amt = d2s.build_amt_X(sc, tr, TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)

    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)

    scen_features = {}
    for scen_label, cor_key, mask_key in SCENARIOS:
        na_key, nm_key = d2s.neighbor_keys(scen_label)
        d2s.require_keys(te, [cor_key, mask_key, na_key, nm_key], f'test/{scen_label}')
        scen_features[scen_label] = dict(
            X_amt=d2s.build_amt_X(sc, te, cor_key, pidx, na_key, nm_key),
            cor_norm=te[cor_key].astype(np.float32),
        )

    n_tr_obs = int(tr_obs.sum())
    n_tr_wet = int(((tr_gt > WET_THRESH) & tr_obs).sum())
    print(f'\n  Feature dim={X_tr_amt.shape[1]}  Train obs(all)={n_tr_obs:,}  '
          f'(of which wet={n_tr_wet:,}, {n_tr_wet/n_tr_obs:.1%})')

    all_records = []

    for seed in SEEDS:
        print(f'\n{"="*60}')
        print(f'  SEED = {seed}')
        print('=' * 60)

        reg = RandomForestRegressor(n_estimators=400, random_state=seed,
                                    min_samples_leaf=2, n_jobs=-1)
        reg.fit(X_tr_amt[tr_obs], tr_gt[tr_obs])

        for scen_label, cor_key, mask_key in SCENARIOS:
            sf = scen_features[scen_label]
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            gt_m = te_gt[m]; n_m = int(m.sum())

            pred_scen = np.zeros_like(te_gt)
            pred_scen[m] = np.maximum(reg.predict(sf['X_amt'][m]), 0.0)

            recon_norm = np.nan_to_num(sf['cor_norm'].copy(), nan=0.0)
            recon_norm[:, pidx] = d2s.to_norm(sc, pred_scen, pidx)
            recon_norm[m, pidx] = d2s.to_norm(sc, pred_scen[m], pidx)
            npy_path = os.path.join(SRC_DIR, f'single_stage_rf_test_seed{seed}_{scen_label}.npy')
            np.save(npy_path, recon_norm.astype(np.float32))

            r = d2s.metrics(gt_m, pred_scen[m], f'SingleStageRF_seed{seed}', scen_label, n_m)
            all_records.append(r)
            wet_frac_pred = float((pred_scen[m] > WET_THRESH).mean())
            print(f'  [{scen_label} | {cor_key}] wet_pred={wet_frac_pred:.4f} '
                  f'F1={r["f1"]:.4f} bias={r["bias"]:+.4f} '
                  f'RMSE_wet={r["rmse_wet"]} RMSE_p95={r["rmse_p95"]}')

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(RESULTS_DIR, 'single_stage_rf_evaluation.csv'), index=False)

    SCENS = ['10pct', '20pct', 'block7d', 'block30d']
    print('\n' + '=' * 70)
    print('  SINGLE-STAGE-RF -- F1 / Bias / RMSE_wet / RMSE_p95 (mean +/- std)')
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
    print('  -> results/single_stage_rf_evaluation.csv')
    print('  -> src/single_stage_rf_test_seed{seed}_{scenario}.npy')
    print('=' * 70)


if __name__ == '__main__':
    main()

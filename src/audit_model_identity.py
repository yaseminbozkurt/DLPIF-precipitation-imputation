"""
audit_model_identity.py
=========================
Independent, structural (not prediction-based) audit that the pretrained
Stage-1 occurrence RandomForestClassifier used by
calibrate_occurrence_probability.py (RQ2, frozen at git tag `rq2-freeze`)
is genuinely never refit or mutated by the calibration step.

Why not just diff predict_proba() before/after (as
calibrate_occurrence_probability.py's own inline assertion does)?
RandomForestClassifier(n_jobs=-1)'s parallel tree-averaging order is not
guaranteed identical across repeated predict_proba() calls on IDENTICAL
input -- verified separately, ~1e-16-scale float jitter with zero
mutation in between -- so a prediction-array diff is a fragile invariant.
This script instead hashes the model's STRUCTURE, which is untouched by
prediction and can only change if the estimator is refit:
  - get_params(deep=True)            hyperparameters
  - n_features_in_, classes_
  - per-tree children_left, children_right, feature, threshold, value
    (every array that fully determines what the forest computes)

This does NOT re-run or overwrite any RQ2 result file (test_calibration_
metrics.csv, thresholds.csv, etc. under results/rq2_calibration/ are
untouched, and rq2-freeze is not moved). It re-executes only the
calibration-FITTING call path (fit_platt / fit_isotonic, imported directly
from calibrate_occurrence_probability.py so there is no drift between what
this audits and what RQ2 actually ran) against the exact frozen VAL-CAL
partition recorded in validation_split_manifest.csv, hashes the RF before
and after, and -- as a bonus determinism check -- compares the freshly
re-fit Platt/Isotonic mappings' parameters against the ones
calibrate_occurrence_probability.py persisted to calibration_models/.

Output:
  results/rq2_calibration/model_identity_audit.csv
"""
import os
import pickle
import hashlib
import numpy as np
import pandas as pd

import direct_two_stage_rf as d2s
from calibrate_occurrence_probability import fit_platt, fit_isotonic

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
MODEL_DIR = os.path.join(RQ2_DIR, 'calibration_models')
SEEDS = [42, 123, 456]


def hash_rf_structure(rf):
    """SHA256 over hyperparameters + every tree's decision structure.
    Deliberately excludes predict_proba() output -- see module docstring.
    """
    h = hashlib.sha256()

    params = rf.get_params(deep=True)
    h.update(repr(sorted(params.items(), key=lambda kv: kv[0])).encode('utf-8'))
    h.update(repr(int(rf.n_features_in_)).encode('utf-8'))
    h.update(np.asarray(rf.classes_).tobytes())

    for est in rf.estimators_:
        t = est.tree_
        h.update(t.children_left.tobytes())
        h.update(t.children_right.tobytes())
        h.update(t.feature.tobytes())
        h.update(t.threshold.tobytes())
        h.update(np.ascontiguousarray(t.value).tobytes())

    return h.hexdigest()


def estimator_params_close(est_a, est_b, atol=1e-6):
    """Approximate (not byte-exact) comparison of the small post-hoc
    calibration mappings' fitted parameters. Byte-exact hashing was tried
    first and failed non-deterministically even across two back-to-back
    fits on the same VAL-CAL rows -- traced to the same ~1e-16-scale
    predict_proba() float jitter documented in
    calibrate_occurrence_probability.py, which perturbs raw_p_cal itself
    (not just the RF structure hash_rf_structure() above deliberately
    avoids depending on). Isotonic regression's PAVA breakpoints and
    LogisticRegression's lbfgs solution are both sensitive to such
    input-level float noise even though the RF that produced it is
    provably unchanged (see hash_rf_structure() results, which ARE
    byte-exact). allclose is the correct tool here, not a workaround.
    """
    if hasattr(est_a, 'coef_'):  # LogisticRegression (Platt)
        return (np.allclose(est_a.coef_, est_b.coef_, atol=atol)
                and np.allclose(est_a.intercept_, est_b.intercept_, atol=atol))
    if hasattr(est_a, 'X_thresholds_'):  # IsotonicRegression
        return (est_a.X_thresholds_.shape == est_b.X_thresholds_.shape
                and np.allclose(est_a.y_thresholds_, est_b.y_thresholds_, atol=atol))
    return False


def main():
    print('=' * 70)
    print('  MODEL IDENTITY AUDIT -- structural RF hash before/after calibration')
    print('  (RQ2 results NOT re-run or overwritten; rq2-freeze untouched)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    va = d2s.load_npz('val')

    manifest_path = os.path.join(RQ2_DIR, 'validation_split_manifest.csv')
    manifest = pd.read_csv(manifest_path, parse_dates=['date'])
    cal_dates = set(manifest.loc[manifest['split'] == 'VAL-CAL', 'date'])

    va_dates = pd.to_datetime(va['dates'])
    is_cal_row = pd.Series(va_dates).isin(cal_dates).to_numpy()

    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)
    X_va_full = d2s.build_occ_X(va[d2s.TRAIN_COR_KEY].astype(np.float32),
                                va['temporal'].astype(np.float32),
                                va[tr_na_key].astype(np.float32),
                                va[tr_nm_key].astype(np.float32), pidx)
    va_gt_mm = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    y_va_full = (va_gt_mm > d2s.WET_THRESH).astype(int)
    mask_cal = va_obs & is_cal_row

    rows = []
    for seed in SEEDS:
        pkl_path = os.path.join(SRC_DIR, f'direct_two_stage_occurrence_seed{seed}.pkl')
        with open(pkl_path, 'rb') as f:
            occ_rf = pickle.load(f)['rf']

        n_trees = len(occ_rf.estimators_)
        hash_before = hash_rf_structure(occ_rf)

        raw_p_cal = occ_rf.predict_proba(X_va_full[mask_cal])[:, 1]
        y_cal = y_va_full[mask_cal]
        platt = fit_platt(raw_p_cal, y_cal)
        iso = fit_isotonic(raw_p_cal, y_cal)

        hash_after = hash_rf_structure(occ_rf)
        identical = (hash_before == hash_after)
        status = 'IDENTICAL' if identical else 'MISMATCH -- RF STRUCTURE CHANGED'
        print(f'  seed={seed}  n_trees={n_trees}  hash_before[:12]={hash_before[:12]}  '
              f'hash_after[:12]={hash_after[:12]}  {status}')
        assert identical, f'seed={seed}: RF structural hash changed after calibration fitting'

        # Bonus determinism check: does re-fitting reproduce the persisted mappings?
        platt_reproduced = iso_reproduced = None
        platt_path = os.path.join(MODEL_DIR, f'platt_seed{seed}.pkl')
        iso_path = os.path.join(MODEL_DIR, f'isotonic_seed{seed}.pkl')
        if os.path.exists(platt_path):
            with open(platt_path, 'rb') as f:
                saved_platt = pickle.load(f)
            platt_reproduced = estimator_params_close(platt, saved_platt)
        if os.path.exists(iso_path):
            with open(iso_path, 'rb') as f:
                saved_iso = pickle.load(f)
            iso_reproduced = estimator_params_close(iso, saved_iso)
        print(f'           platt_reproduced={platt_reproduced}  iso_reproduced={iso_reproduced}')

        rows.append(dict(seed=seed, n_trees=n_trees, n_features_in=int(occ_rf.n_features_in_),
                         hash_before=hash_before, hash_after=hash_after,
                         structurally_identical=identical,
                         n_val_cal=int(mask_cal.sum()),
                         platt_reproduced=platt_reproduced, iso_reproduced=iso_reproduced))

    df = pd.DataFrame(rows)
    out_path = os.path.join(RQ2_DIR, 'model_identity_audit.csv')
    df.to_csv(out_path, index=False)
    print(f'\n  -> {out_path}')
    assert df['structurally_identical'].all(), 'audit FAILED for at least one seed'
    print('  [OK] All 3 seeds: pretrained occurrence RF structurally unchanged by calibration.')
    print('=' * 70)


if __name__ == '__main__':
    main()

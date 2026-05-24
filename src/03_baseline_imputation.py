"""
03_baseline_imputation.py  — MVP (4 methods)
=============================================
Baseline imputation methods for comparison with WGAN-GP.

Methods:
  1. Mean Imputation
  2. Linear Interpolation
  3. KNN Imputation (k=5)
  4. MICE (IterativeImputer + RandomForest)

All fitted on train data only. Evaluated on test set at 10% and 20%
artificial missingness, plus block scenarios (7-day, 30-day) — MVP.

Outputs:
  baseline_results.pkl
  baseline_rmse.csv
"""
import sys, io, os
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
import pickle
import warnings; warnings.filterwarnings('ignore')

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_npz(name):
    return np.load(os.path.join(OUTPUT_DIR, f'preprocessed_{name}.npz'), allow_pickle=True)


def get_meteo_vars(npz):
    return list(npz['meteo_vars'])


def get_scenario_keys(npz):
    """Return list of (key_corrupted, key_art_mask, scenario_label) for all scenarios."""
    keys = list(npz.files)
    scenarios = []
    for k in keys:
        if k.startswith('corrupted_'):
            suffix = k[len('corrupted_'):]
            mask_k = f'art_mask_{suffix}'
            if mask_k in keys:
                scenarios.append((k, mask_k, suffix))
    return scenarios


def rmse(pred, gt, ev_mask):
    sel = ev_mask.astype(bool)
    if sel.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred[sel] - gt[sel]) ** 2)))


def mae(pred, gt, ev_mask):
    sel = ev_mask.astype(bool)
    if sel.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred[sel] - gt[sel])))


def per_var_metrics(pred, gt, art_mask, meteo_vars):
    out = {}
    for i, v in enumerate(meteo_vars):
        out[v] = {
            'RMSE': rmse(pred[:, i], gt[:, i], art_mask[:, i]),
            'MAE' : mae(pred[:, i],  gt[:, i], art_mask[:, i]),
        }
    mean_rmse = np.nanmean([out[v]['RMSE'] for v in meteo_vars])
    mean_mae  = np.nanmean([out[v]['MAE']  for v in meteo_vars])
    out['MEAN'] = {'RMSE': mean_rmse, 'MAE': mean_mae}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mean Imputation
# ─────────────────────────────────────────────────────────────────────────────
def mean_imputation(train_data, test_corrupted):
    print("  [1] Mean Imputation ... ", end='')
    col_means = np.nanmean(train_data, axis=0)
    out = test_corrupted.copy()
    for c in range(out.shape[1]):
        nan_idx = np.isnan(out[:, c])
        out[nan_idx, c] = col_means[c]
    out = np.clip(out, 0, 1)
    print("done")
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Linear Interpolation
# ─────────────────────────────────────────────────────────────────────────────
def linear_interpolation(train_data, test_corrupted, meteo_vars):
    print("  [2] Linear Interpolation ... ", end='')
    df  = pd.DataFrame(test_corrupted, columns=meteo_vars)
    out = df.interpolate(method='linear', limit_direction='both').values
    out = np.clip(out, 0, 1)
    print("done")
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. KNN Imputation
# ─────────────────────────────────────────────────────────────────────────────
def knn_imputation(train_data, test_corrupted):
    print("  [3] KNN Imputation (k=5) ... ", end='')
    # Fit KNN on train (fill train NaN with median first)
    train_clean = train_data.copy()
    medians     = np.nanmedian(train_clean, axis=0)
    for c in range(train_clean.shape[1]):
        nan_idx = np.isnan(train_clean[:, c])
        train_clean[nan_idx, c] = medians[c]
    knn = KNNImputer(n_neighbors=5)
    knn.fit(train_clean)
    out = knn.transform(test_corrupted)
    out = np.clip(out, 0, 1)
    print("done")
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MICE (IterativeImputer with RF)
# ─────────────────────────────────────────────────────────────────────────────
def mice_imputation(train_data, test_corrupted):
    print("  [4] MICE (IterativeImputer + RF) ... ", end='')
    mice = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1),
        max_iter=10, random_state=RANDOM_SEED
    )
    # Fill train NaN with median for fitting
    train_clean = train_data.copy()
    medians     = np.nanmedian(train_clean, axis=0)
    for c in range(train_clean.shape[1]):
        nan_idx = np.isnan(train_clean[:, c])
        train_clean[nan_idx, c] = medians[c]
    mice.fit(train_clean)
    out = mice.transform(test_corrupted)
    out = np.clip(out, 0, 1)
    print("done")
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BASELINE IMPUTATION — MVP (4 methods)")
    print("=" * 60)

    tr_npz   = load_npz('train')
    te_npz   = load_npz('test')
    meteo_vars = get_meteo_vars(tr_npz)

    train_data = tr_npz['data'].astype(np.float64)
    test_data  = te_npz['data'].astype(np.float64)

    print(f"\n  Train shape: {train_data.shape}  Test shape: {test_data.shape}")
    print(f"  Meteo vars: {meteo_vars}\n")

    # Get all test scenarios
    scenarios = get_scenario_keys(te_npz)
    print(f"  Test scenarios: {[s[2] for s in scenarios]}\n")

    # ── Run methods on the 10% scenario (primary)
    miss_key   = 'corrupted_10pct'
    amask_key  = 'art_mask_10pct'
    test_cor   = te_npz[miss_key].astype(np.float64)
    test_art   = te_npz[amask_key].astype(np.float32)

    # Inform user about all available scenarios in this npz
    print(f"\n  [Note] MVP scenarios in test npz: {[s[2] for s in scenarios]}")

    results_10pct = {}
    print("  ─── Primary evaluation (10% random missingness) ───")
    imp = mean_imputation(train_data, test_cor)
    results_10pct['mean'] = imp
    rv = per_var_metrics(imp, test_data, test_art, meteo_vars)
    print(f"       Mean RMSE={rv['MEAN']['RMSE']:.4f}")

    imp = linear_interpolation(train_data, test_cor, meteo_vars)
    results_10pct['linear'] = imp
    rv = per_var_metrics(imp, test_data, test_art, meteo_vars)
    print(f"       Linear RMSE={rv['MEAN']['RMSE']:.4f}")

    imp = knn_imputation(train_data, test_cor)
    results_10pct['knn'] = imp
    rv = per_var_metrics(imp, test_data, test_art, meteo_vars)
    print(f"       KNN RMSE={rv['MEAN']['RMSE']:.4f}")

    imp = mice_imputation(train_data, test_cor)
    results_10pct['mice'] = imp
    rv = per_var_metrics(imp, test_data, test_art, meteo_vars)
    print(f"       MICE RMSE={rv['MEAN']['RMSE']:.4f}")

    # ── Evaluate all methods over all scenarios
    methods = {
        'mean'  : lambda c: mean_imputation(train_data, c),
        'linear': lambda c: linear_interpolation(train_data, c, meteo_vars),
        'knn'   : lambda c: knn_imputation(train_data, c),
        'mice'  : lambda c: mice_imputation(train_data, c),
    }

    # Store all imputed arrays keyed by (method, scenario)
    all_results = {}
    records     = []

    print("\n  ─── All-scenario evaluation ───")
    for (ck, ak, label) in scenarios:
        test_cor_s = te_npz[ck].astype(np.float64)
        test_art_s = te_npz[ak].astype(np.float32)
        for mname, mfunc in methods.items():
            imp_s = mfunc(test_cor_s)
            all_results[(mname, label)] = imp_s
            rv = per_var_metrics(imp_s, test_data, test_art_s, meteo_vars)
            row = {'method': mname, 'scenario': label}
            for v in meteo_vars:
                row[v + '_RMSE'] = rv[v]['RMSE']
                row[v + '_MAE']  = rv[v]['MAE']
            row['MEAN_RMSE'] = rv['MEAN']['RMSE']
            row['MEAN_MAE']  = rv['MEAN']['MAE']
            records.append(row)
            print(f"    {mname:8s} | {label:12s} | RMSE={rv['MEAN']['RMSE']:.4f}  MAE={rv['MEAN']['MAE']:.4f}")

    # Save
    pkl_path = os.path.join(OUTPUT_DIR, 'baseline_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'primary_10pct': results_10pct, 'all_scenarios': all_results}, f)
    print(f"\n  Results saved → {pkl_path}")

    csv_path = os.path.join(OUTPUT_DIR, 'baseline_rmse.csv')
    df_rmse  = pd.DataFrame(records).sort_values(['scenario', 'MEAN_RMSE'])
    df_rmse.to_csv(csv_path, index=False)
    print(f"  RMSE table saved → {csv_path}")

    print("\n  Primary scenario (10% random) — RMSE summary:")
    prim = df_rmse[df_rmse['scenario'] == '10pct'][['method', 'MEAN_RMSE']].sort_values('MEAN_RMSE')
    print(prim.to_string(index=False))

    print("\n" + "=" * 60)
    print("  BASELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

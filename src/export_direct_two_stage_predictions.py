# -*- coding: utf-8 -*-
"""
export_direct_two_stage_predictions.py
========================================
Row-level export of Direct-Two-Stage-RF (D2S-RF, backbone-free
classify-then-regress baseline; see direct_two_stage_rf.py) predictions at
artificially-masked PRECIP positions, for all 3 seeds x 4 scenarios.

Reads the (N,7) normalised reconstruction arrays already saved by
direct_two_stage_rf.py:
    src/direct_two_stage_rf_test_seed{42,123,456}_{10pct,20pct,block7d,block30d}.npy

Writes a single row-level CSV in the same schema as
results/masked_predictions_all.csv (produced by 05_generate_seed_outputs.py
for the other methods), so D2S-RF can be merged into the same downstream
analyses (08_drizzle_false_wet.py, 10_dry_spell_cdd.py,
11_spatial_correlation.py, 06_audit_predictions.py) without touching those
scripts.

Output:
  results/direct_two_stage_rf_masked_predictions.csv
    columns: scenario, seed, method, date, station_id, y_true, y_pred,
             is_masked, is_true_wet, is_pred_wet
"""
import os, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

WET_THRESH = 0.1
SEEDS      = [42, 123, 456]
SCENARIOS  = {
    '10pct':    'art_mask_10pct',
    '20pct':    'art_mask_20pct',
    'block7d':  'art_mask_block7d',
    'block30d': 'art_mask_block30d',
}
METHOD_LABEL = 'DirectTwoStageRF'


def load_scaler():
    with open(os.path.join(SRC_DIR, 'scaler.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d['scaler'], list(d['meteo_vars'])


def inverse_transform(sc, arr_norm):
    arr = arr_norm.copy().astype(np.float64)
    nan_m = np.isnan(arr)
    arr[nan_m] = 0.0
    out = sc.inverse_transform(arr)
    out[nan_m] = np.nan
    return out


def main():
    print('=' * 65)
    print('  export_direct_two_stage_predictions.py')
    print('  Row-level D2S-RF predictions -> masked_predictions schema')
    print('=' * 65)

    sc, mv = load_scaler()
    pidx = mv.index('PRECIP')

    te = np.load(os.path.join(SRC_DIR, 'preprocessed_test.npz'), allow_pickle=True)
    data_orig = inverse_transform(sc, te['data'].astype(np.float32))
    dates = pd.to_datetime(te['dates'])
    sids  = te['station_ids']
    y_true_full = data_orig[:, pidx]

    frames = []
    missing = []

    for seed in SEEDS:
        for scen_label, mask_key in SCENARIOS.items():
            npy_path = os.path.join(
                SRC_DIR, f'direct_two_stage_rf_test_seed{seed}_{scen_label}.npy')
            if not os.path.exists(npy_path):
                missing.append(os.path.basename(npy_path))
                continue

            arr_norm = np.load(npy_path).astype(np.float32)
            arr_orig = inverse_transform(sc, arr_norm)
            y_pred_full = arr_orig[:, pidx]

            art_mask = te[mask_key].astype(np.float32)
            mask_precip = art_mask[:, pidx].astype(bool)

            rows = pd.DataFrame({
                'scenario':    scen_label,
                'seed':        str(seed),
                'method':      METHOD_LABEL,
                'date':        dates[mask_precip].strftime('%Y-%m-%d'),
                'station_id':  sids[mask_precip],
                'y_true':      np.round(y_true_full[mask_precip], 4),
                'y_pred':      np.round(np.clip(y_pred_full[mask_precip], 0, None), 4),
                'is_masked':   1,
                'is_true_wet': (y_true_full[mask_precip] > WET_THRESH).astype(int),
                'is_pred_wet': (np.clip(y_pred_full[mask_precip], 0, None) > WET_THRESH).astype(int),
            })
            frames.append(rows)
            print(f'  [seed {seed} | {scen_label}] {len(rows):,} masked PRECIP rows exported')

    if missing:
        print(f'\n  [WARN] Missing prediction files (skipped): {missing}')

    if not frames:
        print('\n  [ERROR] No D2S-RF prediction files found. '
              'Run direct_two_stage_rf.py first.')
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(['scenario', 'seed', 'date', 'station_id']).reset_index(drop=True)

    out_path = os.path.join(RESULTS_DIR, 'direct_two_stage_rf_masked_predictions.csv')
    df.to_csv(out_path, index=False)

    print(f'\n  -> {out_path}  ({len(df):,} rows total)')
    print(f'  Seeds: {SEEDS}  Scenarios: {list(SCENARIOS.keys())}')
    print('  Done.')
    print('=' * 65)


if __name__ == '__main__':
    main()

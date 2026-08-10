"""
compute_p95_threshold.py
==========================
Derives the canonical extreme-event PRECIP threshold (P95_THRESH) from the
train+val wet-day distribution only -- never from test.

This script exists because the previous hardcoded value (16.74 mm) was
found, on independent verification, to match the TEST partition's p95
(16.735mm) rather than train+val (20.80mm on the pre-2005-restricted
dataset) -- i.e. the "extreme event" threshold had quietly been calibrated
on the evaluation data. Run this script any time the underlying dataset or
train/val split changes, and copy its printed value into the P95_THRESH
constant in canonical_metrics.py (and nowhere else -- every other script
should import P95_THRESH from canonical_metrics, not hardcode its own copy).
"""
import os
import pickle
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WET_THRESH = 0.1


def main():
    with open(os.path.join(SRC_DIR, 'scaler.pkl'), 'rb') as f:
        d = pickle.load(f)
    sc = d['scaler']
    pidx = list(d['meteo_vars']).index('PRECIP')

    def real_precip(split):
        z = np.load(os.path.join(SRC_DIR, f'preprocessed_{split}.npz'), allow_pickle=True)
        mask = z['real_mask'][:, pidx].astype(bool)
        inv = sc.inverse_transform(np.nan_to_num(z['data'].astype(np.float64), nan=0.0))[:, pidx]
        return inv[mask]

    p_tr = real_precip('train')
    p_va = real_precip('val')
    p_te = real_precip('test')  # reported for transparency only -- NOT used for the threshold

    for name, p in [('train', p_tr), ('val', p_va), ('test', p_te)]:
        wet = p[p > WET_THRESH]
        print(f'{name:6s}  n_obs={len(p):6d}  n_wet={len(wet):5d}  '
              f'p95={np.percentile(wet, 95):.4f} mm')

    trainval_wet = np.concatenate([p_tr, p_va])
    trainval_wet = trainval_wet[trainval_wet > WET_THRESH]
    p95 = float(np.percentile(trainval_wet, 95))
    print(f'\ntrain+val  n_wet={len(trainval_wet)}  p95={p95:.4f} mm')
    print(f'\nP95_THRESH = {round(p95, 2)}   <- copy into canonical_metrics.py')


if __name__ == '__main__':
    main()

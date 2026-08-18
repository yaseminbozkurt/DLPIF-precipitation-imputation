"""
repackage_saits_val_10pct.py
==============================
Produces full-length SAITS validation-set imputations under the
corrupted_10pct scenario (preprocessed_val.npz), the same scenario
train_saits_v2.py already uses internally for its own validation loss
(see that script's X_va/X_va_ori construction) -- inference-only reuse of
the already-trained, already-selected checkpoint, no retraining.

This exists to give the SAITS-backbone RQ3 conformal robustness check
(stage2_conformal_uq_saits.py) a genuine VAL-CAL calibration split,
identical in construction to repackage_saits_outputs.py's test-set
repackaging (truncated non-overlapping windows + one overlapping tail
window per station to cover the remainder rows) -- just pointed at
preprocessed_val.npz instead of preprocessed_test.npz.

Output: src/saits_val_seed{seed}_10pct.npy (n_full_val, 7), normalised
[0,1], PRECIP included -- same array convention as
saits_test_seed{seed}_{scenario}.npy.
"""
import os
import pickle
import numpy as np

from repackage_saits_outputs import (
    find_last_checkpoint, tail_window_predictions,
)
from train_saits_v2 import (
    N_STEPS, N_FEATURES, SAITS_CFG, to_log1p, log1p_to_mm, mm_to_norm,
    build_3d, reconstruct_flat, PROJECT_DIR, EXPERIMENTS_DIR,
)

SRC_DIR = PROJECT_DIR


def repackage_val_seed(seed):
    print(f'\n{"="*70}\n  Repackaging SAITS VAL (corrupted_10pct) for seed={seed}\n{"="*70}')

    with open(os.path.join(SRC_DIR, "scaler.pkl"), "rb") as f:
        sc_data = pickle.load(f)
    sc = sc_data["scaler"]
    mvars = list(sc_data["meteo_vars"])
    pidx = mvars.index("PRECIP")

    npz_va = np.load(os.path.join(SRC_DIR, "preprocessed_val.npz"), allow_pickle=True)
    ids_va = npz_va["station_ids"]
    station_list = list(dict.fromkeys(ids_va.tolist()))
    n_full = len(npz_va["corrupted_10pct"])
    n_sta = len(station_list)
    per_station_full = n_full // n_sta
    n_drop = per_station_full % N_STEPS
    print(f'  n_full={n_full}  n_sta={n_sta}  per_station_full={per_station_full}  '
         f'N_STEPS={N_STEPS}  n_drop/station={n_drop}')

    exp_dir = os.path.join(EXPERIMENTS_DIR, f"saits_v2_seed{seed}")
    ckpt_path, last_epoch = find_last_checkpoint(exp_dir)
    print(f'  Loading checkpoint: epoch {last_epoch} -> {ckpt_path}')

    from pypots.imputation import SAITS
    saits = SAITS(n_steps=N_STEPS, n_features=N_FEATURES, **SAITS_CFG,
                  epochs=1, patience=None, device="cpu",
                  saving_path=None, model_saving_strategy=None, verbose=False)
    saits.load(ckpt_path)

    corr_log = to_log1p(npz_va["corrupted_10pct"].astype(np.float32), pidx, sc)
    X_va_3d = build_3d(corr_log, ids_va)
    imp3d = saits.impute({"X": X_va_3d}).astype(np.float32)
    flat_norm_trunc = reconstruct_flat(imp3d, ids_va)
    trunc_per_sta = per_station_full - n_drop
    assert flat_norm_trunc.shape[0] == n_sta * trunc_per_sta

    if n_drop == 0:
        full_norm = flat_norm_trunc.astype(np.float32)
    else:
        tail_imp_log = tail_window_predictions(saits, corr_log, ids_va, station_list)
        full_norm = np.empty((n_full, N_FEATURES), dtype=np.float32)
        for si, sid in enumerate(station_list):
            sta_rows_trunc = flat_norm_trunc[si::n_sta]
            assert len(sta_rows_trunc) == trunc_per_sta
            tail_log = tail_imp_log[sid][-n_drop:]
            tail_mm = log1p_to_mm(tail_log, pidx)
            tail_norm = mm_to_norm(tail_mm, pidx, sc)
            sta_rows_full = np.concatenate([sta_rows_trunc, tail_norm], axis=0)
            full_norm[si::n_sta] = sta_rows_full

    out_path = os.path.join(SRC_DIR, f"saits_val_seed{seed}_10pct.npy")
    np.save(out_path, full_norm.astype(np.float32))
    print(f'  -> {out_path} ({full_norm.shape[0]} rows)')


if __name__ == "__main__":
    for seed in [42, 123, 456]:
        repackage_val_seed(seed)

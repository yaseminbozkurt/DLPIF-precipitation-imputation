"""
repackage_saits_outputs.py
============================
Post-processing step, run AFTER train_saits_v2.py has completed for a given
seed (reads its saved last-epoch checkpoint; does not retrain anything).

train_saits_v2.py's non-overlapping N_STEPS(=30)-day windowing drops each
station's trailing (rows_per_station % N_STEPS) rows -- the exact count
depends on the dataset size and is NOT a fixed number (it was 5/station on
the original 1973-2023 dataset at 2795 rows/station; it is 22/station on
the post-2005-restricted dataset at 1042 rows/station -- recomputed below,
never hardcoded, since silently reusing an old drop-count would either
crash on a shape mismatch or, worse, leave real masked evaluation cells
with no genuine SAITS prediction).

This script reloads each seed's trained model and runs ONE additional
overlapping inference window per station (the station's final N_STEPS rows,
which overlaps the last regular window but also covers the previously
untouched final `n_drop` rows), splices those `n_drop` rows onto the
existing (truncated) reconstruction, and writes a full (n_full,7)
normalised array to the canonical location/naming used by
build_canonical_outputs.py:
  src/saits_test_seed{seed}_{scenario}.npy

Usage:
  python baselines/repackage_saits_outputs.py --seed 42
  python baselines/repackage_saits_outputs.py --seed 42 --seed 123 --seed 456
"""
import os, sys, glob, re, argparse, pickle
import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for train_saits_v2 import

from train_saits_v2 import (  # noqa: E402  (reuse exact transform/windowing logic)
    N_STEPS, N_FEATURES, SAITS_CFG, to_log1p, log1p_to_mm, mm_to_norm,
    build_3d, reconstruct_flat, PROJECT_DIR, EXPERIMENTS_DIR,
)

SCENARIOS = {
    "10pct":       ("corrupted_10pct",       "art_mask_10pct"),
    "20pct":       ("corrupted_20pct",       "art_mask_20pct"),
    "block7d":     ("corrupted_block7d",     "art_mask_block7d"),
    "block30d":    ("corrupted_block30d",    "art_mask_block30d"),
    "netblock30d": ("corrupted_netblock30d", "art_mask_netblock30d"),
}


def find_last_checkpoint(exp_dir):
    ckpts = glob.glob(os.path.join(exp_dir, "**", "SAITS_epoch*.pypots"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f'no SAITS_epoch*.pypots checkpoint found under {exp_dir}')
    def epoch_num(p):
        m = re.search(r"epoch(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return max(ckpts, key=epoch_num), max(epoch_num(c) for c in ckpts)


def tail_window_predictions(saits, corr_log, ids_te, station_list):
    """For each station, run inference on the final N_STEPS rows (an
    overlapping window vs. the regular non-overlapping windowing) and
    return {station_id: (N_STEPS, F) imputed array in log1p-norm space}."""
    out = {}
    for sid in station_list:
        sta = corr_log[ids_te == sid].astype(np.float32)
        tail = sta[-N_STEPS:][None, :, :]   # (1, N_STEPS, F)
        imp = saits.impute({"X": tail}).astype(np.float32)[0]  # (N_STEPS, F)
        out[sid] = imp
    return out


def repackage_seed(seed):
    print(f'\n{"="*70}\n  Repackaging SAITS outputs for seed={seed}\n{"="*70}')

    with open(os.path.join(SRC_DIR, "scaler.pkl"), "rb") as f:
        sc_data = pickle.load(f)
    sc = sc_data["scaler"]
    mvars = list(sc_data["meteo_vars"])
    pidx = mvars.index("PRECIP")

    npz_te = np.load(os.path.join(SRC_DIR, "preprocessed_test.npz"), allow_pickle=True)
    ids_te = npz_te["station_ids"]
    station_list = list(dict.fromkeys(ids_te.tolist()))
    n_full = len(npz_te["data"])
    n_sta = len(station_list)
    per_station_full = n_full // n_sta
    n_drop = per_station_full % N_STEPS  # rows/station never covered by non-overlapping windows
    print(f'  n_full={n_full}  n_sta={n_sta}  per_station_full={per_station_full}  '
         f'N_STEPS={N_STEPS}  n_drop/station={n_drop}')
    if n_drop == 0:
        print('  n_drop=0: no tail padding needed, truncated output already covers all rows.')

    exp_dir = os.path.join(EXPERIMENTS_DIR, f"saits_v2_seed{seed}")
    ckpt_path, last_epoch = find_last_checkpoint(exp_dir)
    print(f'  Loading checkpoint: epoch {last_epoch} -> {ckpt_path}')

    from pypots.imputation import SAITS
    saits = SAITS(n_steps=N_STEPS, n_features=N_FEATURES, **SAITS_CFG,
                  epochs=1, patience=None, device="cpu",
                  saving_path=None, model_saving_strategy=None, verbose=False)
    saits.load(ckpt_path)

    # existing (truncated, 11160-row) predictions from train_saits_v2.py
    v2_results_dir = os.path.join(SRC_DIR, "results_dl", "saits_v2")

    for scen, (ck, ak) in SCENARIOS.items():
        if ck not in npz_te.files:
            continue
        trunc_path = os.path.join(v2_results_dir,
                                  f"saits_imputed_test_{scen}_v2_seed{seed}_flat.npy")
        if not os.path.exists(trunc_path):
            raise FileNotFoundError(
                f'expected train_saits_v2.py output not found: {trunc_path} '
                f'(run train_saits_v2.py --seed {seed} first)')
        flat_norm_trunc = np.load(trunc_path)  # normalised [0,1], PRECIP included
        trunc_per_sta = per_station_full - n_drop
        assert flat_norm_trunc.shape[0] == n_sta * trunc_per_sta, (
            f'unexpected truncated shape {flat_norm_trunc.shape}, '
            f'expected {n_sta * trunc_per_sta} rows (n_sta={n_sta}, trunc_per_sta={trunc_per_sta})')

        if n_drop == 0:
            full_norm = flat_norm_trunc.astype(np.float32)
        else:
            corr_log = to_log1p(npz_te[ck].astype(np.float32), pidx, sc)
            tail_imp_log = tail_window_predictions(saits, corr_log, ids_te, station_list)

            # rebuild full-length flat array, per-station: first trunc_per_sta
            # rows from the existing truncated reconstruction (already reordered
            # date-major/station-minor by reconstruct_flat), last n_drop rows
            # from the tail window's final n_drop local positions, converted
            # back through the same log1p -> mm -> [0,1] pipeline as
            # train_saits_v2.py.
            full_norm = np.empty((n_full, N_FEATURES), dtype=np.float32)
            for si, sid in enumerate(station_list):
                # truncated rows for this station, in this station's own local time order
                sta_rows_trunc = flat_norm_trunc[si::n_sta]  # date-major stride = n_sta
                assert len(sta_rows_trunc) == trunc_per_sta
                tail_log = tail_imp_log[sid][-n_drop:]                    # (n_drop, F) log1p-norm
                tail_mm = log1p_to_mm(tail_log, pidx)                     # -> mm
                tail_norm = mm_to_norm(tail_mm, pidx, sc)                  # -> [0,1]
                sta_rows_full = np.concatenate([sta_rows_trunc, tail_norm], axis=0)  # (per_station_full, F)
                full_norm[si::n_sta] = sta_rows_full

        out_path = os.path.join(SRC_DIR, f"saits_test_seed{seed}_{scen}.npy")
        np.save(out_path, full_norm.astype(np.float32))
        print(f'  [{scen}] {trunc_path.split(os.sep)[-1]} ({flat_norm_trunc.shape[0]} rows) '
             f'+ tail -> {out_path} ({full_norm.shape[0]} rows)')

    print(f'  Done seed={seed}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", required=True)
    args = parser.parse_args()
    for seed in args.seed:
        repackage_seed(seed)


if __name__ == "__main__":
    main()

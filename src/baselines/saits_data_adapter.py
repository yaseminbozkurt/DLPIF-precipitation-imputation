"""
baselines_dl/saits_data_adapter.py
====================================
Minimal adapter: converts existing preprocessed_*.npz files into the
3-D format expected by pypots SAITS — (n_samples, n_steps, n_features).

Rules obeyed:
  - No existing files are modified.
  - No training or evaluation is performed here.
  - Only reads from the parent directory; writes nothing.

Usage
-----
    from baselines_dl.saits_data_adapter import load_saits_datasets
    train_set, val_set, test_sets = load_saits_datasets(n_steps=30)

Or run directly for a shape check:
    python baselines_dl/saits_data_adapter.py --n_steps 30
"""

import os
import argparse
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
# Parent dir = project root (one level up from this file's location)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NPZ_PATHS = {
    "train": os.path.join(PROJECT_DIR, "preprocessed_train.npz"),
    "val":   os.path.join(PROJECT_DIR, "preprocessed_val.npz"),
    "test":  os.path.join(PROJECT_DIR, "preprocessed_test.npz"),
}

# Missingness scenarios present in each split
# (key suffix → corrupted array key, art_mask key)
SCENARIOS_COMMON = {
    "10pct":   ("corrupted_10pct",    "art_mask_10pct"),
    "20pct":   ("corrupted_20pct",    "art_mask_20pct"),
}
SCENARIOS_TEST_ONLY = {
    "block7d":  ("corrupted_block7d",  "art_mask_block7d"),
    "block30d": ("corrupted_block30d", "art_mask_block30d"),
}


# ── Core helpers ───────────────────────────────────────────────────────────────

def _split_by_station(arr_2d: np.ndarray, station_ids: np.ndarray) -> dict:
    """
    Split a flat (N, F) array into per-station (T, F) arrays.

    Rows in the NPZ are ordered DATE-first, STATION_ID-second (as sorted by
    01_data_preprocessing.py), so each station contributes every 4th row in
    the same cyclic order.  We use station_ids to make this robust.

    Returns
    -------
    dict  station_id -> (T, F) float32 array  (NaN preserved)
    """
    out = {}
    unique_stations = list(dict.fromkeys(station_ids.tolist()))  # preserve sort order
    for sid in unique_stations:
        mask = station_ids == sid
        out[sid] = arr_2d[mask].astype(np.float32)
    return out


def _window_station(arr_station: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Slide a (T, F) station time series into non-overlapping windows.

    Trailing rows that don't fill a complete window are discarded so that
    every sample has exactly n_steps time steps.

    Returns
    -------
    (n_windows, n_steps, n_features) float32
    """
    T, F = arr_station.shape
    n_windows = T // n_steps
    if n_windows == 0:
        raise ValueError(
            f"n_steps={n_steps} is larger than station time-series length {T}. "
            "Use a smaller n_steps."
        )
    # Trim tail, then reshape
    trimmed = arr_station[: n_windows * n_steps]          # (n_windows * n_steps, F)
    return trimmed.reshape(n_windows, n_steps, F)          # (n_windows, n_steps, F)


def _build_3d(arr_2d: np.ndarray, station_ids: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Full pipeline: (N, F) flat  →  (n_samples, n_steps, F) windowed.

    Windows from all stations are concatenated along the sample axis.
    """
    per_station = _split_by_station(arr_2d, station_ids)
    windows = [_window_station(v, n_steps) for v in per_station.values()]
    return np.concatenate(windows, axis=0)                 # (n_samples, n_steps, F)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_saits_datasets(n_steps: int = 30) -> tuple:
    """
    Load and convert all three splits into SAITS-ready dicts.

    Parameters
    ----------
    n_steps : int
        Window length (time steps per sample).  Use 30 or 90 to match the
        ablation runs already in the project (seq30 / seq90).

    Returns
    -------
    train_set : dict
        {'X': (n, n_steps, 7), 'X_ori': (n, n_steps, 7),
         'masks': {'10pct': ..., '20pct': ...}}

        'X'     — data with REAL missing values as NaN (training input)
        'X_ori' — same array *without* artificial corruption (ground truth
                  for loss; equals 'data' from the NPZ, NaN where truly missing)

    val_set : dict   (same structure as train_set)

    test_sets : dict  scenario_name -> dict
        Each entry: {'X': corrupted_3d, 'X_ori': gt_3d, 'art_mask': mask_3d}

        'X'        — artificially corrupted (model input at inference)
        'X_ori'    — clean ground truth (for external evaluation only)
        'art_mask' — 1 where values were artificially hidden (bool uint8)
    """

    results = {}

    for split in ("train", "val", "test"):
        npz = np.load(NPZ_PATHS[split], allow_pickle=True)
        data        = npz["data"].astype(np.float32)        # (N, 7), NaN = real missing
        station_ids = npz["station_ids"]

        # Ground-truth 3-D array (NaN preserved for truly missing cells)
        X_ori_3d = _build_3d(data, station_ids, n_steps)

        # For train/val X == X_ori (no extra artificial corruption at this stage;
        # pypots will apply its own internal masking during training if X_ori is given)
        X_3d = X_ori_3d.copy()

        # Per-scenario corrupted arrays (keep same masks as the rest of the project)
        scenarios = dict(SCENARIOS_COMMON)
        if split == "test":
            scenarios.update(SCENARIOS_TEST_ONLY)

        masks_3d = {}
        for scen_label, (ck, ak) in scenarios.items():
            if ck not in npz.files:
                continue
            corrupted = npz[ck].astype(np.float32)
            art_mask  = npz[ak].astype(np.float32)
            masks_3d[scen_label] = {
                "X":        _build_3d(corrupted, station_ids, n_steps),
                "art_mask": _build_3d(art_mask,  station_ids, n_steps).astype(np.uint8),
            }

        results[split] = {
            "X":     X_3d,
            "X_ori": X_ori_3d,
            "masks": masks_3d,
        }

    train_set = results["train"]
    val_set   = results["val"]
    test_sets = {
        scen: {
            "X":        results["test"]["masks"][scen]["X"],
            "X_ori":    results["test"]["X_ori"],
            "art_mask": results["test"]["masks"][scen]["art_mask"],
        }
        for scen in results["test"]["masks"]
    }

    return train_set, val_set, test_sets


# ── CLI shape check ────────────────────────────────────────────────────────────

def _print_shapes(n_steps: int) -> None:
    print("=" * 60)
    print(f"  SAITS Data Adapter — shape check  (n_steps={n_steps})")
    print("=" * 60)

    train_set, val_set, test_sets = load_saits_datasets(n_steps=n_steps)

    print("\n-- TRAIN --")
    print(f"  X      : {train_set['X'].shape}   (n_samples, n_steps, n_features)")
    print(f"  X_ori  : {train_set['X_ori'].shape}")
    for scen, d in train_set["masks"].items():
        print(f"  mask [{scen}]  X={d['X'].shape}  art_mask={d['art_mask'].shape}")

    print("\n-- VAL --")
    print(f"  X      : {val_set['X'].shape}")
    print(f"  X_ori  : {val_set['X_ori'].shape}")
    for scen, d in val_set["masks"].items():
        print(f"  mask [{scen}]  X={d['X'].shape}  art_mask={d['art_mask'].shape}")

    print("\n-- TEST (per scenario) --")
    for scen, d in test_sets.items():
        print(f"  [{scen}]")
        print(f"    X        : {d['X'].shape}")
        print(f"    X_ori    : {d['X_ori'].shape}")
        print(f"    art_mask : {d['art_mask'].shape}  "
              f"(masked cells: {int(d['art_mask'].sum()):,})")

    print("\n-- NaN sanity check (train X) --")
    x = train_set["X"]
    nan_pct = 100.0 * np.isnan(x).sum() / x.size
    print(f"  NaN in train X : {nan_pct:.2f}%  (real missing only)")

    print("\n  [OK] Adapter ready. Pass train_set['X'] to SAITS.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAITS data adapter shape check")
    parser.add_argument(
        "--n_steps", type=int, default=30,
        help="Window length in time steps (default: 30)"
    )
    args = parser.parse_args()
    _print_shapes(args.n_steps)

# -*- coding: utf-8 -*-
"""
05_generate_seed_outputs.py
===========================
Generates three types of outputs from existing imputation artifacts:

OUTPUT 1 — results/metrics_seed{42,123,456}.csv
    Per-seed aggregate metrics (method x scenario). Filtered from existing
    clean_full_evaluation.csv and multiseed_clean_evaluation.csv.

OUTPUT 2 — results/masked_predictions_all.csv
    Row-level PRECIP predictions at artificially-masked positions.
    Columns: scenario, seed, method, date, station_id, y_true, y_pred,
             is_masked, is_true_wet, is_pred_wet

OUTPUT 3 — results/filled_datasets/filled_{method}_{scenario}_seed{seed}.csv
    Complete test-period dataset with imputed values substituted only at
    artificially-masked positions. Original observed values are preserved.
    Columns (sade versiyon — precip focused):
        date, station_id, lat, lon, elev,
        TMIN, TMEAN, TMAX, RH_MEAN, P_MEAN, WIND_MEAN, PRECIP,
        PRECIP_original, PRECIP_was_imputed,
        scenario, method, seed

Stations:  17155 (KUTAHYA), 17704 (TAVSANLI), 17748 (SIMAV), 17750 (GEDIZ)
Test set:  2016-05-07 → 2023-12-31  (N=11180, 4 stations × 2795 unique dates)
SAITS:     N=11160 → first 11160 rows of NPZ (last 20 rows = last 5 days × 4 stations)

Methods available:
    Deterministic (no seed): mean, linear, knn, mice
    Seeded (42, 123, 456):   WGAN-GP_raw, Precip2Stage, DLPIF/AmountRF
    Seeded (7, 42, 123):     SAITS

PrecipFix: aggregate metrics only (no prediction arrays available).
"""

import os
import sys
import io
import glob
import pickle
import warnings
import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FILLED_DIR  = os.path.join(RESULTS_DIR, "filled_datasets")
SAITS_DIR   = os.path.join(SRC_DIR, "results_dl", "saits_v2")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FILLED_DIR, exist_ok=True)

WET_THRESH  = 0.1   # mm — wet-day classification threshold
METEO_VARS  = ["TMIN", "TMEAN", "TMAX", "RH_MEAN", "P_MEAN", "WIND_MEAN", "PRECIP"]
PRECIP_IDX  = METEO_VARS.index("PRECIP")

STATION_META = {
    17155: {"name": "KUTAHYA", "lat": 39.4171, "lon": 29.9891, "elev": 969},
    17704: {"name": "TAVSANLI","lat": 39.5384, "lon": 29.4941, "elev": 833},
    17748: {"name": "SIMAV",   "lat": 39.0925, "lon": 28.9786, "elev": 809},
    17750: {"name": "GEDIZ",   "lat": 38.9947, "lon": 29.4003, "elev": 736},
}

# Scenario label → npz art_mask key  (labels match multiseed_clean_rerun.py SCENARIOS)
SCENARIOS = {
    "10pct":    "art_mask_10pct",
    "20pct":    "art_mask_20pct",
    "block7d":  "art_mask_block7d",
    "block30d": "art_mask_block30d",
}

# Scenario label → baseline_results.pkl scenario key
BASELINE_SCEN_KEY = {
    "10pct":    "10pct",
    "20pct":    "20pct",
    "block7d":  "block7d",
    "block30d": "block30d",
}

# Scenario label used in clean_full_evaluation.csv
EVAL_SCEN_KEY = {
    "10pct":    "10pct",
    "20pct":    "20pct",
    "block7d":  "block7d",
    "block30d": "block30d",
}

# SAITS flat file naming
SAITS_SCEN_KEY = {
    "10pct":    "10pct",
    "20pct":    "20pct",
    "block7d":  "block7d",
    "block30d": "block30d",
}

# Loaders

def load_scaler():
    with open(os.path.join(SRC_DIR, "scaler.pkl"), "rb") as f:
        d = pickle.load(f)
    return d["scaler"]

def inverse_transform(sc, arr_norm):
    """Inverse-transform a (N,7) normalised array; NaN preserved."""
    arr = arr_norm.copy().astype(np.float64)
    nan_m = np.isnan(arr)
    arr[nan_m] = 0.0
    out = sc.inverse_transform(arr)
    out[nan_m] = np.nan
    return out

def load_test_npz():
    return np.load(os.path.join(SRC_DIR, "preprocessed_test.npz"), allow_pickle=True)

def load_baselines():
    with open(os.path.join(SRC_DIR, "baseline_results.pkl"), "rb") as f:
        d = pickle.load(f)
    return d["all_scenarios"]   # {(method_name, scenario_key): arr(N,7)}

# SECTION 1 — Seed-based metrics CSVs

def make_seed_metrics():
    print("\n" + "=" * 65)
    print("  SECTION 1 — Seed-based metrics CSVs")
    print("=" * 65)

    full_path  = os.path.join(RESULTS_DIR, "clean_full_evaluation.csv")
    multi_path = os.path.join(RESULTS_DIR, "multiseed_clean_evaluation.csv")

    if not os.path.exists(full_path):
        print(f"  [SKIP] Not found: {full_path}")
        return
    if not os.path.exists(multi_path):
        print(f"  [SKIP] Not found: {multi_path}")
        return

    df_full  = pd.read_csv(full_path)
    df_multi = pd.read_csv(multi_path)

    # Add seed column
    def extract_seed(method_str):
        for s in ["seed42", "seed123", "seed456", "seed7"]:
            if s in str(method_str):
                return int(s.replace("seed", ""))
        return None

    df_full["seed"]  = df_full["method"].apply(extract_seed)
    df_multi["seed"] = df_multi["method"].apply(extract_seed)

    # Deterministic methods (no seed) → assign seed label "deterministic"
    # They appear in df_full with seed=None; include them in every seed file
    df_det = df_full[df_full["seed"].isna()].copy()
    df_det["seed_label"] = "deterministic"

    for seed in [42, 123, 456]:
        # seeded rows from full eval
        df_s = df_full[df_full["seed"] == seed].copy()
        # seeded rows from multiseed eval (may have different columns)
        df_m = df_multi[df_multi["seed"] == seed].copy()

        # Combine; prefer full_eval rows (more columns)
        combined = pd.concat([df_det, df_s, df_m], ignore_index=True)

        # Remove duplicate method×scenario
        combined = combined.drop_duplicates(subset=["method", "scenario"], keep="first")
        combined = combined.sort_values(["scenario", "method"]).reset_index(drop=True)

        out_path = os.path.join(RESULTS_DIR, f"metrics_seed{seed}.csv")
        combined.to_csv(out_path, index=False)
        print(f"  -> Saved: {out_path}  ({len(combined)} rows)")

    print("  Done.\n")

# SECTION 2 & 3 — Masked predictions + Filled datasets

def make_predictions_and_filled():
    print("=" * 65)
    print("  SECTION 2+3 — Masked predictions & Filled datasets")
    print("=" * 65)

    sc      = load_scaler()
    te      = load_test_npz()
    bl_dict = load_baselines()

    # Ground truth in original units
    data_orig = inverse_transform(sc, te["data"].astype(np.float32))  # (N,7)
    dates      = pd.to_datetime(te["dates"])
    sids       = te["station_ids"]
    N          = len(dates)

    # SAITS has 11160 rows — first 11160 of NPZ
    SAITS_N = 11160

    # ── Collect all imputation arrays per (method, scenario, seed) ────────────
    # Format: {(method_label, scenario_label, seed_label): arr_orig (N,7)}
    # seed_label: int seed or "deterministic"
    imputed = {}

    # 1) Baseline methods (deterministic, all scenarios)
    for scen_label, bl_key in BASELINE_SCEN_KEY.items():
        for method_name in ["mean", "linear", "knn", "mice"]:
            key_bl = (method_name, bl_key)
            if key_bl not in bl_dict:
                print(f"  [WARN] Baseline not found: {key_bl}")
                continue
            arr_norm = bl_dict[key_bl].astype(np.float32)
            arr_orig = inverse_transform(sc, arr_norm)
            imputed[(method_name, scen_label, "deterministic")] = arr_orig

    # 2) WGAN-GP_raw (seeded, scenario-independent imputation array)
    for seed in [42, 123, 456]:
        p = os.path.join(SRC_DIR, f"gan_imputed_test_modeB_seed{seed}.npy")
        if not os.path.exists(p):
            print(f"  [WARN] WGAN-GP raw not found: {os.path.basename(p)}")
            continue
        arr_norm = np.load(p).astype(np.float32)
        arr_orig = inverse_transform(sc, arr_norm)
        for scen_label in SCENARIOS:
            imputed[("WGAN-GP_raw", scen_label, seed)] = arr_orig

    # 3) Precip2Stage (seeded, scenario-SPECIFIC)
    for seed in [42, 123, 456]:
        # Warn if old scenario-agnostic file still exists (should not be used)
        old_p = os.path.join(SRC_DIR, f"gan_imputed_test_modeB_seed{seed}_msclean_precip2stage.npy")
        if os.path.exists(old_p):
            print(f"  [WARNING] scenario-agnostic Precip2Stage file detected but ignored: "
                  f"{os.path.basename(old_p)}")
        for scen_label in SCENARIOS:
            p = os.path.join(SRC_DIR,
                f"gan_imputed_test_modeB_seed{seed}_msclean_precip2stage_{scen_label}.npy")
            if not os.path.exists(p):
                print(f"  [WARN] Precip2Stage not found: {os.path.basename(p)}")
                continue
            arr_norm = np.load(p).astype(np.float32)
            arr_orig = inverse_transform(sc, arr_norm)
            imputed[("Precip2Stage", scen_label, seed)] = arr_orig

    # 4) AmountRF/DLPIF (seeded, scenario-SPECIFIC)
    for seed in [42, 123, 456]:
        # Warn if old scenario-agnostic file still exists (should not be used)
        old_p = os.path.join(SRC_DIR, f"gan_imputed_test_modeB_seed{seed}_msclean_amountrf.npy")
        if os.path.exists(old_p):
            print(f"  [WARNING] scenario-agnostic DLPIF file detected but ignored: "
                  f"{os.path.basename(old_p)}")
        for scen_label in SCENARIOS:
            p = os.path.join(SRC_DIR,
                f"gan_imputed_test_modeB_seed{seed}_msclean_amountrf_{scen_label}.npy")
            if not os.path.exists(p):
                print(f"  [WARN] DLPIF/AmountRF not found: {os.path.basename(p)}")
                continue
            arr_norm = np.load(p).astype(np.float32)
            arr_orig = inverse_transform(sc, arr_norm)
            imputed[("DLPIF", scen_label, seed)] = arr_orig

    # 5) SAITS (seeded, scenario-specific flat files)
    for seed in [7, 42, 123]:
        for scen_label, saits_key in SAITS_SCEN_KEY.items():
            fname = f"saits_imputed_test_{saits_key}_v2_seed{seed}_flat.npy"
            p = os.path.join(SAITS_DIR, fname)
            if not os.path.exists(p):
                print(f"  [WARN] SAITS not found: {fname}")
                continue
            arr_norm_saits = np.load(p).astype(np.float32)  # (11160, 7)
            # Pad last 20 rows with NaN to match NPZ (N=11180)
            pad = np.full((N - SAITS_N, 7), np.nan, dtype=np.float32)
            arr_norm_full = np.concatenate([arr_norm_saits, pad], axis=0)
            arr_orig = inverse_transform(sc, arr_norm_full)
            imputed[("SAITS", scen_label, seed)] = arr_orig

    print(f"\n  Collected {len(imputed)} (method, scenario, seed) combinations.")

    # ── Build outputs ─────────────────────────────────────────────────────────
    all_masked_rows = []

    # Per-scenario art_mask (N,7)
    art_masks = {
        scen_label: te[mask_key].astype(np.float32)
        for scen_label, mask_key in SCENARIOS.items()
    }

    for (method_label, scen_label, seed_label), arr_orig in sorted(imputed.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
        art_mask = art_masks[scen_label]            # (N,7)
        mask_precip = art_mask[:, PRECIP_IDX].astype(bool)  # (N,) — masked positions

        y_true = data_orig[:, PRECIP_IDX]           # ground truth PRECIP (N,)
        y_pred = arr_orig[:, PRECIP_IDX]             # imputed PRECIP (N,)

        # ── SECTION 2: Masked predictions CSV (masked positions only) ─────────
        if mask_precip.any():
            rows = {
                "scenario":    scen_label,
                "seed":        str(seed_label),
                "method":      method_label,
                "date":        dates[mask_precip].strftime("%Y-%m-%d"),
                "station_id":  sids[mask_precip],
                "y_true":      np.round(y_true[mask_precip], 4),
                "y_pred":      np.round(np.clip(y_pred[mask_precip], 0, None), 4),
                "is_masked":   1,
                "is_true_wet": (y_true[mask_precip] > WET_THRESH).astype(int),
                "is_pred_wet": (np.clip(y_pred[mask_precip], 0, None) > WET_THRESH).astype(int),
            }
            all_masked_rows.append(pd.DataFrame(rows))

        # ── SECTION 3: Filled dataset CSV ─────────────────────────────────────
        # filled = original value where not masked, imputed value where masked
        precip_orig   = y_true.copy()                       # (N,) original PRECIP
        precip_filled = precip_orig.copy()
        valid_mask    = mask_precip & ~np.isnan(y_pred)
        precip_filled[valid_mask] = np.clip(y_pred[valid_mask], 0, None)
        was_imputed   = mask_precip.astype(int)

        # All 7 vars filled (use original for non-masked, imputed for masked)
        data_filled = data_orig.copy()
        for vi in range(7):
            col_mask = art_mask[:, vi].astype(bool)
            pred_col = arr_orig[:, vi]
            valid_c  = col_mask & ~np.isnan(pred_col)
            if vi == PRECIP_IDX:
                data_filled[valid_c, vi] = np.clip(pred_col[valid_c], 0, None)
            else:
                data_filled[valid_c, vi] = pred_col[valid_c]

        df_filled = pd.DataFrame({
            "date":               dates.strftime("%Y-%m-%d"),
            "station_id":         sids,
            "lat":                [STATION_META.get(s, {}).get("lat", np.nan) for s in sids],
            "lon":                [STATION_META.get(s, {}).get("lon", np.nan) for s in sids],
            "elev":               [STATION_META.get(s, {}).get("elev", np.nan) for s in sids],
            "TMIN":               np.round(data_filled[:, 0], 4),
            "TMEAN":              np.round(data_filled[:, 1], 4),
            "TMAX":               np.round(data_filled[:, 2], 4),
            "RH_MEAN":            np.round(data_filled[:, 3], 4),
            "P_MEAN":             np.round(data_filled[:, 4], 4),
            "WIND_MEAN":          np.round(data_filled[:, 5], 4),
            "PRECIP":             np.round(precip_filled, 4),
            "PRECIP_original":    np.round(precip_orig, 4),
            "PRECIP_was_imputed": was_imputed,
            "scenario":           scen_label,
            "method":             method_label,
            "seed":               str(seed_label),
        })

        # Build filename
        seed_str = f"seed{seed_label}" if isinstance(seed_label, int) else seed_label
        fname_out = f"filled_{method_label}_{scen_label}_{seed_str}.csv"
        out_path  = os.path.join(FILLED_DIR, fname_out)
        df_filled.to_csv(out_path, index=False)
        n_imp = int(was_imputed.sum())
        print(f"  -> {fname_out}  ({len(df_filled)} rows, {n_imp} PRECIP positions imputed)")

    # ── Save masked predictions CSV ───────────────────────────────────────────
    if all_masked_rows:
        df_masked = pd.concat(all_masked_rows, ignore_index=True)
        df_masked = df_masked.sort_values(
            ["scenario", "method", "seed", "date", "station_id"]
        ).reset_index(drop=True)
        out_masked = os.path.join(RESULTS_DIR, "masked_predictions_all.csv")
        df_masked.to_csv(out_masked, index=False)
        print(f"\n  -> masked_predictions_all.csv  ({len(df_masked):,} rows)")

    print("\n  Done.")

# MAIN

def main():
    print("=" * 65)
    print("  05_generate_seed_outputs.py")
    print("  DLPIF Cleanrepo — Seed Metrics + Predictions + Filled Datasets")
    print("=" * 65)

    make_seed_metrics()
    make_predictions_and_filled()

    print("\n" + "=" * 65)
    print("  ALL OUTPUTS COMPLETE")
    print("=" * 65)
    print("  results/metrics_seed42.csv")
    print("  results/metrics_seed123.csv")
    print("  results/metrics_seed456.csv")
    print("  results/masked_predictions_all.csv")
    print("  results/filled_datasets/  (filled_*.csv files)")
    print("=" * 65)

if __name__ == "__main__":
    main()

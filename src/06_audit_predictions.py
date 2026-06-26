# -*- coding: utf-8 -*-
"""
06_audit_predictions.py
=======================
Recomputes precipitation metrics from masked_predictions_all.csv and
compares them against reference values in clean_full_evaluation.csv
and metrics_seed*.csv.

Metrics recomputed (PRECIP-focused, masked positions only):
    freq_gt, freq_pred, bias,
    precision, recall, f1, csi,
    rmse_wet, mae_wet, rmse_p95, mae_p95

Tolerance:  abs(diff) < 1e-3   (warn level)
            abs(diff) < 1e-4   (pass level)

Outputs:
    results/audit_metrics_log.csv   — full comparison table
    results/audit_discrepancies.csv — rows where abs(diff) >= 1e-4
"""

import os
import sys
import io
import warnings
import numpy as np
import pandas as pd

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

WET_THRESH = 0.1   # mm
P95_THRESH = 16.74  # mm  (same as multiseed_clean_rerun.py)

TOL_PASS  = 1e-4
TOL_WARN  = 1e-3

# Method name mapping: pred CSV label → reference CSV label variants
# In masked_predictions_all.csv, method names are:
#   DLPIF, SAITS, WGAN-GP_raw, knn, linear, mean, mice
# In clean_full_evaluation.csv, method names are:
#   AmountRF_clean_seed42, SAITS (not present?), WGAN-GP_raw_seed42,
#   knn, linear, mean, mice, Precip2Stage_clean_seed42, PrecipFix_seed42
# We build a lookup: (pred_method, seed_str) -> list of possible ref method names

def ref_method_names(pred_method: str, seed_val: str) -> list:
    """Return candidate reference method names for a given pred method + seed."""
    if pred_method == "DLPIF":
        if seed_val == "deterministic":
            return [f"AmountRF_clean_seed42", f"AmountRF_clean_seed123",
                    f"AmountRF_clean_seed456"]
        return [f"AmountRF_clean_seed{seed_val}"]
    if pred_method == "WGAN-GP_raw":
        if seed_val == "deterministic":
            return [f"WGAN-GP_raw_seed42"]
        return [f"WGAN-GP_raw_seed{seed_val}"]
    if pred_method == "SAITS":
        if seed_val == "deterministic":
            return []
        return [f"SAITS_seed{seed_val}", f"SAITS"]   # SAITS may not be in full_eval
    if pred_method in ("mean", "linear", "knn", "mice"):
        return [pred_method]
    return [pred_method]

# Scenario name mapping: pred → ref (now identity since labels are unified)
SCEN_MAP = {
    "10pct":    "10pct",
    "20pct":    "20pct",
    "block7d":  "block7d",
    "block30d": "block30d",
}

# Metric computation

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all required metrics on masked positions only."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 0, None)

    gt_wet   = y_true > WET_THRESH
    pred_wet = y_pred > WET_THRESH

    tp = int(( pred_wet &  gt_wet).sum())
    fp = int(( pred_wet & ~gt_wet).sum())
    fn = int((~pred_wet &  gt_wet).sum())
    tn = int((~pred_wet & ~gt_wet).sum())

    freq_gt   = float(gt_wet.mean())
    freq_pred = float(pred_wet.mean())
    bias      = round(freq_pred - freq_gt, 6)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    csi  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # rmse/mae on wet-day ground truth positions
    wet_sel = gt_wet
    if wet_sel.sum() > 0:
        rmse_wet = float(np.sqrt(np.mean((y_pred[wet_sel] - y_true[wet_sel]) ** 2)))
        mae_wet  = float(np.mean(np.abs(y_pred[wet_sel] - y_true[wet_sel])))
    else:
        rmse_wet = mae_wet = np.nan

    # p95 positions (ground truth >= p95 threshold)
    p95_sel = y_true >= P95_THRESH
    if p95_sel.sum() > 0:
        rmse_p95 = float(np.sqrt(np.mean((y_pred[p95_sel] - y_true[p95_sel]) ** 2)))
        mae_p95  = float(np.mean(np.abs(y_pred[p95_sel] - y_true[p95_sel])))
    else:
        rmse_p95 = mae_p95 = np.nan

    return dict(
        freq_gt   = round(freq_gt,   4),
        freq_pred = round(freq_pred, 4),
        bias      = round(bias,      4),
        precision = round(prec,      4),
        recall    = round(rec,       4),
        f1        = round(f1,        4),
        csi       = round(csi,       4),
        rmse_wet  = round(rmse_wet,  4) if not np.isnan(rmse_wet)  else np.nan,
        mae_wet   = round(mae_wet,   4) if not np.isnan(mae_wet)   else np.nan,
        rmse_p95  = round(rmse_p95,  2) if not np.isnan(rmse_p95)  else np.nan,
        mae_p95   = round(mae_p95,   2) if not np.isnan(mae_p95)   else np.nan,
    )

METRIC_COLS = [
    "freq_gt", "freq_pred", "bias",
    "precision", "recall", "f1", "csi",
    "rmse_wet", "mae_wet", "rmse_p95", "mae_p95",
]

# Load reference tables

def load_reference() -> pd.DataFrame:
    """
    Merge clean_full_evaluation.csv and metrics_seed*.csv into one lookup table.
    Normalise column names so they match METRIC_COLS.
    """
    dfs = []

    # clean_full_evaluation.csv
    p = os.path.join(RESULTS_DIR, "clean_full_evaluation.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        # Column aliases
        if "F1" in df.columns and "f1" not in df.columns:
            df = df.rename(columns={"F1": "f1"})
        if "CSI" in df.columns and "csi" not in df.columns:
            df = df.rename(columns={"CSI": "csi"})
        if "RMSE_wet" in df.columns and "rmse_wet" not in df.columns:
            df = df.rename(columns={"RMSE_wet": "rmse_wet"})
        dfs.append(df)

    # metrics_seed*.csv
    for seed in [42, 123, 456]:
        p = os.path.join(RESULTS_DIR, f"metrics_seed{seed}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "F1" in df.columns and "f1" not in df.columns:
                df = df.rename(columns={"F1": "f1"})
            if "CSI" in df.columns and "csi" not in df.columns:
                df = df.rename(columns={"CSI": "csi"})
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No reference CSV files found in results/")

    ref = pd.concat(dfs, ignore_index=True)

    # Keep only columns we need
    keep = ["method", "scenario"] + [c for c in METRIC_COLS if c in ref.columns]
    ref = ref[keep].drop_duplicates(subset=["method", "scenario"]).reset_index(drop=True)
    return ref

def lookup_ref(ref_df: pd.DataFrame, ref_method: str, ref_scen: str) -> pd.Series | None:
    """Return the reference row for a given method × scenario, or None."""
    row = ref_df[(ref_df["method"] == ref_method) & (ref_df["scenario"] == ref_scen)]
    if len(row) == 0:
        return None
    return row.iloc[0]

# MAIN

def main():
    print("=" * 70)
    print("  06_audit_predictions.py")
    print("  Recompute metrics from masked_predictions_all.csv")
    print("  Compare against reference CSVs")
    print("=" * 70)

    # Load predictions
    pred_path = os.path.join(RESULTS_DIR, "masked_predictions_all.csv")
    if not os.path.exists(pred_path):
        print(f"  ERROR: {pred_path} not found. Run 05_generate_seed_outputs.py first.")
        sys.exit(1)

    print(f"\n  Loading masked_predictions_all.csv ...")
    df_pred = pd.read_csv(pred_path, dtype={"seed": str})
    print(f"  Rows: {len(df_pred):,}  Methods: {sorted(df_pred['method'].unique())}")

    # Load reference
    print("  Loading reference metric CSVs ...")
    ref_df = load_reference()
    print(f"  Reference rows: {len(ref_df)}  Methods: {sorted(ref_df['method'].unique())[:8]}...")

    # Group predictions by (method, scenario, seed)
    groups = df_pred.groupby(["method", "scenario", "seed"], sort=True)
    print(f"\n  Groups to evaluate: {len(groups)}")

    audit_rows       = []   # full comparison table
    discrepancy_rows = []   # only rows with abs(diff) >= TOL_PASS

    for (pred_method, pred_scen, seed_val), grp in groups:
        recomputed = compute_metrics(grp["y_true"].values, grp["y_pred"].values)

        ref_scen  = SCEN_MAP.get(pred_scen, pred_scen)
        candidates = ref_method_names(pred_method, seed_val)

        # Try each candidate until we find a match
        ref_row   = None
        matched   = None
        for cand in candidates:
            ref_row = lookup_ref(ref_df, cand, ref_scen)
            if ref_row is not None:
                matched = cand
                break

        if ref_row is None:
            # No reference found — record as "no_ref" for all metrics
            for metric in METRIC_COLS:
                audit_rows.append(dict(
                    method          = pred_method,
                    scenario        = pred_scen,
                    seed            = seed_val,
                    ref_method      = "|".join(candidates) if candidates else "?",
                    metric          = metric,
                    recomputed      = recomputed.get(metric, np.nan),
                    reference       = np.nan,
                    abs_diff        = np.nan,
                    status          = "NO_REF",
                ))
            continue

        # Compare each metric
        for metric in METRIC_COLS:
            rec_val = recomputed.get(metric, np.nan)
            ref_val = ref_row.get(metric, np.nan) if metric in ref_row.index else np.nan

            if pd.isna(ref_val):
                status   = "NO_REF_METRIC"
                abs_diff = np.nan
            elif pd.isna(rec_val):
                status   = "NO_RECOMP"
                abs_diff = np.nan
            else:
                abs_diff = abs(float(rec_val) - float(ref_val))
                if abs_diff < TOL_PASS:
                    status = "PASS"
                elif abs_diff < TOL_WARN:
                    status = "WARN"
                else:
                    status = "FAIL"

            row = dict(
                method     = pred_method,
                scenario   = pred_scen,
                seed       = seed_val,
                ref_method = matched,
                metric     = metric,
                recomputed = round(float(rec_val), 6) if not pd.isna(rec_val) else np.nan,
                reference  = round(float(ref_val), 6) if not pd.isna(ref_val) else np.nan,
                abs_diff   = round(abs_diff, 8)  if not pd.isna(abs_diff)  else np.nan,
                status     = status,
            )
            audit_rows.append(row)
            if status in ("WARN", "FAIL"):
                discrepancy_rows.append(row)

    # ── Save ─────────────────────────────────────────────────────────────────
    df_audit = pd.DataFrame(audit_rows)
    df_disc  = pd.DataFrame(discrepancy_rows) if discrepancy_rows else pd.DataFrame(columns=df_audit.columns)

    audit_out = os.path.join(RESULTS_DIR, "audit_metrics_log.csv")
    disc_out  = os.path.join(RESULTS_DIR, "audit_discrepancies.csv")

    df_audit.to_csv(audit_out, index=False)
    df_disc.to_csv(disc_out,  index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    status_counts = df_audit["status"].value_counts().to_dict()

    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)
    print(f"  Total metric comparisons : {len(df_audit)}")
    for s in ["PASS", "WARN", "FAIL", "NO_REF", "NO_REF_METRIC", "NO_RECOMP"]:
        n = status_counts.get(s, 0)
        if n > 0:
            flag = "  OK" if s == "PASS" else " !!" if s in ("WARN", "FAIL") else "  --"
            print(f"  {flag}  {s:<20}: {n}")

    if discrepancy_rows:
        print("\n  DISCREPANCIES (abs_diff >= 1e-4):")
        print(f"  {'method':<22} {'scenario':<12} {'seed':<14} {'metric':<12} {'recomputed':>12} {'reference':>12} {'abs_diff':>12} status")
        print("  " + "-" * 100)
        for r in discrepancy_rows:
            rec_s = f"{r['recomputed']:.6f}" if not pd.isna(r['recomputed']) else "NaN"
            ref_s = f"{r['reference']:.6f}"  if not pd.isna(r['reference'])  else "NaN"
            dif_s = f"{r['abs_diff']:.2e}"   if not pd.isna(r['abs_diff'])   else "NaN"
            print(f"  {r['method']:<22} {r['scenario']:<12} {str(r['seed']):<14} {r['metric']:<12} {rec_s:>12} {ref_s:>12} {dif_s:>12} {r['status']}")
    else:
        print("\n  All compared metrics are within tolerance (< 1e-4). PASS.")

    print(f"\n  -> audit_metrics_log.csv   ({len(df_audit)} rows)")
    print(f"  -> audit_discrepancies.csv ({len(df_disc)} rows)")
    print("=" * 70)

if __name__ == "__main__":
    main()

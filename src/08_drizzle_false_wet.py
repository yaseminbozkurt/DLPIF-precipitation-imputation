# -*- coding: utf-8 -*-
"""
08_drizzle_false_wet.py
=======================
Drizzle distribution and false wet rate analysis.

For each (method × scenario × seed):
  - Filter masked positions where ground truth is DRY (y_true <= 0.1 mm)
  - Compute distribution of predicted values on those true-dry positions

This reveals two failure modes:
  1. WGAN-GP raw: predicts non-zero precip on dry days (drizzle artefact)
     → same pattern as Mean imputation (constant value ≠ 0)
  2. Methods with good occurrence correction (DLPIF, Precip2Stage):
     → almost all true-dry predictions land in [0, 0.1) bucket

Output columns per row:
  scenario, seed, method,
  n_masked,           -- total masked positions
  n_true_dry,         -- masked positions where y_true <= WET_THRESH
  false_wet_rate,     -- fraction of true-dry where y_pred > WET_THRESH
  median_pred_true_dry,
  mean_pred_true_dry,
  p95_pred_true_dry,
  share_0_0p1,        -- fraction of true-dry preds in [0, 0.1)
  share_0p1_1,        -- fraction of true-dry preds in [0.1, 1)
  share_1_5,          -- fraction of true-dry preds in [1, 5)
  share_gt_5          -- fraction of true-dry preds in [5, inf)

Outputs:
  results/analysis_false_wet_rate.csv
  results/analysis_drizzle_distribution.csv   (same data, wider)
"""

import os, sys, io, warnings
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

# Drizzle bins (mm)
BINS = [(0.0, 0.1), (0.1, 1.0), (1.0, 5.0), (5.0, np.inf)]
BIN_LABELS = ["share_0_0p1", "share_0p1_1", "share_1_5", "share_gt_5"]

METHOD_ORDER = [
    "mean", "linear", "knn", "mice",
    "WGAN-GP_raw",
    "PrecipFix",
    "Precip2Stage",
    "DLPIF",
    "SAITS",
]

SCENARIO_ORDER  = ["10pct", "20pct", "block7d", "block30d"]
SCENARIO_LABELS = {
    "10pct":    "Random 10%",
    "20pct":    "Random 20%",
    "block7d":  "Block 7d",
    "block30d": "Block 30d",
}

def method_sort_key(m):
    try:
        return METHOD_ORDER.index(m)
    except ValueError:
        return len(METHOD_ORDER)

def compute_group(grp: pd.DataFrame) -> dict:
    """Compute all metrics for one (method, scenario, seed) group."""
    masked  = grp[grp["is_masked"] == 1].copy()

    n_masked   = len(masked)
    true_dry   = masked[masked["is_true_wet"] == 0]
    n_true_dry = len(true_dry)

    if n_true_dry == 0:
        return dict(
            n_masked         = n_masked,
            n_true_dry       = 0,
            false_wet_rate   = np.nan,
            median_pred_true_dry = np.nan,
            mean_pred_true_dry   = np.nan,
            p95_pred_true_dry    = np.nan,
            **{lbl: np.nan for lbl in BIN_LABELS},
        )

    preds = true_dry["y_pred"].clip(lower=0).values

    false_wet_rate = float((preds > WET_THRESH).mean())
    median_p = float(np.median(preds))
    mean_p   = float(np.mean(preds))
    p95_p    = float(np.percentile(preds, 95))

    shares = {}
    for lbl, (lo, hi) in zip(BIN_LABELS, BINS):
        if np.isinf(hi):
            shares[lbl] = float((preds >= lo).mean())
        else:
            shares[lbl] = float(((preds >= lo) & (preds < hi)).mean())

    return dict(
        n_masked              = n_masked,
        n_true_dry            = n_true_dry,
        false_wet_rate        = round(false_wet_rate, 6),
        median_pred_true_dry  = round(median_p, 4),
        mean_pred_true_dry    = round(mean_p, 4),
        p95_pred_true_dry     = round(p95_p, 4),
        **{k: round(v, 6) for k, v in shares.items()},
    )

def main():
    print("=" * 65)
    print("  08_drizzle_false_wet.py")
    print("  Drizzle Distribution & False Wet Rate Analysis")
    print("=" * 65)

    pred_path = os.path.join(RESULTS_DIR, "masked_predictions_all.csv")
    if not os.path.exists(pred_path):
        print(f"  ERROR: {pred_path} not found.")
        sys.exit(1)

    print(f"\n  Loading masked_predictions_all.csv ...")
    df = pd.read_csv(pred_path, dtype={"seed": str})
    print(f"  {len(df):,} rows  |  methods: {sorted(df['method'].unique())}")

    # ── Compute per (method, scenario, seed) ─────────────────────────────────
    records = []
    for (method, scenario, seed), grp in df.groupby(
        ["method", "scenario", "seed"], sort=False
    ):
        stats = compute_group(grp)
        records.append(dict(
            scenario = scenario,
            seed     = seed,
            method   = method,
            **stats,
        ))

    result = pd.DataFrame(records)

    # Sort
    result["_morder"] = result["method"].map(method_sort_key)
    result["_sorder"] = result["scenario"].map(
        {s: i for i, s in enumerate(SCENARIO_ORDER)})
    result = result.sort_values(["_sorder", "_morder", "seed"]).drop(
        columns=["_morder", "_sorder"]).reset_index(drop=True)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    fwr_cols = ["scenario", "seed", "method",
                "n_masked", "n_true_dry", "false_wet_rate",
                "median_pred_true_dry", "mean_pred_true_dry", "p95_pred_true_dry"]
    fwr_path = os.path.join(RESULTS_DIR, "analysis_false_wet_rate.csv")
    result[fwr_cols].to_csv(fwr_path, index=False)
    print(f"\n  -> analysis_false_wet_rate.csv  ({len(result)} rows)")

    drz_cols = fwr_cols + BIN_LABELS
    drz_path = os.path.join(RESULTS_DIR, "analysis_drizzle_distribution.csv")
    result[drz_cols].to_csv(drz_path, index=False)
    print(f"  -> analysis_drizzle_distribution.csv  ({len(result)} rows)")

    # ── Console summary: mean across seeds per method × scenario ─────────────
    print("\n" + "=" * 65)
    print("  FALSE WET RATE  (% of true-dry masked positions predicted wet)")
    print("  Averaged across available seeds")
    print("=" * 65)

    agg = (result.groupby(["scenario", "method"])[
            ["false_wet_rate", "mean_pred_true_dry", "p95_pred_true_dry",
             "share_0_0p1", "share_gt_5"]]
           .mean().round(4).reset_index())

    for scen in SCENARIO_ORDER:
        sub = agg[agg["scenario"] == scen].sort_values(
            "_morder" if "_morder" in agg.columns else "method",
            key=lambda x: x.map(method_sort_key)
        )
        print(f"\n  {SCENARIO_LABELS[scen]}")
        print(f"  {'Method':<22} {'FalseWet%':>10} {'mean_pred':>10}"
              f" {'p95_pred':>10} {'share[0-0.1)':>14} {'share[>5mm]':>12}")
        print("  " + "-" * 82)
        for _, row in sub.iterrows():
            fwr_pct = f"{row['false_wet_rate']*100:.2f}%"
            print(f"  {row['method']:<22} {fwr_pct:>10}"
                  f" {row['mean_pred_true_dry']:>10.4f}"
                  f" {row['p95_pred_true_dry']:>10.4f}"
                  f" {row['share_0_0p1']*100:>13.1f}%"
                  f" {row['share_gt_5']*100:>11.1f}%")

    print("\n" + "=" * 65)
    print("  KEY INSIGHT — WGAN-GP vs Mean vs DLPIF on true-dry positions")
    print("=" * 65)

    focus = result[result["method"].isin(["mean", "WGAN-GP_raw", "DLPIF", "Precip2Stage"])]
    focus_agg = (focus.groupby(["method"])[
        ["false_wet_rate", "mean_pred_true_dry", "share_0_0p1", "share_gt_5"]]
        .mean().round(4))
    print(focus_agg.to_string())

    print("\n  Done.")
    print("=" * 65)

if __name__ == "__main__":
    main()

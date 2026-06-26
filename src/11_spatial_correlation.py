# -*- coding: utf-8 -*-
"""
11_spatial_correlation.py
=========================
Calculates inter-station spatial correlation errors for precipitation.

Metrics:
  - pearson_amount_corr_mae: MAE of the Pearson correlation matrix (precipitation amount)
  - spearman_amount_corr_mae: MAE of the Spearman correlation matrix (precipitation amount)
  - wet_occurrence_corr_mae: MAE of the Pearson correlation matrix for binary wet/dry occurrence

The observed reference is derived from PRECIP_original for each filled dataset.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FILLED_DIR  = os.path.join(RESULTS_DIR, "filled_datasets")

WET_THRESH  = 0.1

def compute_corr_mae(df, val_col, obs_col, method='pearson', is_binary=False):
    # Pivot so rows are dates, cols are stations
    pivot_pred = df.pivot(index='date', columns='station_id', values=val_col)
    pivot_obs = df.pivot(index='date', columns='station_id', values=obs_col)

    if is_binary:
        pivot_pred = (pivot_pred > WET_THRESH).astype(int)
        pivot_obs = (pivot_obs > WET_THRESH).astype(int)

    corr_pred = pivot_pred.corr(method=method).values
    corr_obs = pivot_obs.corr(method=method).values

    # Calculate MAE of the upper triangle (excluding diagonal)
    idx = np.triu_indices_from(corr_obs, k=1)

    mae = np.mean(np.abs(corr_obs[idx] - corr_pred[idx]))
    return float(mae)

def main():
    print("=" * 65)
    print("  11_spatial_correlation.py")
    print("  Spatial Consistency Analysis")
    print("=" * 65)

    csv_files = glob.glob(os.path.join(FILLED_DIR, "filled_*.csv"))
    if not csv_files:
        print(f"No filled datasets found in {FILLED_DIR}")
        return

    records = []

    for idx, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        parts = filename.replace("filled_", "").replace(".csv", "").split("_")

        seed_part = parts[-1]
        scenario = parts[-2]
        method = "_".join(parts[:-2])

        if scenario == "random10": scenario = "10pct"
        if scenario == "random20": scenario = "20pct"

        print(f"[{idx+1}/{len(csv_files)}] Processing {method} | {scenario} | {seed_part}")

        df = pd.read_csv(filepath)

        pearson_amount_mae = compute_corr_mae(df, 'PRECIP', 'PRECIP_original', method='pearson', is_binary=False)
        spearman_amount_mae = compute_corr_mae(df, 'PRECIP', 'PRECIP_original', method='spearman', is_binary=False)
        occurrence_mae = compute_corr_mae(df, 'PRECIP', 'PRECIP_original', method='pearson', is_binary=True)

        records.append({
            'method': method,
            'scenario': scenario,
            'seed': seed_part.replace('seed', ''),
            'pearson_amount_corr_mae': pearson_amount_mae,
            'spearman_amount_corr_mae': spearman_amount_mae,
            'wet_occurrence_corr_mae': occurrence_mae
        })

    result_df = pd.DataFrame(records)

    out_csv = os.path.join(RESULTS_DIR, "analysis_spatial_correlation.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"\nDetailed metrics saved to {out_csv}")

    # Aggregate
    agg_df = result_df.groupby(['method', 'scenario', 'seed']).mean().reset_index()
    final_summary = agg_df.groupby(['method', 'scenario'])[['pearson_amount_corr_mae',
                                                            'spearman_amount_corr_mae',
                                                            'wet_occurrence_corr_mae']].mean().reset_index()

    METHOD_ORDER = ["mean", "linear", "knn", "mice", "WGAN-GP_raw", "SAITS", "PrecipFix", "Precip2Stage", "DLPIF"]
    SCENARIO_ORDER = ["10pct", "20pct", "block7d", "block30d"]

    final_summary["_morder"] = final_summary["method"].map(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)
    final_summary["_sorder"] = final_summary["scenario"].map(lambda x: SCENARIO_ORDER.index(x) if x in SCENARIO_ORDER else 99)
    final_summary = final_summary.sort_values(["_sorder", "_morder"])

    md_lines = ["# Spatial Correlation Analysis Summary\n"]
    md_lines.append("Inter-station correlation matrix Mean Absolute Error (MAE) compared to observed values.\n")
    md_lines.append("| Scenario | Method | Pearson Amount MAE | Spearman Amount MAE | Wet Occurrence MAE |")
    md_lines.append("|---|---|---|---|---|")

    for _, row in final_summary.iterrows():
        md_lines.append(f"| {row['scenario']} | {row['method']} | {row['pearson_amount_corr_mae']:.4f} | "
                        f"{row['spearman_amount_corr_mae']:.4f} | {row['wet_occurrence_corr_mae']:.4f} |")

    md_out = os.path.join(RESULTS_DIR, "analysis_spatial_correlation_summary.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Summary markdown saved to {md_out}")
    print("=" * 65)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
10_dry_spell_cdd.py
===================
Calculates dry-spell metrics (Maximum Consecutive Dry Days, mean length, p95 length)
for the observed (original) and imputed (filled) test datasets.

A dry day is defined as PRECIP <= 0.1 mm.
A dry spell is a consecutive sequence of dry days.

Metrics computed per station, then averaged across stations for each filled dataset:
  - CDD (max consecutive dry days)
  - mean_dry_spell_length
  - p95_dry_spell_length
  - n_dry_spells

Outputs:
  - results/analysis_dry_spells.csv (detailed station-level & aggregated)
  - results/analysis_dry_spells_summary.md
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

def get_dry_spell_lengths(precip_array):
    """
    Given a 1D array of precipitation values, returns a list of dry spell lengths.
    A dry spell is consecutive days with PRECIP <= 0.1.
    """
    is_dry = (precip_array <= WET_THRESH).astype(int)
    # Find start and end of dry spells
    # Pad with 0s to catch spells at the very beginning or end
    padded = np.pad(is_dry, (1, 1), mode='constant')
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    lengths = ends - starts
    return lengths

def compute_spell_metrics(lengths):
    if len(lengths) == 0:
        return {
            'cdd': 0.0,
            'mean_dry': 0.0,
            'p95_dry': 0.0,
            'n_spells': 0
        }
    return {
        'cdd': float(np.max(lengths)),
        'mean_dry': float(np.mean(lengths)),
        'p95_dry': float(np.percentile(lengths, 95)),
        'n_spells': len(lengths)
    }

def main():
    print("=" * 65)
    print("  10_dry_spell_cdd.py")
    print("  Dry-spell and CDD Hydrological Index Analysis")
    print("=" * 65)

    csv_files = glob.glob(os.path.join(FILLED_DIR, "filled_*.csv"))
    if not csv_files:
        print(f"No filled datasets found in {FILLED_DIR}")
        return

    records = []

    for idx, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        # Parse filename: filled_{method}_{scenario}_{seed}.csv
        # Seed might be 'seed42', 'deterministic'
        parts = filename.replace("filled_", "").replace(".csv", "").split("_")

        # Scenario is usually the second from last, seed is last
        # But method names might have underscores (WGAN-GP_raw)
        seed_part = parts[-1]
        scenario = parts[-2]
        method = "_".join(parts[:-2])

        # Random10/20 fix mapping just in case
        if scenario == "random10": scenario = "10pct"
        if scenario == "random20": scenario = "20pct"

        print(f"[{idx+1}/{len(csv_files)}] Processing {method} | {scenario} | {seed_part}")

        df = pd.read_csv(filepath)

        # Sort by station and date to ensure temporal order
        df = df.sort_values(['station_id', 'date'])

        for station_id, grp in df.groupby('station_id'):
            # Observed metrics
            obs_lengths = get_dry_spell_lengths(grp['PRECIP_original'].values)
            obs_metrics = compute_spell_metrics(obs_lengths)

            # Imputed metrics (PRECIP column contains the combined series)
            pred_lengths = get_dry_spell_lengths(grp['PRECIP'].values)
            pred_metrics = compute_spell_metrics(pred_lengths)

            records.append({
                'method': method,
                'scenario': scenario,
                'seed': seed_part.replace('seed', ''),
                'station_id': station_id,
                'cdd_obs': obs_metrics['cdd'],
                'cdd_pred': pred_metrics['cdd'],
                'cdd_abs_error': abs(obs_metrics['cdd'] - pred_metrics['cdd']),
                'mean_dry_obs': obs_metrics['mean_dry'],
                'mean_dry_pred': pred_metrics['mean_dry'],
                'mean_dry_abs_error': abs(obs_metrics['mean_dry'] - pred_metrics['mean_dry']),
                'p95_dry_obs': obs_metrics['p95_dry'],
                'p95_dry_pred': pred_metrics['p95_dry'],
                'p95_dry_abs_error': abs(obs_metrics['p95_dry'] - pred_metrics['p95_dry']),
                'n_dry_spells_obs': obs_metrics['n_spells'],
                'n_dry_spells_pred': pred_metrics['n_spells']
            })

    result_df = pd.DataFrame(records)

    # Save detailed station-level results
    out_csv = os.path.join(RESULTS_DIR, "analysis_dry_spells.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"\nDetailed metrics saved to {out_csv}")

    # Aggregate by method, scenario, seed
    agg_df = result_df.groupby(['method', 'scenario', 'seed']).mean().reset_index()
    # Average across seeds for final summary
    final_summary = agg_df.groupby(['method', 'scenario'])[['cdd_obs', 'cdd_pred', 'cdd_abs_error',
                                                            'mean_dry_obs', 'mean_dry_pred', 'mean_dry_abs_error',
                                                            'p95_dry_obs', 'p95_dry_pred', 'p95_dry_abs_error',
                                                            'n_dry_spells_obs', 'n_dry_spells_pred']].mean().reset_index()

    # Create Markdown summary
    METHOD_ORDER = ["mean", "linear", "knn", "mice", "WGAN-GP_raw", "SAITS", "PrecipFix", "Precip2Stage", "DLPIF"]
    SCENARIO_ORDER = ["10pct", "20pct", "block7d", "block30d"]

    final_summary["_morder"] = final_summary["method"].map(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99)
    final_summary["_sorder"] = final_summary["scenario"].map(lambda x: SCENARIO_ORDER.index(x) if x in SCENARIO_ORDER else 99)
    final_summary = final_summary.sort_values(["_sorder", "_morder"])

    md_lines = ["# Dry-spell and CDD Analysis Summary\n"]
    md_lines.append("Averages across 4 stations and all available seeds.\n")
    md_lines.append("| Scenario | Method | Obs CDD | Pred CDD | CDD Error | Obs Mean | Pred Mean | Mean Error | Obs p95 | Pred p95 | p95 Error |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for _, row in final_summary.iterrows():
        md_lines.append(f"| {row['scenario']} | {row['method']} | {row['cdd_obs']:.1f} | {row['cdd_pred']:.1f} | {row['cdd_abs_error']:.2f} | "
                        f"{row['mean_dry_obs']:.2f} | {row['mean_dry_pred']:.2f} | {row['mean_dry_abs_error']:.2f} | "
                        f"{row['p95_dry_obs']:.1f} | {row['p95_dry_pred']:.1f} | {row['p95_dry_abs_error']:.2f} |")

    md_out = os.path.join(RESULTS_DIR, "analysis_dry_spells_summary.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Summary markdown saved to {md_out}")
    print("=" * 65)

if __name__ == "__main__":
    main()

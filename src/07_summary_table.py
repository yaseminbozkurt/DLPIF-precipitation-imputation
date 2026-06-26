# -*- coding: utf-8 -*-
"""
07_summary_table.py
====================
Generates a mean +/- std summary table across seeds (42, 123, 456)
for each Method x Scenario combination.

Metrics reported:
  Bias | F1 | CSI | Wet-day RMSE | Extreme RMSE (p95)

Method groups:
  Deterministic (no seed): mean, linear, knn, mice          -> value only, std = n/a
  Seeded (3 runs):         WGAN-GP_raw, PrecipFix,
                           Precip2Stage, DLPIF/AmountRF     -> mean +/- std
  SAITS (seeds 7,42,123):  -> mean +/- std (3 seeds)

Sources:
  results/clean_full_evaluation.csv   -- all methods incl. WGAN-GP, PrecipFix
  results/multiseed_clean_evaluation.csv -- Precip2Stage, AmountRF (same data)

Outputs:
  results/summary_mean_std.csv         -- machine-readable
  results/summary_mean_std_wide.md     -- publication-ready markdown table
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

METRICS = ["bias", "f1", "csi", "rmse_wet", "rmse_p95"]

METRIC_LABELS = {
    "bias":     "Bias",
    "f1":       "F1",
    "csi":      "CSI",
    "rmse_wet": "RMSE$_{wet}$",
    "rmse_p95": "RMSE$_{p95}$",
}

SCENARIO_ORDER  = ["10pct", "20pct", "block7d", "block30d"]
SCENARIO_LABELS = {
    "10pct":    "Random 10%",
    "20pct":    "Random 20%",
    "block7d":  "Block 7d",
    "block30d": "Block 30d",
}

# Display name and seed group for each method family
METHOD_FAMILIES = [
    # (display_label,  pattern_in_csv,          seeds)
    ("Mean",           "mean",                  None),
    ("Linear",         "linear",                None),
    ("KNN",            "knn",                   None),
    ("MICE",           "mice",                  None),
    ("WGAN-GP (raw)",  "WGAN-GP_raw",           [42, 123, 456]),
    ("PrecipFix",      "PrecipFix",             [42, 123, 456]),
    ("Precip2Stage",   "Precip2Stage_clean",    [42, 123, 456]),
    ("DLPIF",          "AmountRF_clean",        [42, 123, 456]),
]


def load_data():
    df_full = pd.read_csv(os.path.join(RESULTS_DIR, "clean_full_evaluation.csv"))
    df_ms   = pd.read_csv(os.path.join(RESULTS_DIR, "multiseed_clean_evaluation.csv"))
    # Union (full already contains Precip2Stage / AmountRF rows too)
    combined = pd.concat([df_full, df_ms], ignore_index=True).drop_duplicates(
        subset=["method", "scenario"]
    )
    return combined

def get_rows(df, pattern, seeds, scenario):
    """Return rows matching pattern x seeds x scenario."""
    scen_df = df[df["scenario"] == scenario]
    if seeds is None:
        # Deterministic: exact match
        rows = scen_df[scen_df["method"] == pattern]
    else:
        # Seeded: method contains pattern + seed
        masks = [scen_df["method"].str.contains(f"{pattern}_seed{s}", regex=False)
                 for s in seeds]
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m
        rows = scen_df[combined_mask]
    return rows

def fmt_cell(vals, metric, suppress_zero_std=True):
    """Format a list of values as 'mean ± std' or single value.

    Decimal places per metric:
        bias        -> 4
        f1, csi     -> 4
        rmse_wet    -> 2
        rmse_p95    -> 2
        (precision, recall also 4)
    If std rounds to zero at chosen precision, show value only (marked *).
    """
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) == 0:
        return "—"
    d = 2 if metric in ("rmse_wet", "rmse_p95", "mae_wet", "mae_p95") else 4
    if len(vals) == 1:
        return f"{vals[0]:.{d}f}"
    mu  = np.mean(vals)
    std = np.std(vals, ddof=1)
    # Determine meaningful std decimal places
    ds = d + 1  # one extra digit for std
    std_str = f"{std:.{ds}f}"
    if suppress_zero_std and float(std_str) == 0.0:
        # std rounds to zero — show mean only with dagger to signal zero variation
        return f"{mu:.{d}f}†"
    return f"{mu:.{d}f} ± {std:.{ds}f}"

def build_table(df):
    """Build long-form records: one row per (scenario, method, metric)."""
    records = []
    for scenario in SCENARIO_ORDER:
        for label, pattern, seeds in METHOD_FAMILIES:
            rows = get_rows(df, pattern, seeds, scenario)
            if len(rows) == 0:
                continue
            row_rec = {"scenario": scenario, "method": label}
            for metric in METRICS:
                if metric not in rows.columns:
                    row_rec[metric] = np.nan
                    row_rec[f"{metric}_std"] = np.nan
                    row_rec[f"{metric}_fmt"] = "—"
                    continue
                vals = rows[metric].dropna().tolist()
                row_rec[metric]           = np.mean(vals) if vals else np.nan
                row_rec[f"{metric}_std"]  = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
                row_rec[f"{metric}_fmt"]  = fmt_cell(vals, metric)
            records.append(row_rec)
    return pd.DataFrame(records)

def build_markdown(tbl):
    """Render a wide markdown table: rows = method, columns = scenario × metric."""
    lines = []

    for scenario in SCENARIO_ORDER:
        scen_label = SCENARIO_LABELS[scenario]
        sub = tbl[tbl["scenario"] == scenario].reset_index(drop=True)
        if sub.empty:
            continue

        lines.append(f"\n### {scen_label}\n")

        # Header
        header = "| Method | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
        sep    = "|---|" + "|".join(["---"] * len(METRICS)) + "|"
        lines.append(header)
        lines.append(sep)

        for _, row in sub.iterrows():
            cells = [row["method"]] + [str(row.get(f"{m}_fmt", "—")) for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)

def main():
    print("=" * 65)
    print("  07_summary_table.py — Mean ± Std Summary Table")
    print("=" * 65)

    df = load_data()
    print(f"  Loaded {len(df)} rows from evaluation CSVs.")
    print(f"  Methods: {sorted(df['method'].unique())}\n")

    tbl = build_table(df)

    # ── Save machine-readable CSV ─────────────────────────────────────────────
    csv_cols = (["scenario", "method"] +
                [f"{m}" for m in METRICS] +
                [f"{m}_std" for m in METRICS] +
                [f"{m}_fmt" for m in METRICS])
    csv_cols = [c for c in csv_cols if c in tbl.columns]
    csv_out  = os.path.join(RESULTS_DIR, "summary_mean_std.csv")
    tbl[csv_cols].to_csv(csv_out, index=False)
    print(f"  -> summary_mean_std.csv  ({len(tbl)} rows)")

    # ── Print console table ───────────────────────────────────────────────────
    print()
    for scenario in SCENARIO_ORDER:
        sub = tbl[tbl["scenario"] == scenario]
        if sub.empty:
            continue
        print(f"\n  {'─'*70}")
        print(f"  Scenario: {SCENARIO_LABELS[scenario]}")
        print(f"  {'─'*70}")
        hdr = f"  {'Method':<22}" + "".join(f"  {METRIC_LABELS[m]:<20}" for m in METRICS)
        print(hdr)
        print("  " + "─" * 115)
        for _, row in sub.iterrows():
            line = f"  {row['method']:<22}"
            for m in METRICS:
                cell = str(row.get(f"{m}_fmt", "—"))
                line += f"  {cell:<20}"
            print(line)

    # ── Save markdown ─────────────────────────────────────────────────────────
    md_header = """# Precipitation Imputation — Mean ± Std Summary Table

Metrics computed across seeds 42, 123, 456 (seeded methods) or single run
(deterministic methods: Mean, Linear, KNN, MICE).

| Symbol | Definition |
|---|---|
| Bias | freq_pred − freq_gt (wet-day frequency bias) |
| F1 | Harmonic mean of precision and recall (wet-day classification) |
| CSI | Critical Success Index = TP / (TP + FP + FN) |
| RMSE$_{wet}$ | RMSE on ground-truth wet-day positions (mm) |
| RMSE$_{p95}$ | RMSE on positions where ground truth ≥ 16.74 mm (extreme events, mm) |

*Values shown as mean ± std (std computed with ddof=1 across seeds).*
*Single values indicate deterministic methods or only one seed available.*
"""
    md_body = build_markdown(tbl)
    md_out   = os.path.join(RESULTS_DIR, "summary_mean_std_wide.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_header + md_body + "\n")
    print(f"\n  -> summary_mean_std_wide.md")
    print("=" * 65)

if __name__ == "__main__":
    main()

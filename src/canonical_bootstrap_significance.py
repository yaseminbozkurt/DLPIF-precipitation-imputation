# -*- coding: utf-8 -*-
"""
canonical_bootstrap_significance.py
=====================================
Paired block-bootstrap significance tests for the manuscript's Discussion/
Conclusion claims, recomputed against the canonical (seed-42, primary
realization) predictions -- specifically because the corrected station-wise
Linear baseline changes the DLPIF-vs-Linear comparison materially enough
that the old (pre-fix) p-value/CI no longer applies.

Resampling unit: for each (method-pair, scenario), masked positions are
grouped into contiguous same-station runs (temporally adjacent masked rows
for the same station_id) -- this is the block-bootstrap resampling unit
described in the manuscript (Section 4.4: "contiguous missingness events
(blocks) per station... rather than independent time steps"), applied
uniformly across scenarios so the same methodology covers both random and
block-missingness masks without a separate special case.

Comparisons recomputed (seed 42, the paper's stated "primary model
realization" for all bootstrap results). WGAN-GP/Precip2Stage removed
2026-07-23 (no GPU available to retrain on the post-2005 dataset); SAITS is
the remaining deep-learning comparison baseline:
  1. DLPIF vs Linear  -- F1 (wet-day classification), 10pct & block30d
  2. DLPIF vs SAITS   -- RMSE_wet,                     10pct & block30d
  3. DLPIF vs SAITS   -- Extreme (p95) RMSE,            10pct

Output:
  results/canonical/analysis/bootstrap_significance.csv
  results/canonical/analysis/bootstrap_significance_summary.md
"""
import os
import warnings
import numpy as np
import pandas as pd

import canonical_metrics as cm
import build_canonical_outputs as bco

warnings.filterwarnings('ignore')

ANALYSIS_DIR = os.path.join(bco.CANON_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

N_BOOT = 10000
SEED_FOR_TEST = 42
RNG_SEED = 20260721


def load_pred(method):
    return pd.read_csv(os.path.join(bco.PRED_DIR, f'{method}.csv'), dtype={'seed': str})


def contiguous_blocks(df_sorted):
    """Assign a block id to each row: consecutive rows (already sorted by
    station_id, row_index) belonging to the same station AND differing by
    row_index<=4 (same-date-cycle adjacency; the underlying array is
    DATE-major/STATION-minor with 4 stations per date, so consecutive
    masked days for one station are 4 rows apart) are grouped as one block.
    """
    row_idx = df_sorted['row_index'].to_numpy()
    station = df_sorted['station_id'].to_numpy()
    block_id = np.zeros(len(df_sorted), dtype=np.int64)
    cur = 0
    for i in range(1, len(df_sorted)):
        same_station = station[i] == station[i - 1]
        adjacent_day = (row_idx[i] - row_idx[i - 1]) in (4,)  # 4 stations/date
        if not (same_station and adjacent_day):
            cur += 1
        block_id[i] = cur
    return block_id


def paired_block_bootstrap(gt_a, pred_a, gt_b, pred_b, block_id, metric_fn, n_boot=N_BOOT, rng=None):
    """metric_fn(gt, pred) -> scalar. Returns (point_diff, ci_low, ci_high, p_value)
    for (metric_b - metric_a), resampling whole blocks with replacement.
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    unique_blocks = np.unique(block_id)
    n_blocks = len(unique_blocks)
    block_index_map = {b: np.where(block_id == b)[0] for b in unique_blocks}

    point = metric_fn(gt_b, pred_b) - metric_fn(gt_a, pred_a)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_index_map[b] for b in sampled_blocks])
        diffs[i] = metric_fn(gt_b[idx], pred_b[idx]) - metric_fn(gt_a[idx], pred_a[idx])

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: proportion of bootstrap diffs crossing zero, doubled
    p_ge = np.mean(diffs <= 0) if point > 0 else np.mean(diffs >= 0)
    p_value = min(1.0, 2 * p_ge)
    return float(point), float(ci_low), float(ci_high), float(p_value)


def f1_metric(gt, pred):
    return cm.precip_metrics(gt, pred)['f1']

def rmse_wet_metric(gt, pred):
    v = cm.precip_metrics(gt, pred)['rmse_wet']
    return v if v == v else 0.0

def rmse_p95_metric(gt, pred):
    v = cm.precip_metrics(gt, pred)['rmse_p95']
    return v if v == v else 0.0


def get_seed42_arrays(method, scenario):
    df = load_pred(method)
    sub = df[(df['scenario'] == scenario)]
    if method != 'Linear':
        sub = sub[sub['seed'].astype(str) == str(SEED_FOR_TEST)]
    sub = sub.sort_values(['station_id', 'row_index']).reset_index(drop=True)
    return sub


def main():
    print('=' * 70)
    print('  canonical_bootstrap_significance.py')
    print(f'  N_BOOT={N_BOOT}  seed={SEED_FOR_TEST} (primary realization)')
    print('=' * 70)

    comparisons = [
        ('Linear', 'AmountRF_DLPIF', 'f1', f1_metric, '10pct'),
        ('Linear', 'AmountRF_DLPIF', 'f1', f1_metric, 'block30d'),
        ('SAITS', 'AmountRF_DLPIF', 'rmse_wet', rmse_wet_metric, '10pct'),
        ('SAITS', 'AmountRF_DLPIF', 'rmse_wet', rmse_wet_metric, 'block30d'),
        ('SAITS', 'AmountRF_DLPIF', 'rmse_p95', rmse_p95_metric, '10pct'),
        ('SingleStageRF', 'AmountRF_DLPIF', 'f1', f1_metric, '10pct'),
        ('SingleStageRF', 'AmountRF_DLPIF', 'f1', f1_metric, 'block30d'),
        ('DirectTwoStageRF', 'MissingnessIndicatorRF', 'f1', f1_metric, '10pct'),
        ('DirectTwoStageRF', 'MissingnessIndicatorRF', 'f1', f1_metric, '20pct'),
        ('DirectTwoStageRF', 'MissingnessIndicatorRF', 'f1', f1_metric, 'block7d'),
        ('DirectTwoStageRF', 'MissingnessIndicatorRF', 'f1', f1_metric, 'block30d'),
        # Network-wide simultaneous multivariate block scenario (netblock30d):
        # DirectTwoStageRF collapses to F1=0 (occurrence classifier predicts
        # dry everywhere when local+neighbour context is entirely zero-filled).
        # Test whether naive baselines that don't rely on an adaptive
        # threshold significantly outperform it in this specific scenario.
        ('DirectTwoStageRF', 'Linear', 'f1', f1_metric, 'netblock30d'),
        ('DirectTwoStageRF', 'SAITS', 'f1', f1_metric, 'netblock30d'),
    ]

    rng = np.random.default_rng(RNG_SEED)
    records = []
    for method_a, method_b, metric_name, metric_fn, scenario in comparisons:
        da = get_seed42_arrays(method_a, scenario)
        db = get_seed42_arrays(method_b, scenario)
        # Align on row_index (both must cover the identical masked-position set)
        merged = da[['row_index', 'station_id', 'y_true', 'y_pred']].merge(
            db[['row_index', 'y_true', 'y_pred']], on='row_index',
            suffixes=('_a', '_b'))
        assert np.allclose(merged['y_true_a'], merged['y_true_b']), \
            f'{method_a} vs {method_b} @ {scenario}: ground truth mismatch'
        merged = merged.sort_values(['station_id', 'row_index']).reset_index(drop=True)

        block_id = contiguous_blocks(merged)
        n_blocks = len(np.unique(block_id))

        gt = merged['y_true_a'].to_numpy()
        pred_a = merged['y_pred_a'].to_numpy()
        pred_b = merged['y_pred_b'].to_numpy()

        point, lo, hi, p = paired_block_bootstrap(
            gt, pred_a, gt, pred_b, block_id, metric_fn, rng=rng)

        records.append(dict(
            method_a=method_a, method_b=method_b, metric=metric_name, scenario=scenario,
            n_masked=len(merged), n_blocks=n_blocks,
            value_a=round(metric_fn(gt, pred_a), 4), value_b=round(metric_fn(gt, pred_b), 4),
            paired_diff=round(point, 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
            p_value=round(p, 4), significant_at_05=bool(p < 0.05),
        ))
        print(f'  [{method_b} - {method_a}] {metric_name} @ {scenario}: '
              f'diff={point:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  '
              f'n_blocks={n_blocks}')

    df = pd.DataFrame(records)
    out_csv = os.path.join(ANALYSIS_DIR, 'bootstrap_significance.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}')

    md_lines = ['# Bootstrap Significance (recomputed, seed 42 primary realization)', '',
               f'Block-bootstrap ({N_BOOT} resamples), contiguous same-station masked-run '
               'blocks as the resampling unit.', '',
               '| Comparison | Metric | Scenario | Diff (B - A) | 95% CI | p-value | Significant (p<0.05) |',
               '|---|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['method_b']} vs {r['method_a']} | {r['metric']} | {r['scenario']} | "
            f"{r['paired_diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
            f"{r['p_value']:.4f} | {r['significant_at_05']} |")
    md_path = os.path.join(ANALYSIS_DIR, 'bootstrap_significance_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()

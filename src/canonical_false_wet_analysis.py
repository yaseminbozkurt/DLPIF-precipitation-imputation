# -*- coding: utf-8 -*-
"""
canonical_false_wet_analysis.py
=================================
Drizzle distribution and false-wet-rate analysis (manuscript Table 4),
computed exclusively from results/canonical/predictions/*.csv -- the
canonical successor to 08_drizzle_false_wet.py, which read the
(pre-canonical) results/masked_predictions_all.csv.

For each (method, scenario, seed): among masked positions where ground
truth is DRY (y_true <= 0.1 mm), what does the predicted-value
distribution look like? Reveals whether a method (e.g. raw WGAN-GP)
smears low-intensity "drizzle" onto structurally dry days.

Output columns per row:
  method, scenario, seed,
  n_evaluated, n_true_dry, false_wet_rate,
  median_pred_true_dry, mean_pred_true_dry, p95_pred_true_dry,
  share_0_0p1, share_0p1_1, share_1_5, share_gt_5,
  mean_pred_given_false_wet, median_pred_given_false_wet

mean_pred_true_dry / median_pred_true_dry / p95_pred_true_dry are computed
over ALL true-dry positions (correctly-predicted-zero ones included), so
they are diluted by whatever share of true-dry positions a method gets
exactly right -- NOT a "residual false-wet amount" despite informally
being described that way in early drafts of the manuscript text. The
mean_pred_given_false_wet / median_pred_given_false_wet columns added
2026-08-04 are the actually-conditional quantity (mean/median predicted
amount, restricted to the false_wet_rate subset that is > WET_THRESH):
e.g. DLPIF's mean_pred_true_dry of ~0.42 mm is diluted by an ~88%
correctly-dry share and corresponds to a much larger
mean_pred_given_false_wet of ~3.49 mm -- DLPIF's Stage 2 regressor
assigns a full, not residual, amount estimate on the minority of
positions Stage 1 misclassifies as wet, exactly as the manuscript's own
mechanistic explanation already argued, but the previously-reported
unconditional number understated the magnitude by roughly 8x.

Outputs:
  results/canonical/analysis/false_wet_rate.csv
  results/canonical/analysis/false_wet_rate_summary.md
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

WET_THRESH = cm.WET_THRESH
BINS = [(0.0, 0.1), (0.1, 1.0), (1.0, 5.0), (5.0, np.inf)]
BIN_LABELS = ['share_0_0p1', 'share_0p1_1', 'share_1_5', 'share_gt_5']

METHOD_ORDER = ['Mean', 'Linear', 'KNN', 'MICE', 'SAITS', 'AmountRF_DLPIF', 'DirectTwoStageRF']
SCENARIO_ORDER = ['10pct', '20pct', 'block7d', 'block30d']


def compute_group(grp: pd.DataFrame) -> dict:
    true_dry = grp[grp['y_true'] <= WET_THRESH]
    n_evaluated = len(grp)
    n_true_dry = len(true_dry)

    if n_true_dry == 0:
        return dict(n_evaluated=n_evaluated, n_true_dry=0, false_wet_rate=np.nan,
                   median_pred_true_dry=np.nan, mean_pred_true_dry=np.nan,
                   p95_pred_true_dry=np.nan, **{lbl: np.nan for lbl in BIN_LABELS})

    preds = true_dry['y_pred'].clip(lower=0).to_numpy()
    is_false_wet = preds > WET_THRESH
    false_wet_rate = float(is_false_wet.mean())
    false_wet_preds = preds[is_false_wet]

    shares = {}
    for lbl, (lo, hi) in zip(BIN_LABELS, BINS):
        shares[lbl] = (float((preds >= lo).mean()) if np.isinf(hi)
                      else float(((preds >= lo) & (preds < hi)).mean()))

    return dict(
        n_evaluated=n_evaluated, n_true_dry=n_true_dry,
        false_wet_rate=round(false_wet_rate, 6),
        median_pred_true_dry=round(float(np.median(preds)), 4),
        mean_pred_true_dry=round(float(np.mean(preds)), 4),
        p95_pred_true_dry=round(float(np.percentile(preds, 95)), 4),
        mean_pred_given_false_wet=(round(float(np.mean(false_wet_preds)), 4)
                                   if len(false_wet_preds) else np.nan),
        median_pred_given_false_wet=(round(float(np.median(false_wet_preds)), 4)
                                     if len(false_wet_preds) else np.nan),
        **{k: round(v, 6) for k, v in shares.items()},
    )


def main():
    print('=' * 65)
    print('  canonical_false_wet_analysis.py -- Table 4 source')
    print('=' * 65)

    records = []
    for method in bco.METHODS:
        path = os.path.join(bco.PRED_DIR, f'{method}.csv')
        df = pd.read_csv(path, dtype={'seed': str})
        for (scenario, seed), grp in df.groupby(['scenario', 'seed']):
            stats = compute_group(grp)
            records.append(dict(method=method, scenario=scenario, seed=seed, **stats))
        print(f'  [OK] {method}: {df["scenario"].nunique()} scenarios x '
              f'{df["seed"].nunique()} seed(s)')

    result = pd.DataFrame(records)
    result['_m'] = result['method'].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    result['_s'] = result['scenario'].map(lambda s: SCENARIO_ORDER.index(s) if s in SCENARIO_ORDER else 99)
    result = result.sort_values(['_s', '_m', 'seed']).drop(columns=['_m', '_s']).reset_index(drop=True)

    out_csv = os.path.join(ANALYSIS_DIR, 'false_wet_rate.csv')
    result.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}  ({len(result)} rows)')

    agg = (result.groupby(['scenario', 'method'])
           [['false_wet_rate', 'mean_pred_true_dry', 'p95_pred_true_dry',
             'share_0_0p1', 'share_gt_5', 'mean_pred_given_false_wet',
             'median_pred_given_false_wet']]
           .mean().round(4).reset_index())

    md_lines = ['# False-Wet-Rate Analysis (Table 4 source)', '',
               'Computed from results/canonical/predictions/*.csv, mean across seeds. '
               '"Mean pred (true-dry)" is UNCONDITIONAL over all true-dry positions '
               '(diluted by correctly-predicted-zero positions); '
               '"Mean pred | false-wet" is CONDITIONAL on the prediction actually exceeding '
               'the 0.1 mm threshold -- the true residual amount on the misclassified subset.',
               '',
               '| Scenario | Method | False Wet Rate | Mean pred (true-dry, unconditional) | '
               'Mean pred \\| false-wet | p95 pred (true-dry) | share[0,0.1) | share[>5mm] |',
               '|---|---|---|---|---|---|---|---|']
    for scen in SCENARIO_ORDER:
        sub = agg[agg['scenario'] == scen]
        sub = sub.iloc[sub['method'].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99).argsort()]
        for _, r in sub.iterrows():
            md_lines.append(
                f"| {scen} | {r['method']} | {r['false_wet_rate']*100:.2f}% | "
                f"{r['mean_pred_true_dry']:.4f} | {r['mean_pred_given_false_wet']:.4f} | "
                f"{r['p95_pred_true_dry']:.4f} | "
                f"{r['share_0_0p1']*100:.1f}% | {r['share_gt_5']*100:.1f}% |")
    md_path = os.path.join(ANALYSIS_DIR, 'false_wet_rate_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')
    print('=' * 65)


if __name__ == '__main__':
    main()

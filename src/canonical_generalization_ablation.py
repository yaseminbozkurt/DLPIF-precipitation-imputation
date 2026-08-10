# -*- coding: utf-8 -*-
"""
canonical_generalization_ablation.py
======================================
Cross-scenario generalisation ablation (manuscript Table 6): does training
the Stage 1/Stage 2 correction models on the 20% random-missingness
partition (instead of the published 10% partition) change performance
when the resulting models are evaluated across all four test scenarios?

Model A ("published pipeline", 10%-trained, seed 42): reused directly from
results/canonical/predictions/DirectTwoStageRF.csv -- bit-identical to
AmountRF_DLPIF for PRECIP (canonical_audit.csv's
dlpif_vs_direct_equivalence check + check_d2s_dlpif_equivalence.py's
36/36 exact match), so no WGAN-GP backbone needs to be touched to obtain
DLPIF's actual published-pipeline numbers.

Model B (ablation, 20%-trained, seed 42): trained here by reusing
direct_two_stage_rf.py's exact feature-construction and training code
(build_occ_X, build_amt_X, train_occ) with TRAIN_SCENARIO='20pct' /
TRAIN_COR_KEY='corrupted_20pct' instead of the published 10pct, then
evaluated on all four test scenarios via canonical_metrics.precip_metrics.

Both models are scored with the exact same canonical metric implementation
used everywhere else in this pipeline (canonical_metrics.py), so the A/B
comparison isolates the effect of training-scenario choice from any
metric-definition drift.

Outputs:
  results/canonical/analysis/generalization_ablation.csv
  results/canonical/analysis/generalization_ablation_summary.md
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

import canonical_metrics as cm
import build_canonical_outputs as bco

sys.stdout.flush()
import direct_two_stage_rf as d2s  # reuse exact feature/RF code

warnings.filterwarnings('ignore')

ANALYSIS_DIR = os.path.join(bco.CANON_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

SEED = 42
ABLATION_SCENARIO = '20pct'
ABLATION_COR_KEY = 'corrupted_20pct'
SCENARIO_ORDER = ['10pct', '20pct', 'block7d', 'block30d']


def model_a_metrics():
    """Published pipeline (10pct-trained), seed 42, from the canonical
    DirectTwoStageRF predictions (bit-identical to AmountRF_DLPIF)."""
    path = os.path.join(bco.PRED_DIR, 'DirectTwoStageRF.csv')
    df = pd.read_csv(path, dtype={'seed': str})
    df = df[df['seed'] == str(SEED)]
    out = {}
    for scen in SCENARIO_ORDER:
        sub = df[df['scenario'] == scen]
        if sub.empty:
            raise bco.MissingCanonicalInput(
                f'Model A: no DirectTwoStageRF rows for seed {SEED}, scenario {scen}')
        out[scen] = cm.precip_metrics(sub['y_true'].to_numpy(), sub['y_pred'].to_numpy())
    return out, path


def train_model_b(sc, pidx, tr, va, te):
    na_key, nm_key = d2s.neighbor_keys(ABLATION_SCENARIO)
    d2s.require_keys(tr, [ABLATION_COR_KEY, na_key, nm_key], 'train (20pct ablation)')
    d2s.require_keys(va, [ABLATION_COR_KEY, na_key, nm_key], 'val (20pct ablation)')

    X_tr_full = d2s.build_occ_X(tr[ABLATION_COR_KEY].astype(np.float32),
                                tr['temporal'].astype(np.float32),
                                tr[na_key].astype(np.float32),
                                tr[nm_key].astype(np.float32), pidx)
    X_va_full = d2s.build_occ_X(va[ABLATION_COR_KEY].astype(np.float32),
                                va['temporal'].astype(np.float32),
                                va[na_key].astype(np.float32),
                                va[nm_key].astype(np.float32), pidx)
    X_tr_amt = d2s.build_amt_X(sc, tr, ABLATION_COR_KEY, pidx, na_key, nm_key)

    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]

    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > d2s.WET_THRESH) & tr_obs

    X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs] > d2s.WET_THRESH).astype(int)
    X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs] > d2s.WET_THRESH).astype(int)

    occ_rf, occ_cut, vm = d2s.train_occ(X_tr_occ, y_tr, X_va_occ, y_va, SEED)
    print(f'  Model B (20pct-trained) occ RF: cutoff={occ_cut}  val_F1={vm["val_f1"]}')

    from sklearn.ensemble import RandomForestRegressor
    amt_rf = RandomForestRegressor(n_estimators=400, random_state=SEED,
                                   min_samples_leaf=2, n_jobs=-1)
    amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])
    return occ_rf, occ_cut, amt_rf


def model_b_metrics(sc, pidx, te, occ_rf, occ_cut, amt_rf):
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
    out = {}
    for scen_label, cor_key, mask_key in d2s.SCENARIOS:
        na_key, nm_key = d2s.neighbor_keys(scen_label)
        d2s.require_keys(te, [cor_key, mask_key, na_key, nm_key], f'test/{scen_label}')
        X_occ = d2s.build_occ_X(te[cor_key].astype(np.float32),
                                te['temporal'].astype(np.float32),
                                te[na_key].astype(np.float32),
                                te[nm_key].astype(np.float32), pidx)
        X_amt = d2s.build_amt_X(sc, te, cor_key, pidx, na_key, nm_key)

        m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
        gt_m = te_gt[m]

        te_proba = occ_rf.predict_proba(X_occ)[:, 1]
        te_wet_pred = (te_proba >= occ_cut).astype(bool)

        pred_scen = np.zeros_like(te_gt)
        apply_sel = m & te_wet_pred
        if apply_sel.sum() > 0:
            pred_scen[apply_sel] = np.maximum(amt_rf.predict(X_amt[apply_sel]), 0.0)

        out[scen_label] = cm.precip_metrics(gt_m, pred_scen[m])
    return out


def main():
    print('=' * 70)
    print('  canonical_generalization_ablation.py -- Table 6 source')
    print(f'  Model A: published 10pct-trained pipeline (seed {SEED})')
    print(f'  Model B: {ABLATION_SCENARIO}-trained ablation (seed {SEED})')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')

    a_metrics, a_source = model_a_metrics()
    occ_rf, occ_cut, amt_rf = train_model_b(sc, pidx, tr, va, te)
    b_metrics = model_b_metrics(sc, pidx, te, occ_rf, occ_cut, amt_rf)

    records = []
    for scen in SCENARIO_ORDER:
        a = a_metrics[scen]; b = b_metrics[scen]
        records.append(dict(
            scenario=scen,
            model_a_f1=a['f1'], model_b_f1=b['f1'],
            delta_f1=round(b['f1'] - a['f1'], 4),
            model_a_rmse_p95=a['rmse_p95'], model_b_rmse_p95=b['rmse_p95'],
            delta_rmse_p95=(round(b['rmse_p95'] - a['rmse_p95'], 2)
                            if a['rmse_p95'] == a['rmse_p95'] and b['rmse_p95'] == b['rmse_p95']
                            else np.nan),
            model_a_source=a_source,
        ))
        print(f'  [{scen}] A: F1={a["f1"]:.4f} RMSE_p95={a["rmse_p95"]}   '
              f'B: F1={b["f1"]:.4f} RMSE_p95={b["rmse_p95"]}   '
              f'dF1={records[-1]["delta_f1"]:+.4f}  dRMSE_p95={records[-1]["delta_rmse_p95"]:+.2f}')

    df = pd.DataFrame(records)
    out_csv = os.path.join(ANALYSIS_DIR, 'generalization_ablation.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n  -> {out_csv}')

    max_abs_df1 = float(df['delta_f1'].abs().max())
    md_lines = ['# Cross-Scenario Generalisation Ablation (Table 6 source)', '',
               f'Model A: published pipeline, trained on 10% random missingness (seed {SEED}). '
               f'Model B: ablation, trained on {ABLATION_SCENARIO} random missingness (seed {SEED}). '
               'Both evaluated on all four test scenarios using the identical canonical metric '
               'implementation (canonical_metrics.precip_metrics).', '',
               f'Maximum |ΔF1| across all four scenarios: **{max_abs_df1:.4f}**', '',
               '| Scenario | Model A F1 | Model B F1 | ΔF1 | Model A RMSE_p95 | Model B RMSE_p95 | ΔRMSE_p95 |',
               '|---|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['scenario']} | {r['model_a_f1']:.4f} | {r['model_b_f1']:.4f} | "
            f"{r['delta_f1']:+.4f} | {r['model_a_rmse_p95']:.2f} | {r['model_b_rmse_p95']:.2f} | "
            f"{r['delta_rmse_p95']:+.2f} |")
    md_path = os.path.join(ANALYSIS_DIR, 'generalization_ablation_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'  -> {md_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()

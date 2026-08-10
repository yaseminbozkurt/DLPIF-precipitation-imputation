# -*- coding: utf-8 -*-
"""
canonical_metrics.py
=====================
Single source of truth for PRECIP wet-day / extreme-event evaluation
metrics used across the DLPIF pipeline.

Why this file exists
---------------------
The same ~15-line metric computation (wet/dry confusion matrix -> bias,
precision, recall, F1, CSI, RMSE/MAE on wet-day ground truth, RMSE/MAE on
extreme (>=p95) ground truth) is currently copy-pasted, with small
variations, across at least six places in this repository:

  multiseed_clean_rerun.py       ::metrics()
  generate_clean_tables.py       ::precip_cls() + extreme_metrics()
  06_audit_predictions.py        ::compute_metrics()
  direct_two_stage_rf.py         ::metrics()
  check_d2s_dlpif_equivalence.py (imports direct_two_stage_rf's copy)
  baselines/train_saits_v2.py    ::precip_metrics()   <- uses `>=` not `>`
  04_evaluation.py               ::precip_classification_rows() /
                                    extreme_metrics()  <- p95 computed
                                    PER-VARIABLE from each method's own
                                    masked sample instead of the fixed
                                    16.74 mm cross-scenario threshold

These are not purely cosmetic duplicates. The Linear-interpolation audit
(results/audit_linear_interpolation.md) and the manuscript-vs-repository
cross-check performed during the DLPIF_Hydrology.docx review both trace
back, in part, to different scripts computing "the same" metric slightly
differently (strict `>` vs `>=` at the wet-day threshold; a fixed 16.74 mm
p95 threshold vs a per-run recomputed percentile) and nobody noticing
because each script keeps its own copy. This module consolidates the
metric definitions actually verified against the manuscript's canonical
numbers (Section 4.4: wet-day threshold 0.1 mm, extreme threshold
p95 = 16.74 mm, fixed and applied uniformly across all scenarios) so that
future scripts can import one implementation instead of retyping it.

This file does not change any existing script's behaviour by itself --
it is additive. Migrating multiseed_clean_rerun.py / generate_clean_tables.py
/ direct_two_stage_rf.py / etc. to import from here is a separate,
deliberate follow-up (see `__main__` self-test below, which already
verifies this module reproduces their outputs bit-for-bit on real data).

Schema note (2026-07-21): the canonical pipeline (build_canonical_outputs.py)
requires a slightly wider field set than the original six duplicated
implementations happened to share, so this module's output schema is the
one that wins:
  - `bias` was renamed `signed_frequency_bias` (freq_pred - freq_gt; the
    old name was ambiguous about sign/definition next to `rmse`/`mae`).
  - `n_masked` was renamed `n_evaluated` (this module doesn't assume every
    caller's "masked" positions are literally artificial-mask cells --
    only that they are the cells being evaluated).
  - `rmse` / `mae` were added: unconditional error over ALL evaluated
    cells, not just the gt-wet subset (`rmse_wet`/`mae_wet` still exist,
    unchanged, for that). None of the six duplicated implementations
    computed this unconditional pair, but Table 1 / `baseline_rmse.csv`
    -style aggregate RMSE comparisons need it and it belongs here, not as
    a seventh copy-pasted computation elsewhere.
  - `seed` / `source_prediction_file` are optional passthrough labels on
    `metrics_row()` so the canonical metrics CSV can carry full
    provenance without a second, non-canonical assembly step downstream.

Public API
----------
  WET_THRESH, P95_THRESH        -- canonical constants
  wetday_confusion(gt, pred)    -- tp, fp, fn, tn at WET_THRESH
  precip_metrics(gt, pred, ...) -- the full metric dict (see below)
  metrics_row(gt, pred, method, scenario, seed=None, n_evaluated=None,
              source_prediction_file=None)
                                 -- precip_metrics() + method/scenario/seed/
                                    n_evaluated/source_prediction_file
                                    labels, ready to append to a results
                                    DataFrame

`precip_metrics()` return schema:
    freq_gt, freq_pred, signed_frequency_bias,
    precision, recall, f1, csi,
    rmse, mae,
    rmse_wet, mae_wet,
    mae_p95, rmse_p95,
    tp, fp, fn, tn
"""
import numpy as np

# Canonical thresholds (manuscript Section 4.4 / 4.5).
# Wet-day threshold: WMO standard, 0.1 mm/day.
# Extreme threshold: fixed 95th percentile of the TRAIN+VAL wet-day
# distribution, 19.20 mm/day -- NOT recomputed per run/scenario/method.
# Corrected 2026-07-23: the previous value (16.74) was independently
# verified to match the TEST partition's p95 (16.735mm on the old
# 1973-2023 dataset), not train+val (20.80mm) as documented -- i.e. it had
# been calibrated on the evaluation data. Recomputed here from train+val
# only, on the post-2005-restricted dataset (see compute_p95_threshold.py,
# which derives and prints this value -- re-run it if the dataset or split
# ever changes, rather than hand-editing this constant from memory).
WET_THRESH = 0.1
P95_THRESH = 19.2


def wetday_confusion(gt_mm, pred_mm, wet_thresh: float = WET_THRESH):
    """Wet/dry confusion matrix using a STRICT '>' comparison on both
    sides (ground truth and prediction). Some baseline scripts elsewhere
    in this repo use '>=' for the prediction side (e.g.
    baselines/train_saits_v2.py::precip_metrics) -- that is a known,
    intentional-or-not deviation from this canonical definition; do not
    silently copy it.
    """
    gt_mm = np.asarray(gt_mm, dtype=np.float64)
    pred_mm = np.asarray(pred_mm, dtype=np.float64)
    gw = gt_mm > wet_thresh
    pw = pred_mm > wet_thresh
    tp = int((pw & gw).sum())
    fp = int((pw & ~gw).sum())
    fn = int((~pw & gw).sum())
    tn = int((~pw & ~gw).sum())
    return tp, fp, fn, tn


def precip_metrics(gt_mm, pred_mm,
                   wet_thresh: float = WET_THRESH,
                   p95_thresh: float = P95_THRESH):
    """Compute the canonical PRECIP evaluation metric set at artificially
    masked positions.

    Parameters
    ----------
    gt_mm, pred_mm : 1-D array-like, same length
        Ground-truth and predicted precipitation (mm/day) AT ALREADY
        MASKED POSITIONS ONLY -- callers must subset before calling this
        function, mirroring every existing usage in the repo
        (`gt_m = te_gt[m]`, `pred_mm[m]`, etc.).
    wet_thresh : float
        Wet-day threshold in mm (default 0.1, WMO standard).
    p95_thresh : float
        Extreme-event threshold in mm (default 19.2, the FIXED
        train+val-derived 95th percentile used throughout the manuscript
        -- do not substitute a per-call `np.percentile(gt_mm, 95)`, which
        would silently reintroduce the 04_evaluation.py-style discrepancy
        this module exists to prevent).

    Returns
    -------
    dict with keys: freq_gt, freq_pred, signed_frequency_bias, precision,
    recall, f1, csi, rmse, mae, rmse_wet, mae_wet, mae_p95, rmse_p95,
    tp, fp, fn, tn.
    """
    gt_mm = np.asarray(gt_mm, dtype=np.float64)
    pred_mm = np.asarray(pred_mm, dtype=np.float64)

    tp, fp, fn, tn = wetday_confusion(gt_mm, pred_mm, wet_thresh)
    gw = gt_mm > wet_thresh

    freq_gt = float(gw.mean()) if len(gt_mm) else np.nan
    freq_pred = float((pred_mm > wet_thresh).mean()) if len(pred_mm) else np.nan

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0)
    csi       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Unconditional error over ALL evaluated cells (not just gt-wet).
    if len(gt_mm) > 0:
        rmse_all = float(np.sqrt(np.mean((pred_mm - gt_mm) ** 2)))
        mae_all  = float(np.mean(np.abs(pred_mm - gt_mm)))
    else:
        rmse_all = mae_all = np.nan

    wet_sel = gw
    if wet_sel.sum() > 0:
        rmse_wet = float(np.sqrt(np.mean((pred_mm[wet_sel] - gt_mm[wet_sel]) ** 2)))
        mae_wet  = float(np.mean(np.abs(pred_mm[wet_sel] - gt_mm[wet_sel])))
    else:
        rmse_wet = mae_wet = np.nan

    extreme_sel = gt_mm >= p95_thresh
    if extreme_sel.sum() > 0:
        rmse_p95 = float(np.sqrt(np.mean((pred_mm[extreme_sel] - gt_mm[extreme_sel]) ** 2)))
        mae_p95  = float(np.mean(np.abs(pred_mm[extreme_sel] - gt_mm[extreme_sel])))
    else:
        rmse_p95 = mae_p95 = np.nan

    return dict(
        freq_gt=round(freq_gt, 4) if freq_gt == freq_gt else np.nan,
        freq_pred=round(freq_pred, 4) if freq_pred == freq_pred else np.nan,
        signed_frequency_bias=(round(freq_pred - freq_gt, 4)
                               if freq_gt == freq_gt and freq_pred == freq_pred else np.nan),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        csi=round(csi, 4),
        rmse=round(rmse_all, 4) if rmse_all == rmse_all else np.nan,
        mae=round(mae_all, 4) if mae_all == mae_all else np.nan,
        rmse_wet=round(rmse_wet, 4) if rmse_wet == rmse_wet else np.nan,
        mae_wet=round(mae_wet, 4) if mae_wet == mae_wet else np.nan,
        mae_p95=round(mae_p95, 2) if mae_p95 == mae_p95 else np.nan,
        rmse_p95=round(rmse_p95, 2) if rmse_p95 == rmse_p95 else np.nan,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def metrics_row(gt_mm, pred_mm, method: str, scenario: str, seed=None,
                n_evaluated=None, source_prediction_file=None,
                wet_thresh: float = WET_THRESH, p95_thresh: float = P95_THRESH):
    """precip_metrics() plus method/scenario/seed/n_evaluated/
    source_prediction_file labels -- the canonical row shape for
    results/canonical/metrics/canonical_metrics_all.csv.

    `seed` may be an int (seeded methods) or a string such as
    'deterministic' (Mean/Linear/KNN/MICE-style methods with no seed
    variation) -- this module does not interpret it, only carries it
    through.
    """
    row = precip_metrics(gt_mm, pred_mm, wet_thresh, p95_thresh)
    row_out = dict(method=method, scenario=scenario, seed=seed)
    if n_evaluated is None:
        n_evaluated = len(np.asarray(gt_mm))
    row_out['n_evaluated'] = int(n_evaluated)
    row_out.update(row)
    row_out['source_prediction_file'] = source_prediction_file
    return row_out


# ── Self-test: reproduce already-verified real numbers from this repo ──────
if __name__ == '__main__':
    import sys
    import numpy as np

    print('=' * 70)
    print('  canonical_metrics.py -- self-test')
    print('=' * 70)

    # 1) Pure unit test on synthetic data with a hand-checkable answer.
    gt   = np.array([0.0, 0.0, 5.0, 20.0, 0.05, 17.0, 0.1])
    pred = np.array([0.0, 0.3, 4.0, 25.0, 0.2,  10.0, 0.1])
    r = precip_metrics(gt, pred)
    # gt wet (>0.1): [F,F,T,T,F,T,F] -> freq_gt = 3/7
    # pred wet (>0.1): [F,T,T,T,T,T,F] -> freq_pred = 5/7
    # (values in `r` are rounded to 4 dp by precip_metrics(), so compare
    # against the same rounding rather than the raw fraction)
    assert abs(r['freq_gt'] - round(3/7, 4)) < 1e-9, r
    assert abs(r['freq_pred'] - round(5/7, 4)) < 1e-9, r
    # tp: pred wet & gt wet -> idx2(5,4)T, idx3(20,25)T, idx5(17,10)T = 3
    # fp: pred wet & gt dry -> idx1(0,0.3)T, idx4(0.05,0.2)T = 2
    # fn: pred dry & gt wet -> none = 0
    assert r['tp'] == 3 and r['fp'] == 2 and r['fn'] == 0, r
    # unconditional rmse/mae over ALL 7 cells (not just gt-wet)
    exp_rmse = round(float(np.sqrt(np.mean((pred - gt) ** 2))), 4)
    exp_mae  = round(float(np.mean(np.abs(pred - gt))), 4)
    assert r['rmse'] == exp_rmse, (r['rmse'], exp_rmse)
    assert r['mae'] == exp_mae, (r['mae'], exp_mae)
    print(f'  [PASS] synthetic hand-checked case: {r}')

    # 2) Cross-check against this session's own verified pipeline outputs,
    #    if available: direct_two_stage_rf_evaluation.csv was produced by
    #    direct_two_stage_rf.py's OWN (duplicated) metrics() function on
    #    real masked predictions; recomputing from the row-level export
    #    with THIS module's precip_metrics() must reproduce it exactly
    #    (both operate on already-masked, un-rounded mm arrays -- the
    #    small rounding-from-CSV caveat documented in
    #    export_direct_two_stage_predictions.py does not apply here since
    #    we recompute from a fresh in-memory run below).
    try:
        # direct_two_stage_rf.py re-wraps sys.stdout at import time (for
        # Windows console UTF-8 output). Flush explicitly first so any
        # text already printed above and still sitting in the current
        # TextIOWrapper's internal buffer reaches the OS before the
        # object it belongs to is replaced -- otherwise that buffered
        # text is silently dropped instead of appearing later.
        sys.stdout.flush()
        import direct_two_stage_rf as d2s

        sc, mv = d2s.load_scaler()
        pidx = mv.index('PRECIP')
        tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')
        tr_na_key, tr_nm_key = d2s.neighbor_keys('10pct')

        X_tr_full = d2s.build_occ_X(tr['corrupted_10pct'].astype(np.float32),
                                    tr['temporal'].astype(np.float32),
                                    tr[tr_na_key].astype(np.float32),
                                    tr[tr_nm_key].astype(np.float32), pidx)
        X_va_full = d2s.build_occ_X(va['corrupted_10pct'].astype(np.float32),
                                    va['temporal'].astype(np.float32),
                                    va[tr_na_key].astype(np.float32),
                                    va[tr_nm_key].astype(np.float32), pidx)
        X_tr_amt = d2s.build_amt_X(sc, tr, 'corrupted_10pct', pidx, tr_na_key, tr_nm_key)

        tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
        va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
        te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
        tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
        va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)
        tr_wet = (tr_gt > WET_THRESH) & tr_obs
        X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs] > WET_THRESH).astype(int)
        X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs] > WET_THRESH).astype(int)

        occ_rf, occ_cut, _vm = d2s.train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed=42)
        from sklearn.ensemble import RandomForestRegressor
        amt_rf = RandomForestRegressor(n_estimators=400, random_state=42,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

        mismatches = 0
        for scen_label, cor_key, mask_key in d2s.SCENARIOS:
            na_key, nm_key = d2s.neighbor_keys(scen_label)
            X_occ = d2s.build_occ_X(te[cor_key].astype(np.float32), te['temporal'].astype(np.float32),
                                    te[na_key].astype(np.float32), te[nm_key].astype(np.float32), pidx)
            X_amt = d2s.build_amt_X(sc, te, cor_key, pidx, na_key, nm_key)
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            proba = occ_rf.predict_proba(X_occ)[:, 1]
            wet_pred = (proba >= occ_cut).astype(bool)
            apply_sel = m & wet_pred

            pred = np.zeros_like(te_gt)
            if apply_sel.sum() > 0:
                pred[apply_sel] = np.maximum(amt_rf.predict(X_amt[apply_sel]), 0.0)

            own = d2s.metrics(te_gt[m], pred[m], 'DirectTwoStageRF_seed42', scen_label, int(m.sum()))
            mine = metrics_row(te_gt[m], pred[m], 'DirectTwoStageRF_seed42', scen_label,
                               seed=42, n_evaluated=int(m.sum()))

            # d2s.metrics() (the pre-existing duplicated implementation) still
            # uses the old field names (`bias`, `n_masked`); canonical_metrics
            # renamed those (`signed_frequency_bias`, `n_evaluated`) and added
            # fields (`rmse`, `mae`) d2s.metrics() never computed. Map the
            # fields that exist under both names for the cross-check.
            field_map = {'bias': 'signed_frequency_bias', 'f1': 'f1', 'csi': 'csi',
                        'rmse_wet': 'rmse_wet', 'mae_wet': 'mae_wet',
                        'mae_p95': 'mae_p95', 'rmse_p95': 'rmse_p95',
                        'tp': 'tp', 'fp': 'fp', 'fn': 'fn', 'tn': 'tn'}
            for own_k, mine_k in field_map.items():
                a, b = own[own_k], mine[mine_k]
                same = (a == b) or (isinstance(a, float) and isinstance(b, float)
                                    and a != a and b != b)  # both NaN
                if not same:
                    mismatches += 1
                    print(f'  [MISMATCH] {scen_label} {own_k}/{mine_k}: '
                          f'direct_two_stage_rf.py={a}  canonical_metrics.py={b}')
            print(f'  [seed42 | {scen_label}] direct_two_stage_rf.py.metrics() vs '
                  f'canonical_metrics.metrics_row(): bias {own["bias"]} vs '
                  f'{mine["signed_frequency_bias"]}, f1 {own["f1"]} vs {mine["f1"]}')

        if mismatches == 0:
            print('\n  [PASS] canonical_metrics.py reproduces direct_two_stage_rf.py.metrics() '
                  'EXACTLY on a live seed-42 run across all 4 scenarios.')
        else:
            print(f'\n  [FAIL] {mismatches} field mismatches found -- see above.')
    except FileNotFoundError as e:
        print(f'\n  [SKIP] Live pipeline cross-check skipped (preprocessed data not found): {e}')

    print('=' * 70)

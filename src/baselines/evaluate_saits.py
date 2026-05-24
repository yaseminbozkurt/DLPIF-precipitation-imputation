"""
baselines_dl/evaluate_saits.py
================================
Evaluate SAITS against KNN, MICE, linear and WGAN-GP_modeB_seed42
using the exact same metrics as 04_evaluation.py.

Existing output files are NOT modified.

Alignment note
--------------
SAITS predictions are (11160, 7) — windowing drops the last 5 days
(20 rows) per split.  All arrays are trimmed to N=11160 so every
method is evaluated on the identical rows.

Outputs  →  results_dl/
  evaluation_saits_comparison_seed{N}.csv   RMSE, MAE, Std-RMSE per variable
  evaluation_saits_summary_seed{N}.csv      macro-average per method x scenario
  evaluation_saits_precip_seed{N}.csv       freq, bias, precision, recall, F1, CSI
  evaluation_saits_extreme_seed{N}.csv      MAE/RMSE at masked PRECIP ≥ 16.74 mm (p95)
"""

import os, sys, pickle, warnings, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# P95 extreme threshold — same value used in generate_clean_tables.py
# and multiseed_clean_rerun.py (16.74 mm = dataset-derived 95th percentile)
P95_THRESH = 16.74

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results_dl")
SAITS_V2_DIR = os.path.join(RESULTS_DIR, "saits_v2")  # flat .npy files saved here by train_saits_v2.py
sys.path.insert(0, PROJECT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (mirrors 04_evaluation.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_orig(sc, arr_norm: np.ndarray) -> np.ndarray:
    arr = arr_norm.copy().astype(np.float64)
    nan_m = np.isnan(arr)
    arr[nan_m] = 0.0
    out = sc.inverse_transform(arr)
    out[nan_m] = np.nan
    return out


def rmse(pred, gt, mask):
    sel = mask.astype(bool)
    return float(np.sqrt(np.mean((pred[sel] - gt[sel]) ** 2))) if sel.sum() else np.nan


def mae(pred, gt, mask):
    sel = mask.astype(bool)
    return float(np.mean(np.abs(pred[sel] - gt[sel]))) if sel.sum() else np.nan


def per_var_metrics(pred_o, gt_o, art_mask, meteo_vars, train_stds, method, scenario):
    rows = []
    for i, v in enumerate(meteo_vars):
        r = rmse(pred_o[:, i], gt_o[:, i], art_mask[:, i])
        m = mae(pred_o[:, i],  gt_o[:, i], art_mask[:, i])
        std = train_stds.get(v, 1.0)
        rows.append({
            "method":    method,
            "scenario":  scenario,
            "variable":  v,
            "RMSE_orig": round(r, 4) if not np.isnan(r) else np.nan,
            "MAE_orig":  round(m, 4) if not np.isnan(m) else np.nan,
            "Std_RMSE":  round(r / std, 4) if (not np.isnan(r) and std > 0) else np.nan,
        })
    return rows


def precip_metrics(pred_o, gt_o, art_mask, meteo_vars, method, scenario, thresh=0.1):
    if "PRECIP" not in meteo_vars:
        return {}
    idx = meteo_vars.index("PRECIP")
    sel = art_mask[:, idx].astype(bool)
    if sel.sum() == 0:
        return {}

    pred_mm = pred_o[sel, idx]
    gt_mm   = gt_o[sel, idx]
    pred_wet = pred_mm > thresh
    gt_wet   = gt_mm   > thresh

    tp = int(( pred_wet &  gt_wet).sum())
    fp = int(( pred_wet & ~gt_wet).sum())
    fn = int((~pred_wet &  gt_wet).sum())
    tn = int((~pred_wet & ~gt_wet).sum())

    freq_gt   = float(gt_wet.mean())
    freq_pred = float(pred_wet.mean())
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    csi   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    bias  = freq_pred - freq_gt

    # RMSE on wet-day ground-truth positions (gt > thresh)
    wet_sel = gt_wet
    rmse_wet = float(np.sqrt(np.mean((pred_mm[wet_sel] - gt_mm[wet_sel]) ** 2))) \
               if wet_sel.sum() > 0 else np.nan

    return {
        "method":    method,
        "scenario":  scenario,
        "freq_gt":   round(freq_gt,   4),
        "freq_pred": round(freq_pred, 4),
        "bias":      round(bias,      4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "F1":        round(f1,   4),
        "CSI":       round(csi,  4),
        "RMSE_wet":  round(rmse_wet, 4) if not np.isnan(rmse_wet) else np.nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_masked": int(sel.sum()),
    }


def extreme_precip_metrics(pred_o, gt_o, art_mask, meteo_vars, method, scenario,
                           p95_thresh=P95_THRESH):
    """
    Extreme precipitation metrics at artificially masked PRECIP positions
    where the TRUE value is >= p95_thresh (16.74 mm by default).

    Logic is identical to generate_clean_tables.py::extreme_metrics() and
    multiseed_clean_rerun.py::metrics() — masked positions only, mm units.

    Parameters
    ----------
    pred_o, gt_o  : (N, 7) arrays in ORIGINAL mm units
    art_mask      : (N, 7) float32 — 1 = artificially masked
    p95_thresh    : float  — fixed dataset p95 threshold in mm (16.74)

    Returns
    -------
    dict with keys: method, scenario, n_masked_precip, n_extreme,
                    mae_p95, rmse_p95
    """
    result = {
        "method":           method,
        "scenario":         scenario,
        "p95_thresh_mm":    p95_thresh,
        "n_masked_precip":  0,
        "n_extreme":        0,
        "mae_p95":          np.nan,
        "rmse_p95":         np.nan,
    }

    if "PRECIP" not in meteo_vars:
        return result

    idx = meteo_vars.index("PRECIP")
    # Step 1: restrict to artificially masked PRECIP positions
    sel = art_mask[:, idx].astype(bool)
    if sel.sum() == 0:
        return result

    gt_masked   = gt_o[sel, idx]    # mm, true values at masked positions
    pred_masked = pred_o[sel, idx]  # mm, predicted values at masked positions

    result["n_masked_precip"] = int(sel.sum())

    # Step 2: among those, keep only extreme events (gt >= p95 threshold)
    extreme_sel = gt_masked >= p95_thresh
    n_extreme   = int(extreme_sel.sum())
    result["n_extreme"] = n_extreme

    if n_extreme == 0:
        return result

    gt_ext   = gt_masked[extreme_sel]
    pred_ext = pred_masked[extreme_sel]

    mae_p95  = float(np.mean(np.abs(pred_ext - gt_ext)))
    rmse_p95 = float(np.sqrt(np.mean((pred_ext - gt_ext) ** 2)))

    result["mae_p95"]  = round(mae_p95,  2)
    result["rmse_p95"] = round(rmse_p95, 2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scaler():
    path = os.path.join(PROJECT_DIR, "scaler.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["scaler"], list(d["meteo_vars"])


def load_all_predictions(N: int, scenarios: list, sc, meteo_vars: list, seed: int):
    """
    Returns dict:  method_label -> scenario -> array (N, 7) in ORIGINAL units
    All arrays are trimmed to N rows and inverse-transformed.
    """
    preds = {}

    # -- Baselines (linear / knn / mice) --
    bl_path = os.path.join(PROJECT_DIR, "baseline_results.pkl")
    with open(bl_path, "rb") as f:
        bl_data = pickle.load(f)
    all_bl = bl_data.get("all_scenarios", {})

    for method in ["linear", "knn", "mice"]:
        preds[method] = {}
        for scen in scenarios:
            key = (method, scen)
            if key not in all_bl:
                continue
            arr = all_bl[key].astype(np.float32)[:N]
            preds[method][scen] = sc.inverse_transform(np.clip(arr, 0, 1))

    # -- WGAN-GP modeB seed42 --
    gan_path = os.path.join(PROJECT_DIR, "gan_imputed_test_modeB_seed42.npy")
    if os.path.exists(gan_path):
        gan_arr = np.load(gan_path).astype(np.float32)[:N]
        preds["WGAN-GP_modeB_seed42"] = {
            scen: sc.inverse_transform(np.clip(gan_arr, 0, 1))
            for scen in scenarios
        }

    # -- SAITS (flat, per scenario, seed-parameterized) --
    saits_label = f"SAITS_seed{seed}"
    preds[saits_label] = {}
    for scen in scenarios:
        p = os.path.join(SAITS_V2_DIR, f"saits_imputed_test_{scen}_v2_seed{seed}_flat.npy")
        if not os.path.exists(p):
            print(f"  [WARN] SAITS prediction not found: {p}")
            continue
        arr = np.load(p).astype(np.float32)
        preds[saits_label][scen] = sc.inverse_transform(np.clip(arr, 0, 1))

    return preds, saits_label


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    seed = args.seed
    print("=" * 64)
    print("  SAITS Evaluation  (metrics identical to 04_evaluation.py)")
    print("=" * 64)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    sc, meteo_vars = load_scaler()
    print(f"  Variables : {meteo_vars}")

    # ── Ground truth ──────────────────────────────────────────────────────
    te_npz   = np.load(os.path.join(PROJECT_DIR, "preprocessed_test.npz"),
                       allow_pickle=True)
    gt_norm  = te_npz["data"].astype(np.float32)
    gt_orig_full = inverse_orig(sc, gt_norm)                        # (11180, 7)

    # ── Trim to SAITS window length ───────────────────────────────────────
    # SAITS drops last 5 rows per station (20 rows total) due to windowing.
    # All methods are trimmed to the same N so metrics are comparable.
    N = 11160
    gt_orig = gt_orig_full[:N]
    print(f"  Evaluation rows: {N}  (trimmed from {len(gt_orig_full)} "
          f"to match SAITS window boundary)")

    # ── Train std (for Std-RMSE, same as 04_evaluation.py) ───────────────
    tr_npz     = np.load(os.path.join(PROJECT_DIR, "preprocessed_train.npz"),
                         allow_pickle=True)
    train_orig = inverse_orig(sc, tr_npz["data"].astype(np.float32))
    train_stds = {v: float(np.nanstd(train_orig[:, i]))
                  for i, v in enumerate(meteo_vars)}
    print(f"  Train stds: { {v: round(s,3) for v,s in train_stds.items()} }")

    # ── Scenarios ─────────────────────────────────────────────────────────
    SCENARIOS = {
        "10pct":    "art_mask_10pct",
        "20pct":    "art_mask_20pct",
        "block7d":  "art_mask_block7d",
        "block30d": "art_mask_block30d",
    }
    avail_scen = [s for s, k in SCENARIOS.items() if k in te_npz.files]
    print(f"  Scenarios : {avail_scen}")

    # ── Load predictions (all methods) ────────────────────────────────────
    print("\n  Loading predictions ...")
    preds, saits_label = load_all_predictions(N, avail_scen, sc, meteo_vars, seed)
    print(f"  Methods   : {list(preds.keys())}")

    # ── Compute metrics ───────────────────────────────────────────────────
    records_var     = []    # per variable
    records_precip  = []    # precipitation classification metrics
    records_extreme = []    # extreme precipitation (p95) metrics

    for scen in avail_scen:
        mask_key = SCENARIOS[scen]
        art_mask = te_npz[mask_key].astype(np.float32)[:N]

        n_masked = int(art_mask.sum())
        print(f"\n  -- {scen}  (masked cells={n_masked:,}) --")

        for method, scen_preds in preds.items():
            if scen not in scen_preds:
                continue
            pred_o = scen_preds[scen]               # (N, 7) original units

            var_rows = per_var_metrics(
                pred_o, gt_orig, art_mask, meteo_vars, train_stds, method, scen
            )
            records_var.extend(var_rows)

            pr = precip_metrics(
                pred_o, gt_orig, art_mask, meteo_vars, method, scen
            )
            if pr:
                records_precip.append(pr)

            ext = extreme_precip_metrics(
                pred_o, gt_orig, art_mask, meteo_vars, method, scen
            )
            records_extreme.append(ext)

            # Quick stdout summary per method
            rmse_avg = round(
                float(np.nanmean([r["RMSE_orig"] for r in var_rows])), 4
            )
            f1_val = pr.get("F1", float("nan")) if pr else float("nan")
            print(f"    {method:<26s}  RMSE={rmse_avg:.4f}  "
                  f"F1={f1_val:.4f}" if not np.isnan(f1_val)
                  else f"    {method:<26s}  RMSE={rmse_avg:.4f}")

    # -- Save per-variable detail (seed-suffixed) --
    df_var = pd.DataFrame(records_var)
    path_var = os.path.join(RESULTS_DIR, f"evaluation_saits_comparison_seed{seed}.csv")
    df_var.to_csv(path_var, index=False)
    print(f"\n  Saved: {os.path.basename(path_var)}")

    # -- Save macro-average summary (seed-suffixed) --
    df_summary = (
        df_var.groupby(["method", "scenario"])[["RMSE_orig", "MAE_orig", "Std_RMSE"]]
        .mean().round(4).reset_index()
        .sort_values(["scenario", "RMSE_orig"])
    )
    path_sum = os.path.join(RESULTS_DIR, f"evaluation_saits_summary_seed{seed}.csv")
    df_summary.to_csv(path_sum, index=False)
    print(f"  Saved: {os.path.basename(path_sum)}")

    # -- Save precipitation classification metrics (seed-suffixed) --
    df_precip = pd.DataFrame(records_precip)
    path_pr = os.path.join(RESULTS_DIR, f"evaluation_saits_precip_seed{seed}.csv")
    df_precip.to_csv(path_pr, index=False)
    print(f"  Saved: {os.path.basename(path_pr)}")

    # -- Save extreme precipitation metrics (seed-suffixed) --
    df_extreme = pd.DataFrame(records_extreme)
    path_ext = os.path.join(RESULTS_DIR, f"evaluation_saits_extreme_seed{seed}.csv")
    df_extreme.to_csv(path_ext, index=False)
    print(f"  Saved: {os.path.basename(path_ext)}")

    # -- Print comparison tables --
    METHODS_ORDER = ["linear", "knn", "mice", "WGAN-GP_modeB_seed42", saits_label]

    print("\n" + "=" * 64)
    print("  COMPARISON TABLE — Macro-avg RMSE (original units)")
    print("=" * 64)
    pivot_rmse = (
        df_summary.pivot(index="method", columns="scenario", values="RMSE_orig")
        .reindex([m for m in METHODS_ORDER if m in df_summary["method"].values])
        .round(4)
    )
    print(pivot_rmse.to_string())

    print("\n" + "=" * 64)
    print("  COMPARISON TABLE — PRECIP F1 / CSI / bias")
    print("=" * 64)
    if len(df_precip):
        precip_show = df_precip[["method", "scenario", "freq_gt", "freq_pred",
                                  "bias", "F1", "CSI"]].copy()
        precip_show = precip_show[
            precip_show["method"].isin(METHODS_ORDER)
        ].sort_values(["scenario", "method"])
        print(precip_show.to_string(index=False))

    # ── Table 4: Extreme precipitation (p95) ─────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  TABLE 4 -- Extreme precipitation (masked PRECIP >= {P95_THRESH} mm)")
    print(f"  MAE and RMSE at masked positions where gt >= {P95_THRESH} mm")
    print(f"  Methodology: IDENTICAL to generate_clean_tables.py::extreme_metrics()")
    print("=" * 64)

    if len(df_extreme):
        SCENS_ORD = ["10pct", "20pct", "block7d", "block30d"]
        ext_show  = df_extreme[df_extreme["method"].isin(METHODS_ORDER)].copy()

        if len(ext_show):
            # Wide pivot: one row per method, columns = scenario × metric
            hdr = f"  {'Method':<26}"
            for s in SCENS_ORD:
                if s in ext_show["scenario"].values:
                    hdr += f"  MAE({s[:4]:4s})  RMSE({s[:4]:4s})"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))

            for mth in METHODS_ORDER:
                row_df = ext_show[ext_show["method"] == mth]
                if row_df.empty:
                    continue
                row_str = f"  {mth:<26}"
                for s in SCENS_ORD:
                    sub = row_df[row_df["scenario"] == s]
                    if sub.empty:
                        row_str += f"  {'n/a':>9}  {'n/a':>9}"
                    else:
                        ma = sub["mae_p95"].values[0]
                        rm = sub["rmse_p95"].values[0]
                        ma_s = f"{ma:.2f}" if not (isinstance(ma, float) and np.isnan(ma)) else "n/a"
                        rm_s = f"{rm:.2f}" if not (isinstance(rm, float) and np.isnan(rm)) else "n/a"
                        row_str += f"  {ma_s:>9}  {rm_s:>9}"
                print(row_str)

            # Print n_extreme per scenario for audit
            print("\n  Sample counts (masked PRECIP positions >= 16.74 mm):")
            for s in SCENS_ORD:
                sub = ext_show[(ext_show["scenario"] == s) &
                               (ext_show["method"] == saits_label)]
                if not sub.empty:
                    nm  = sub["n_masked_precip"].values[0]
                    nex = sub["n_extreme"].values[0]
                    pct = round(100.0 * nex / nm, 1) if nm > 0 else 0
                    print(f"    {s:<10}: {nm:,} masked PRECIP → {nex:,} extreme ({pct}%)")
    else:
        print("  [WARN] No extreme metrics available (SAITS predictions missing?).")

    print("\n" + "=" * 64)
    print("  EVALUATION COMPLETE")
    print(f"  Results in: {RESULTS_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAITS vs baselines")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed used when training SAITS (default: 42)"
    )
    args = parser.parse_args()
    main(args)

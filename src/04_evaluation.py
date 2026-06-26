"""
04_evaluation.py
===========================================
ALL metrics reported in ORIGINAL meteorological units.
Normalized metrics (training) are footnote-only.

Inputs  (same directory):
  preprocessed_test.npz  preprocessed_train.npz
  scaler.pkl
  gan_imputed_test_*.npy    (any mode/seq/seed combo)
  baseline_results.pkl

Outputs:
  evaluation_results_orig.csv   — original-unit RMSE/MAE (variable-wise)
  evaluation_stdrmse.csv        — standardized RMSE per variable
  evaluation_extreme.csv        — extreme metrics (p95 TMAX, wet-day freq)
  physical_check_v2.csv         — TMIN≤TMEAN≤TMAX violations (raw + gt)
  fig_timeseries.png            — 300 dpi
  fig_scatter.png               — 300 dpi
  fig_rmse_bar.png              — 300 dpi
"""
import sys, io, os, glob
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import wasserstein_distance, ks_2samp

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_scaler():
    path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['scaler'], list(d['meteo_vars'])

def inverse_orig(sc, arr_norm):
    """Inverse-transform normalised array; NaN preserved."""
    arr = arr_norm.copy().astype(np.float64)
    nan_m = np.isnan(arr)
    arr[nan_m] = 0.0
    out = sc.inverse_transform(arr)
    out[nan_m] = np.nan
    return out

def rmse_val(pred, gt, mask):
    sel = mask.astype(bool)
    return float(np.sqrt(np.mean((pred[sel] - gt[sel]) ** 2))) if sel.sum() else np.nan

def mae_val(pred, gt, mask):
    sel = mask.astype(bool)
    return float(np.mean(np.abs(pred[sel] - gt[sel]))) if sel.sum() else np.nan

def per_var_metrics(pred_orig, gt_orig, art_mask, meteo_vars, train_stds):
    rows = []
    for i, v in enumerate(meteo_vars):
        r = rmse_val(pred_orig[:, i], gt_orig[:, i], art_mask[:, i])
        m = mae_val(pred_orig[:, i],  gt_orig[:, i], art_mask[:, i])
        std_i = train_stds.get(v, 1.0)
        rows.append({
            'variable'        : v,
            'RMSE_orig'       : round(r, 4) if not np.isnan(r) else np.nan,
            'MAE_orig'        : round(m, 4) if not np.isnan(m) else np.nan,
            'Std_RMSE'        : round(r / std_i, 4) if (not np.isnan(r) and std_i > 0) else np.nan,
        })
    return rows

# Seasonal RMSE (DJF/MAM/JJA/SON)
def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return 'DJF'
    if month in (3, 4, 5):
        return 'MAM'
    if month in (6, 7, 8):
        return 'JJA'
    return 'SON'

def _extract_months_from_npz(te_npz):
    """
    Extract month numbers (1–12) from the DATE array stored in NPZ.
    Keeps preprocessing unchanged; expects DATE to exist.
    """
    date_key = None
    for k in ('DATE', 'date', 'dates', 'Date', 'Dates'):
        if k in te_npz.files:
            date_key = k
            break
    if date_key is None:
        raise KeyError(f"DATE array not found in preprocessed_test.npz. Available keys: {list(te_npz.files)}")
    dates = te_npz[date_key]
    # Robust parsing: supports numpy datetime64, pandas-compatible strings, or Python datetimes
    dt = pd.to_datetime(dates, errors='coerce', infer_datetime_format=True)
    if dt.isna().any():
        # Fall back to parsing as YYYYMMDD integers/strings if present
        dt2 = pd.to_datetime(dates.astype(str), errors='coerce', format='%Y%m%d')
        if not dt2.isna().all():
            dt = dt2
    months = dt.month.to_numpy()
    return months

def seasonal_rmse_rows(pred_orig, gt_orig, art_mask, months, meteo_vars, method, scenario):
    rows = []
    for i, v in enumerate(meteo_vars):
        for season in ('DJF', 'MAM', 'JJA', 'SON'):
            season_mask = np.array([_season_from_month(int(m)) == season for m in months], dtype=bool)
            sel = season_mask & art_mask[:, i].astype(bool)
            if sel.sum() == 0:
                r = np.nan
            else:
                r = float(np.sqrt(np.mean((pred_orig[sel, i] - gt_orig[sel, i]) ** 2)))
            rows.append({
                'Season': season,
                'Method': method,
                'Variable': v,
                'RMSE': round(r, 4) if not np.isnan(r) else np.nan,
                'Scenario': scenario,
            })
    return rows

# Distribution-preservation metrics (masked locations only)
def distribution_metrics_rows(pred_orig, gt_orig, art_mask, meteo_vars, method, scenario):
    """
    Compute distribution distances between imputed and true values
    at artificially masked locations, per variable.
    """
    rows = []
    for i, v in enumerate(meteo_vars):
        sel = art_mask[:, i].astype(bool)
        if sel.sum() == 0:
            w = np.nan
            ks = np.nan
        else:
            x = gt_orig[sel, i]
            y = pred_orig[sel, i]
            # Guard against NaNs that could appear after inverse-transform
            m = (~np.isnan(x)) & (~np.isnan(y))
            if m.sum() == 0:
                w = np.nan
                ks = np.nan
            else:
                x = x[m]
                y = y[m]
                w = float(wasserstein_distance(x, y))
                ks = float(ks_2samp(x, y).statistic)
        rows.append({
            'Variable': v,
            'Method': method,
            'Scenario': scenario,
            'Wasserstein': round(w, 6) if not np.isnan(w) else np.nan,
            'KS_stat': round(ks, 6) if not np.isnan(ks) else np.nan,
        })
    return rows

# Station-wise RMSE (masked locations only)
def _extract_stations_from_npz(te_npz):
    """
    Extract station labels/IDs from the test NPZ without changing preprocessing.
    """
    station_key = None
    for k in ('STATION_ID', 'station_id', 'station_ids', 'STATION', 'station', 'Station', 'STATIONS', 'stations'):
        if k in te_npz.files:
            station_key = k
            break
    if station_key is None:
        raise KeyError(
            f"Station array not found in preprocessed_test.npz. Available keys: {list(te_npz.files)}"
        )
    stations = te_npz[station_key]
    # Normalize to a 1D array of strings for grouping/output.
    stations = np.asarray(stations).reshape(-1)
    stations = stations.astype(str)
    return stations

def station_rmse_rows(pred_orig, gt_orig, art_mask, stations, meteo_vars, method, scenario):
    rows = []
    uniq = pd.unique(stations)
    for st in uniq:
        st_sel = (stations == st)
        if st_sel.sum() == 0:
            continue
        for i, v in enumerate(meteo_vars):
            sel = st_sel & art_mask[:, i].astype(bool)
            if sel.sum() == 0:
                r = np.nan
            else:
                r = float(np.sqrt(np.mean((pred_orig[sel, i] - gt_orig[sel, i]) ** 2)))
            rows.append({
                'Station': st,
                'Method': method,
                'Scenario': scenario,
                'Variable': v,
                'RMSE': round(r, 4) if not np.isnan(r) else np.nan,
            })
    return rows

# Extreme metrics
def extreme_metrics(pred_orig, gt_orig, art_mask, meteo_vars):
    result = {}
    # p95 / p99 RMSE for TMAX
    for var in ['TMAX', 'TMIN', 'TMEAN']:
        if var not in meteo_vars:
            continue
        idx = meteo_vars.index(var)
        sel = art_mask[:, idx].astype(bool)
        if sel.sum() == 0:
            continue
        gt_s   = gt_orig[sel, idx]
        pr_s   = pred_orig[sel, idx]
        p95    = np.percentile(gt_s, 95)
        p99    = np.percentile(gt_s, 99)
        sel95  = gt_s >= p95
        sel99  = gt_s >= p99
        r95 = float(np.sqrt(np.mean((pr_s[sel95] - gt_s[sel95])**2))) if sel95.sum() else np.nan
        r99 = float(np.sqrt(np.mean((pr_s[sel99] - gt_s[sel99])**2))) if sel99.sum() else np.nan
        result[f'{var}_p95_RMSE'] = round(r95, 4)
        result[f'{var}_p99_RMSE'] = round(r99, 4)

    # Wet-day frequency comparison (PRECIP > 0)
    if 'PRECIP' in meteo_vars:
        idx = meteo_vars.index('PRECIP')
        sel = art_mask[:, idx].astype(bool)
        if sel.sum() > 0:
            thresh = 0.1   # mm threshold for "wet day"
            gt_wet   = float((gt_orig[sel, idx]   > thresh).mean())
            pred_wet = float((pred_orig[sel, idx]  > thresh).mean())
            result['PRECIP_wetday_gt']   = round(gt_wet,   4)
            result['PRECIP_wetday_pred'] = round(pred_wet, 4)
            result['PRECIP_wetday_bias'] = round(pred_wet - gt_wet, 4)
    return result

# Precipitation classification metrics
def precip_classification_rows(pred_orig, gt_orig, art_mask, meteo_vars,
                                method, scenario, thresh_mm=0.1):
    """Full wet/dry classification metrics at masked PRECIP locations."""
    if 'PRECIP' not in meteo_vars:
        return []
    idx = meteo_vars.index('PRECIP')
    sel = art_mask[:, idx].astype(bool)
    if sel.sum() == 0:
        return []

    pred_mm = pred_orig[sel, idx]
    gt_mm   = gt_orig[sel, idx]

    pred_wet = pred_mm > thresh_mm
    gt_wet   = gt_mm   > thresh_mm

    tp = int(( pred_wet &  gt_wet).sum())
    fp = int(( pred_wet & ~gt_wet).sum())
    fn = int((~pred_wet &  gt_wet).sum())
    tn = int((~pred_wet & ~gt_wet).sum())
    n  = len(pred_mm)

    freq_gt   = float(gt_wet.mean())
    freq_pred = float(pred_wet.mean())
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    dry_a = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    csi   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return [{
        'method'      : method,
        'scenario'    : scenario,
        'thresh_mm'   : thresh_mm,
        'n_masked'    : int(sel.sum()),
        'freq_gt'     : round(freq_gt,   4),
        'freq_pred'   : round(freq_pred,  4),
        'bias'        : round(freq_pred - freq_gt, 4),
        'precision'   : round(prec, 4),
        'recall'      : round(rec,  4),
        'f1'          : round(f1,   4),
        'dry_accuracy': round(dry_a, 4),
        'csi'         : round(csi,  4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }]

def precip_detailed_rows(pred_orig, gt_orig, art_mask, meteo_vars,
                         method, scenario, thresh_mm=0.1):
    """Additional PRECIP diagnostics at masked locations (wet-only errors + FAR/MR)."""
    if 'PRECIP' not in meteo_vars:
        return []
    idx = meteo_vars.index('PRECIP')
    sel = art_mask[:, idx].astype(bool)
    if sel.sum() == 0:
        return []

    pred_mm = pred_orig[sel, idx]
    gt_mm   = gt_orig[sel, idx]

    pred_wet = pred_mm > thresh_mm
    gt_wet   = gt_mm   > thresh_mm

    tp = int(( pred_wet &  gt_wet).sum())
    fp = int(( pred_wet & ~gt_wet).sum())
    fn = int((~pred_wet &  gt_wet).sum())
    tn = int((~pred_wet & ~gt_wet).sum())

    wet_sel = gt_wet
    rmse_wet = float(np.sqrt(np.mean((pred_mm[wet_sel] - gt_mm[wet_sel]) ** 2))) if wet_sel.sum() else np.nan
    mae_wet  = float(np.mean(np.abs(pred_mm[wet_sel] - gt_mm[wet_sel]))) if wet_sel.sum() else np.nan

    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # false alarm rate
    mr  = fn / (fn + tp) if (fn + tp) > 0 else np.nan  # miss rate

    return [{
        'method': method,
        'scenario': scenario,
        'thresh_mm': thresh_mm,
        'n_masked': int(sel.sum()),
        'rmse_wet_only': round(rmse_wet, 4) if not np.isnan(rmse_wet) else np.nan,
        'mae_wet_only': round(mae_wet, 4) if not np.isnan(mae_wet) else np.nan,
        'FAR': round(float(far), 4) if far == far else np.nan,
        'MR': round(float(mr), 4) if mr == mr else np.nan,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }]

# Physical consistency check (raw, no post-processing)
def physical_check_raw(arr_orig, meteo_vars, label):
    result = {'label': label, 'n_rows': len(arr_orig), 'viol_count': np.nan, 'viol_pct': np.nan}
    try:
        ti = meteo_vars.index('TMIN')
        tm = meteo_vars.index('TMEAN')
        tx = meteo_vars.index('TMAX')
    except ValueError:
        return result
    tmin = arr_orig[:, ti]; tmean = arr_orig[:, tm]; tmax = arr_orig[:, tx]
    viol = int(((tmin > tmean) | (tmean > tmax)).sum())
    result['viol_count'] = viol
    result['viol_pct']   = round(100.0 * viol / len(arr_orig), 3)
    return result

# MAIN
def main():
    print("=" * 64)
    print("  04_EVALUATION v2 — Q1 | All metrics in original units")
    print("=" * 64)

    sc, meteo_vars = load_scaler()

    # Train stats → for standardized RMSE
    tr_npz    = np.load(os.path.join(OUTPUT_DIR, 'preprocessed_train.npz'), allow_pickle=True)
    train_norm = tr_npz['data'].astype(np.float32)
    train_orig = inverse_orig(sc, train_norm)
    train_stds = {}
    for i, v in enumerate(meteo_vars):
        col = train_orig[:, i]
        train_stds[v] = float(np.nanstd(col))
    print(f"  Train std per variable: { {v: round(s,3) for v,s in train_stds.items()} }")

    te_npz  = np.load(os.path.join(OUTPUT_DIR, 'preprocessed_test.npz'), allow_pickle=True)
    gt_norm = te_npz['data'].astype(np.float32)
    gt_orig = inverse_orig(sc, gt_norm)
    months  = _extract_months_from_npz(te_npz)
    stations = _extract_stations_from_npz(te_npz)

    # Physical check on GROUND TRUTH first
    phys_rows = [physical_check_raw(gt_orig, meteo_vars, 'ground_truth')]
    print(f"  GT physical violations: {phys_rows[0]['viol_pct']:.3f}%")

    MVP_SCENARIOS = {
        '10pct'   : ('corrupted_10pct',   'art_mask_10pct'),
        '20pct'   : ('corrupted_20pct',   'art_mask_20pct'),
        'block7d' : ('corrupted_block7d', 'art_mask_block7d'),
        'block30d': ('corrupted_block30d','art_mask_block30d'),
    }
    avail = {k: v for k, v in MVP_SCENARIOS.items()
             if v[0] in te_npz.files and v[1] in te_npz.files}
    print(f"  Available scenarios: {list(avail.keys())}")

    # ── Discover GAN imputation files (original + precipfix + precip2stage) ─
    gan_files = glob.glob(os.path.join(OUTPUT_DIR, 'gan_imputed_test_*.npy'))
    gan_imputs = {}
    for f in sorted(gan_files):
        basename = os.path.basename(f)
        tag = basename.replace('gan_imputed_test_', '').replace('.npy', '')
        arr = np.load(f).astype(np.float32)
        # Label precipfix variants clearly so they appear separately in tables
        if tag.endswith('_precip2stage'):
            label = 'WGAN-GP_' + tag.replace('_precip2stage', '') + '+Precip2Stage'
        elif tag.endswith('_amountrf'):
            label = 'WGAN-GP_' + tag.replace('_amountrf', '') + '+AmountRF'
        elif tag.endswith('_precipfinal'):
            label = 'WGAN-GP_' + tag.replace('_precipfinal', '') + '+PrecipFinal'
        elif tag.endswith('_precipfix'):
            label = 'WGAN-GP_' + tag.replace('_precipfix', '') + '+PrecipFix'
        else:
            label = f'WGAN-GP_{tag}'
        gan_imputs[label] = arr
        print(f"  GAN loaded: {tag}  shape={arr.shape}")

    # ── Load Baseline ─────────────────────────────────────────────────────
    bl_path = os.path.join(OUTPUT_DIR, 'baseline_results.pkl')
    if os.path.exists(bl_path):
        with open(bl_path, 'rb') as f:
            bl_data = pickle.load(f)
        all_bl = bl_data.get('all_scenarios', {})
        print(f"  Baseline  : {len(all_bl)} method×scenario combos")
    else:
        all_bl = {}
        print("  [WARN] baseline_results.pkl not found")

    baseline_methods = ['mean', 'linear', 'knn', 'mice']

    # ── Evaluation ────────────────────────────────────────────────────────
    records_orig    = []
    records_ext     = []
    records_seas    = []
    records_dist    = []
    records_station = []
    records_precip_cls = []       # NEW: precipitation classification metrics
    records_precip_det = []       # NEW: precipitation detailed metrics

    for scen_label, (ck, ak) in avail.items():
        art_mask = te_npz[ak].astype(np.float32)

        # GAN(s)
        for gan_tag, gan_norm in gan_imputs.items():
            gan_o = sc.inverse_transform(np.clip(gan_norm, 0, 1))
            rows  = per_var_metrics(gan_o, gt_orig, art_mask, meteo_vars, train_stds)
            for r in rows:
                r.update({'method': gan_tag, 'scenario': scen_label})
                records_orig.append(r)
            records_seas.extend(seasonal_rmse_rows(
                gan_o, gt_orig, art_mask, months, meteo_vars, gan_tag, scen_label
            ))
            records_dist.extend(distribution_metrics_rows(
                gan_o, gt_orig, art_mask, meteo_vars, gan_tag, scen_label
            ))
            records_station.extend(station_rmse_rows(
                gan_o, gt_orig, art_mask, stations, meteo_vars, gan_tag, scen_label
            ))
            ext = extreme_metrics(gan_o, gt_orig, art_mask, meteo_vars)
            ext.update({'method': gan_tag, 'scenario': scen_label})
            records_ext.append(ext)
            # Physical check (raw GAN output)
            p = physical_check_raw(gan_o, meteo_vars, f'{gan_tag}|{scen_label}')
            phys_rows.append(p)
            # Precipitation classification metrics
            records_precip_cls.extend(
                precip_classification_rows(gan_o, gt_orig, art_mask,
                                           meteo_vars, gan_tag, scen_label)
            )
            records_precip_det.extend(
                precip_detailed_rows(gan_o, gt_orig, art_mask,
                                     meteo_vars, gan_tag, scen_label)
            )

        # Baselines
        for bm in baseline_methods:
            if (bm, scen_label) not in all_bl:
                continue
            bl_o = sc.inverse_transform(np.clip(all_bl[(bm, scen_label)].astype(np.float32), 0, 1))
            rows = per_var_metrics(bl_o, gt_orig, art_mask, meteo_vars, train_stds)
            for r in rows:
                r.update({'method': bm, 'scenario': scen_label})
                records_orig.append(r)
            records_seas.extend(seasonal_rmse_rows(
                bl_o, gt_orig, art_mask, months, meteo_vars, bm, scen_label
            ))
            records_dist.extend(distribution_metrics_rows(
                bl_o, gt_orig, art_mask, meteo_vars, bm, scen_label
            ))
            records_station.extend(station_rmse_rows(
                bl_o, gt_orig, art_mask, stations, meteo_vars, bm, scen_label
            ))
            ext = extreme_metrics(bl_o, gt_orig, art_mask, meteo_vars)
            ext.update({'method': bm, 'scenario': scen_label})
            records_ext.append(ext)
            p = physical_check_raw(bl_o, meteo_vars, f'{bm}|{scen_label}')
            phys_rows.append(p)
            # Precipitation classification metrics
            records_precip_cls.extend(
                precip_classification_rows(bl_o, gt_orig, art_mask,
                                           meteo_vars, bm, scen_label)
            )
            records_precip_det.extend(
                precip_detailed_rows(bl_o, gt_orig, art_mask,
                                     meteo_vars, bm, scen_label)
            )

    # ── Save results ─────────────────────────────────────────────────────
    df_orig = pd.DataFrame(records_orig)
    if len(df_orig):
        df_orig.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_results_orig.csv'), index=False)

        # Summary (macro-average over variables)
        summary = (df_orig.groupby(['method', 'scenario'])[['RMSE_orig', 'MAE_orig', 'Std_RMSE']]
                   .mean().round(4).reset_index()
                   .sort_values(['scenario', 'RMSE_orig']))
        print("\n  ── RMSE in original units (macro-average over all variables) ──")
        print(summary.to_string(index=False))

        # Standardized RMSE pivot
        std_piv = (df_orig.groupby(['method', 'scenario'])['Std_RMSE']
                   .mean().round(4).unstack('scenario'))
        print("\n  ── Standardized RMSE (RMSE/train_std, macro-avg) ──")
        print(std_piv.to_string())
        std_piv.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_stdrmse.csv'))

    df_ext = pd.DataFrame(records_ext)
    if len(df_ext):
        df_ext.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_extreme.csv'), index=False)
        print("\n  ── Extreme metrics (subset) ──")
        ecols = ['method', 'scenario'] + [c for c in df_ext.columns
                                           if 'p95' in c or 'wetday' in c]
        ecols = [c for c in ecols if c in df_ext.columns]
        print(df_ext[ecols].sort_values(['scenario', 'method']).to_string(index=False))

    df_phys = pd.DataFrame(phys_rows)
    df_phys.to_csv(os.path.join(OUTPUT_DIR, 'physical_check_v2.csv'), index=False)
    print("\n  ── Physical check — TMIN≤TMEAN≤TMAX (%) ──")
    print(df_phys[['label', 'viol_count', 'viol_pct']].sort_values('viol_pct').to_string(index=False))

    df_seas = pd.DataFrame(records_seas)
    if len(df_seas):
        df_seas.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_rmse.csv'), index=False)

    df_dist = pd.DataFrame(records_dist)
    if len(df_dist):
        df_dist.to_csv(os.path.join(OUTPUT_DIR, 'distribution_metrics.csv'), index=False)

    df_station = pd.DataFrame(records_station)
    if len(df_station):
        df_station.to_csv(os.path.join(OUTPUT_DIR, 'station_rmse.csv'), index=False)

    # ── Precipitation classification report ───────────────────────────────
    df_pc = pd.DataFrame(records_precip_cls)
    if len(df_pc):
        df_pc.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_precip_classification.csv'),
                     index=False)
        print("\n  ── Precipitation wet-day classification (10pct scenario) ──")
        scen_show = '10pct' if '10pct' in df_pc['scenario'].values else df_pc['scenario'].iloc[0]
        sub = (df_pc[df_pc['scenario'] == scen_show]
               [['method', 'freq_gt', 'freq_pred', 'bias', 'precision', 'recall', 'f1',
                 'dry_accuracy', 'csi']]
               .sort_values('bias', key=lambda x: x.abs()))
        print(sub.to_string(index=False))
        print(f"  Precipitation classification saved → evaluation_precip_classification.csv")

    df_pd = pd.DataFrame(records_precip_det)
    if len(df_pd):
        df_pd.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_precip_detailed.csv'), index=False)
        print(f"  Precipitation detailed metrics saved → evaluation_precip_detailed.csv")

    # ── Compact PRECIP comparison table (requested) ──────────────────────────
    try:
        methods_show = [
            'linear',
            'knn',
            'WGAN-GP_modeB_seed42',
            'WGAN-GP_modeB_seed42+PrecipFix',
            'WGAN-GP_modeB_seed42+Precip2Stage',
        ]
        scens_show = [s for s in ['10pct', '20pct', 'block7d', 'block30d'] if s in avail]

        df_p_rmse = df_orig[(df_orig['variable'] == 'PRECIP') & (df_orig['method'].isin(methods_show))].copy()
        df_p_cls  = df_pc[df_pc['method'].isin(methods_show)].copy() if len(df_pc) else pd.DataFrame()
        df_p_det  = df_pd[df_pd['method'].isin(methods_show)].copy() if len(df_pd) else pd.DataFrame()

        if len(df_p_rmse) and len(df_p_cls) and len(df_p_det):
            print("\n  ── PRECIP compact comparison (masked cells; original units) ──")
            for s in scens_show:
                base = (df_p_rmse[df_p_rmse['scenario'] == s][['method', 'RMSE_orig', 'MAE_orig']]
                        .merge(df_p_cls[df_p_cls['scenario'] == s][['method', 'bias', 'f1', 'csi']],
                               on='method', how='left')
                        .merge(df_p_det[df_p_det['scenario'] == s][['method', 'rmse_wet_only', 'mae_wet_only']],
                               on='method', how='left'))
                base = base.set_index('method').reindex(methods_show).reset_index()
                print(f"\n  Scenario: {s}")
                print(base.round(4).to_string(index=False))
        else:
            print("\n  [WARN] PRECIP compact comparison skipped (missing tables).")
    except Exception as e:
        print(f"\n  [WARN] PRECIP compact comparison failed: {e}")

    # Pick primary scenario
    plot_scen = '10pct' if '10pct' in avail else list(avail.keys())[0]
    ck_p, ak_p  = avail[plot_scen]
    art_mask_p  = te_npz[ak_p].astype(np.float32)
    cor_norm_p  = te_npz[ck_p].astype(np.float32)
    cor_orig_p  = inverse_orig(sc, cor_norm_p)

    try:    tmean_idx = meteo_vars.index('TMEAN')
    except: tmean_idx = 0
    try:    tmax_idx  = meteo_vars.index('TMAX')
    except: tmax_idx  = tmean_idx

    N_SHOW = min(200, len(gt_orig))
    t_ax   = np.arange(N_SHOW)

    # ── Fig 1: Time-series overlay ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(13, 4), dpi=150)
    ax1.plot(t_ax, gt_orig[:N_SHOW, tmean_idx],    color='#2C3E50', lw=1.6, label='Ground Truth', zorder=5)
    ax1.plot(t_ax, cor_orig_p[:N_SHOW, tmean_idx], color='#E74C3C', lw=0.8, alpha=0.5,
             label=f'Corrupted ({plot_scen})', zorder=4)

    colors_g = ['#3498DB', '#8E44AD', '#1ABC9C']
    for ci, (tag, garr) in enumerate(gan_imputs.items()):
        go = sc.inverse_transform(np.clip(garr, 0, 1))
        ax1.plot(t_ax, go[:N_SHOW, tmean_idx], color=colors_g[ci % len(colors_g)],
                 lw=1.2, alpha=0.85, label=tag, zorder=6)

    if ('linear', plot_scen) in all_bl:
        bl_lin = sc.inverse_transform(np.clip(all_bl[('linear', plot_scen)].astype(np.float32), 0, 1))
        ax1.plot(t_ax, bl_lin[:N_SHOW, tmean_idx], color='#27AE60', lw=0.9,
                 alpha=0.75, ls='--', label='Linear Interp.', zorder=3)
    if ('knn', plot_scen) in all_bl:
        bl_knn = sc.inverse_transform(np.clip(all_bl[('knn', plot_scen)].astype(np.float32), 0, 1))
        ax1.plot(t_ax, bl_knn[:N_SHOW, tmean_idx], color='#E67E22', lw=0.9,
                 alpha=0.75, ls=':', label='KNN', zorder=3)

    ax1.set_xlabel('Time (days)', fontsize=11)
    ax1.set_ylabel('TMEAN (°C)', fontsize=11)
    ax1.set_title(f'Time Series — TMEAN  [{plot_scen}]', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right', ncol=3)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'fig_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("  fig_timeseries.png ✓")

    # ── Fig 2: Scatter true vs imputed ─────────────────────────────────────
    all_methods_sc = list(gan_imputs.items()) + [
        (bm, all_bl[(bm, plot_scen)].astype(np.float32))
        for bm in ['knn', 'linear']
        if (bm, plot_scen) in all_bl
    ]
    n_sp  = min(4, len(all_methods_sc))
    ncols = 2; nrows = (n_sp + 1) // 2
    fig2, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows), dpi=150)
    axes = np.array(axes).flatten()

    palette = ['#3498DB','#8E44AD','#E67E22','#27AE60']
    sel_all = art_mask_p[:, tmean_idx].astype(bool)

    for ai, (mname, marr) in enumerate(all_methods_sc[:n_sp]):
        ax_s  = axes[ai]
        mo    = sc.inverse_transform(np.clip(marr, 0, 1)) if isinstance(marr, np.ndarray) else marr
        x_gt  = gt_orig[sel_all, tmean_idx]
        y_imp = mo[sel_all, tmean_idx]
        if len(x_gt) == 0:
            ax_s.set_title(f'{mname} — no data')
            continue
        ax_s.scatter(x_gt, y_imp, alpha=0.2, s=5, color=palette[ai % 4], rasterized=True)
        lims = [min(x_gt.min(), y_imp.min()) - 1, max(x_gt.max(), y_imp.max()) + 1]
        ax_s.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        r_v = float(np.sqrt(np.mean((y_imp - x_gt)**2))) if len(x_gt) > 0 else np.nan
        ax_s.set_title(f'{mname}\nRMSE={r_v:.2f}°C', fontsize=10, fontweight='bold')
        ax_s.set_xlabel('True TMEAN (°C)', fontsize=9)
        ax_s.set_ylabel('Imputed TMEAN (°C)', fontsize=9)
        ax_s.grid(True, alpha=0.3)

    for ai in range(n_sp, len(axes)):
        axes[ai].set_visible(False)

    fig2.suptitle(f'Scatter: True vs Imputed TMEAN ({plot_scen})', fontsize=13, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, 'fig_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("  fig_scatter.png ✓")

    # ── Fig 3: RMSE bar chart ──────────────────────────────────────────────
    if len(df_orig):
        bar = (df_orig.groupby(['method', 'scenario'])['RMSE_orig']
               .mean().reset_index().rename(columns={'RMSE_orig': 'mean_RMSE'}))
        scens_ord   = [s for s in ['10pct','20pct','block7d','block30d'] if s in bar.scenario.values]
        # sort methods: baselines first, then GANs
        base_mths   = [m for m in ['mean','linear','knn','mice'] if m in bar.method.values]
        gan_mths    = [m for m in bar.method.values if m not in base_mths]
        methods_ord = base_mths + sorted(set(gan_mths))

        x     = np.arange(len(scens_ord))
        width = 0.8 / max(len(methods_ord), 1)
        cmap  = plt.cm.get_cmap('tab10', len(methods_ord))

        fig3, ax3 = plt.subplots(figsize=(11, 5), dpi=150)
        for i, mth in enumerate(methods_ord):
            vals = [bar[(bar.method == mth) & (bar.scenario == s)]['mean_RMSE'].values
                    for s in scens_ord]
            vals = [v[0] if len(v) > 0 else 0.0 for v in vals]
            ax3.bar(x + i * width - 0.4 + width / 2, vals, width,
                    label=mth, color=cmap(i), alpha=0.85, edgecolor='white', linewidth=0.5)

        ax3.set_xticks(x); ax3.set_xticklabels(scens_ord, fontsize=11)
        ax3.set_xlabel('Missingness Scenario', fontsize=12)
        ax3.set_ylabel('Mean RMSE — original units', fontsize=12)
        ax3.set_title('RMSE Comparison — All Methods & Scenarios\n(macro-average over 7 variables, original units)',
                      fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='upper left', ncol=2)
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        fig3.tight_layout()
        fig3.savefig(os.path.join(OUTPUT_DIR, 'fig_rmse_bar.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print("  fig_rmse_bar.png ✓")

    print("\n" + "=" * 64)
    print("  EVALUATION COMPLETE ")
    print("=" * 64)

if __name__ == '__main__':
    main()

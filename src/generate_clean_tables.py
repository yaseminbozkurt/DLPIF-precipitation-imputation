"""
generate_clean_tables.py
========================
Produces manuscript tables and supporting diagnostic summaries using clean
DLPIF results (no local PRECIP leakage).

Manuscript tables
-----------------
  Table 1 — Benchmark comparison (10% random and 30-day block scenarios)
  Table 2 — Ablation analysis (10% random scenario)

Supporting diagnostic output (not manuscript tables)
----------------------------------------------------
  Diagnostic A — Ablation detail across all scenarios
  Diagnostic B — Extreme-precipitation MAE/RMSE breakdown

Note: all method output files expected under OUTPUT_DIR (src/) use the _msclean_ prefix,
matching the filenames written by multiseed_clean_rerun.py.
"""
import sys, io, os, pickle, warnings
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except: pass
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
WET_THRESH  = 0.1
P95_THRESH  = 16.74

def load_scaler():
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d['scaler'], list(d['meteo_vars'])

def inv(sc, arr):
    a = np.nan_to_num(arr.copy().astype(np.float64), nan=0.0)
    return sc.inverse_transform(a)

def precip_cls(gt_mm, pred_mm):
    pw = pred_mm > WET_THRESH; gw = gt_mm > WET_THRESH
    tp=int((pw&gw).sum()); fp=int((pw&~gw).sum())
    fn=int((~pw&gw).sum()); tn=int((~pw&~gw).sum())
    fg=float(gw.mean()); fp_=float(pw.mean())
    pr=tp/(tp+fp) if (tp+fp)>0 else 0.0
    rc=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*pr*rc/(pr+rc) if (pr+rc)>0 else 0.0
    csi=tp/(tp+fp+fn) if (tp+fp+fn)>0 else 0.0
    ws=gw
    rw=float(np.sqrt(np.mean((pred_mm[ws]-gt_mm[ws])**2))) if ws.sum() else np.nan
    mw=float(np.mean(np.abs(pred_mm[ws]-gt_mm[ws]))) if ws.sum() else np.nan
    return dict(freq_gt=round(fg,4),freq_pred=round(fp_,4),bias=round(fp_-fg,4),
                precision=round(pr,4),recall=round(rc,4),f1=round(f1,4),
                csi=round(csi,4),rmse_wet=round(rw,4) if rw==rw else np.nan,
                mae_wet=round(mw,4) if mw==mw else np.nan,
                tp=tp,fp=fp,fn=fn,tn=tn)

def extreme_metrics(gt_mm, pred_mm):
    sel = gt_mm >= P95_THRESH
    if sel.sum() == 0: return dict(mae_p95=np.nan, rmse_p95=np.nan)
    return dict(
        mae_p95  = round(float(np.mean(np.abs(pred_mm[sel]-gt_mm[sel]))),2),
        rmse_p95 = round(float(np.sqrt(np.mean((pred_mm[sel]-gt_mm[sel])**2))),2)
    )

def load_npy_precip_mm(sc, path, pidx):
    arr = np.load(path).astype(np.float32)
    return np.clip(arr[:,pidx],0,1)*sc.data_range_[pidx]+sc.data_min_[pidx]

def main():
    sc, mv = load_scaler()
    pidx = mv.index('PRECIP')

    te = np.load(os.path.join(OUTPUT_DIR,'preprocessed_test.npz'), allow_pickle=True)
    gt_mm_full = inv(sc, te['data'].astype(np.float32))[:,pidx]

    SCENARIOS = [
        ('10pct',   'art_mask_10pct'),
        ('20pct',   'art_mask_20pct'),
        ('block7d', 'art_mask_block7d'),
        ('block30d','art_mask_block30d'),
    ]
    SCEN_LABELS = ['10pct','20pct','block7d','block30d']

    # ── Discover method files ─────────────────────────────────────────────────
    # Map method-label → npy path (PRECIP column)
    METHODS = {}

    # Raw GAN seeds
    for seed in [42,123,456]:
        p = os.path.join(OUTPUT_DIR, f'gan_imputed_test_modeB_seed{seed}.npy')
        if os.path.exists(p):
            METHODS[f'WGAN-GP_raw_seed{seed}'] = p

    # PrecipFix (old)
    for seed in [42,123,456]:
        p = os.path.join(OUTPUT_DIR, f'gan_imputed_test_modeB_seed{seed}_precipfix.npy')
        if os.path.exists(p):
            METHODS[f'PrecipFix_seed{seed}'] = p

    # Clean Precip2Stage  (produced by multiseed_clean_rerun.py with _msclean_ prefix)
    for seed in [42,123,456]:
        p = os.path.join(OUTPUT_DIR, f'gan_imputed_test_modeB_seed{seed}_msclean_precip2stage.npy')
        if os.path.exists(p):
            METHODS[f'Precip2Stage_clean_seed{seed}'] = p

    # Clean AmountRF / DLPIF  (produced by multiseed_clean_rerun.py with _msclean_ prefix)
    for seed in [42,123,456]:
        p = os.path.join(OUTPUT_DIR, f'gan_imputed_test_modeB_seed{seed}_msclean_amountrf.npy')
        if os.path.exists(p):
            METHODS[f'AmountRF_clean_seed{seed}'] = p

    print(f'  Methods found: {list(METHODS.keys())}')

    # ── Load baselines ────────────────────────────────────────────────────────
    bl_path = os.path.join(OUTPUT_DIR, 'baseline_results.pkl')
    BL = {}
    if os.path.exists(bl_path):
        with open(bl_path,'rb') as f:
            bd = pickle.load(f)
        BL = bd.get('all_scenarios', {})

    # ── Evaluate ──────────────────────────────────────────────────────────────
    records = []

    for scen_label, mask_key in SCENARIOS:
        if mask_key not in te.files: continue
        m = te[mask_key].astype(np.float32)[:,pidx] > 0.5
        if not m.sum(): continue
        gt_m = gt_mm_full[m]
        n_m  = int(m.sum())

        # Baselines
        for bm in ['mean','linear','knn','mice']:
            if (bm, scen_label) not in BL: continue
            pred_norm = np.clip(BL[(bm,scen_label)].astype(np.float32), 0, 1)
            pred_o = sc.inverse_transform(pred_norm)
            pred_mm = pred_o[:,pidx][m]
            row = precip_cls(gt_m, pred_mm)
            row.update(extreme_metrics(gt_m, pred_mm))
            row.update(method=bm, scenario=scen_label, n_masked=n_m)
            records.append(row)

        # GAN-based methods
        for mname, npy_path in METHODS.items():
            pred_mm = load_npy_precip_mm(sc, npy_path, pidx)[m]
            row = precip_cls(gt_m, pred_mm)
            row.update(extreme_metrics(gt_m, pred_mm))
            row.update(method=mname, scenario=scen_label, n_masked=n_m)
            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, 'clean_full_evaluation.csv'), index=False)

    # ── Aggregate: mean over seeds for DLPIF methods ──────────────────────────
    def agg(df, pattern):
        sub = df[df['method'].str.contains(pattern)]
        if not len(sub): return None
        return sub.groupby('scenario')[['bias','f1','csi','precision','recall',
                                        'rmse_wet','mae_wet','mae_p95','rmse_p95']].mean()

    raw_agg  = agg(df,'WGAN-GP_raw')
    fix_agg  = agg(df,'PrecipFix')
    p2s_agg  = agg(df,'Precip2Stage_clean')
    amt_agg  = agg(df,'AmountRF_clean')

    # Seed std for DLPIF
    def std_col(df, pattern, col):
        sub = df[df['method'].str.contains(pattern)]
        if not len(sub): return {}
        return sub.groupby('scenario')[col].std().round(4).to_dict()

    p2s_std = std_col(df,'Precip2Stage_clean','f1')
    amt_std = std_col(df,'AmountRF_clean','f1')

    bl_agg = df[df['method'].isin(['mean','linear','knn','mice'])]

    # ── TABLE 1: Bias and CSI ─────────────────────────────────────────────────
    print('\n' + '='*80)
    print('  TABLE 1 — Wet-day frequency bias and CSI (CLEAN pipeline)')
    print('='*80)
    print(f"  {'Method':<28}", end='')
    for s in SCEN_LABELS:
        print(f'  Bias({s[:4]})  CSI({s[:4]})', end='')
    print()
    print('  ' + '-'*100)

    def fmt_row(label, df_agg, is_series=False):
        line = f'  {label:<28}'
        for s in SCEN_LABELS:
            if is_series:
                row = df_agg[df_agg['scenario']==s]
                b = row['bias'].values[0] if len(row) else np.nan
                c = row['csi'].values[0] if len(row) else np.nan
            else:
                if df_agg is None or s not in df_agg.index:
                    b,c = np.nan,np.nan
                else:
                    b = df_agg.loc[s,'bias']; c = df_agg.loc[s,'csi']
            line += f'  {b:+.3f}    {c:.3f} '
        print(line)

    for bm in ['mean','linear','knn','mice']:
        fmt_row(bm, bl_agg[bl_agg['method']==bm], is_series=True)
    fmt_row('WGAN-GP (raw)',    raw_agg)
    fmt_row('+PrecipFix',       fix_agg)
    fmt_row('+Precip2Stage (clean)', p2s_agg)
    fmt_row('+AmountRF/DLPIF (clean)', amt_agg)

    # ── TABLE 2: F1 ──────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('  TABLE 2 — Wet-day classification F1 (CLEAN pipeline)')
    print('='*80)
    print(f"  {'Method':<28}  F1(10%)  F1(20%)  F1(7d)   F1(30d)")
    print('  ' + '-'*65)

    def fmt_f1(label, df_agg, std_d=None, is_series=False):
        line = f'  {label:<28}'
        for s in SCEN_LABELS:
            if is_series:
                row = df_agg[df_agg['scenario']==s]
                v = row['f1'].values[0] if len(row) else np.nan
            else:
                v = df_agg.loc[s,'f1'] if (df_agg is not None and s in df_agg.index) else np.nan
            sd = std_d.get(s, np.nan) if std_d else np.nan
            if sd == sd and std_d:
                line += f'  {v:.4f}±{sd:.3f}'
            else:
                line += f'  {v:.4f}      '
        print(line)

    for bm in ['mean','linear','knn','mice']:
        fmt_f1(bm, bl_agg[bl_agg['method']==bm], is_series=True)
    fmt_f1('WGAN-GP (raw)',          raw_agg)
    fmt_f1('+PrecipFix',             fix_agg)
    fmt_f1('+Precip2Stage (clean)',  p2s_agg, p2s_std)
    fmt_f1('+AmountRF/DLPIF (clean)',amt_agg, amt_std)

    # ── DIAGNOSTIC SUMMARY A: Ablation detail (not a manuscript table) ───────────────────
    print('\n' + '='*80)
    print('  DIAGNOSTIC SUMMARY A — Ablation detail (CLEAN, 10pct and 30d scenarios)')
    print('  (Supporting output — not a manuscript table)')
    print('='*80)
    print(f"  {'Stage':<28}  F1(10%)  F1(30d)  Bias(10%)  RMSE_wet(10%)")
    print('  ' + '-'*70)

    def abl_row(label, df_agg, is_series=False):
        def get(agg, scen, col):
            if agg is None: return np.nan
            if is_series:
                row = agg[agg['scenario']==scen]
                return row[col].values[0] if len(row) else np.nan
            return agg.loc[scen,col] if scen in agg.index else np.nan
        f1_10 = get(df_agg,'10pct','f1')
        f1_30 = get(df_agg,'block30d','f1')
        bi_10 = get(df_agg,'10pct','bias')
        rw_10 = get(df_agg,'10pct','rmse_wet')
        print(f'  {label:<28}  {f1_10:.4f}   {f1_30:.4f}   {bi_10:+.4f}    {rw_10:.2f}')

    abl_row('WGAN-GP (raw)',          raw_agg)
    abl_row('+PrecipFix',             fix_agg)
    abl_row('+Precip2Stage (clean)',  p2s_agg)
    abl_row('+AmountRF/DLPIF (clean)',amt_agg)

    # ── DIAGNOSTIC SUMMARY B: Extreme-precipitation metrics (not a manuscript table) ─────────
    print('\n' + '='*80)
    print('  DIAGNOSTIC SUMMARY B — Extreme precipitation MAE/RMSE (p95>=16.74mm, CLEAN)')
    print('  (Supporting output — not a manuscript table)')
    print('='*80)
    print(f"  {'Method':<28}", end='')
    for s in SCEN_LABELS:
        print(f'  MAE({s[:4]}) RMSE({s[:4]})', end='')
    print()
    print('  ' + '-'*100)

    def fmt_ext(label, df_agg, is_series=False):
        line = f'  {label:<28}'
        for s in SCEN_LABELS:
            if is_series:
                row = df_agg[df_agg['scenario']==s]
                ma = row['mae_p95'].values[0] if len(row) else np.nan
                rm = row['rmse_p95'].values[0] if len(row) else np.nan
            else:
                if df_agg is None or s not in df_agg.index:
                    ma,rm = np.nan,np.nan
                else:
                    ma=df_agg.loc[s,'mae_p95']; rm=df_agg.loc[s,'rmse_p95']
            line += f'  {ma:5.2f}   {rm:5.2f} '
        print(line)

    for bm in ['mean','linear','knn','mice']:
        fmt_ext(bm, bl_agg[bl_agg['method']==bm], is_series=True)
    fmt_ext('WGAN-GP (raw)',          raw_agg)
    fmt_ext('+PrecipFix',             fix_agg)
    fmt_ext('+Precip2Stage (clean)',  p2s_agg)
    fmt_ext('+AmountRF/DLPIF (clean)',amt_agg)

    # ── OLD vs CLEAN DELTA ────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('  OLD vs CLEAN DELTA (seed42)')
    print('='*80)
    old_cls = os.path.join(OUTPUT_DIR,'evaluation_precip_classification.csv')
    if os.path.exists(old_cls):
        df_old = pd.read_csv(old_cls)
        print(f"  {'Scenario':<10} {'Method':<20} {'Old_F1':>8} {'New_F1':>8}"
              f" {'Delta':>8} {'Old_Bias':>10} {'New_Bias':>10}")
        print('  '+'-'*80)
        pairs = [
            ('WGAN-GP_modeB_seed42+Precip2Stage','Precip2Stage_clean_seed42'),
            ('WGAN-GP_modeB_seed42+AmountRF',    'AmountRF_clean_seed42'),
        ]
        for scen_label,_ in SCENARIOS:
            for old_m, new_m in pairs:
                old_r = df_old[(df_old['method']==old_m)&(df_old['scenario']==scen_label)]
                new_r = df[(df['method']==new_m)&(df['scenario']==scen_label)]
                if not len(old_r) or not len(new_r): continue
                of = old_r['f1'].values[0]; nf = new_r['f1'].values[0]
                ob = old_r['bias'].values[0]; nb = new_r['bias'].values[0]
                short = old_m.replace('WGAN-GP_modeB_seed42+','')
                print(f'  {scen_label:<10} {short:<20} {of:8.4f} {nf:8.4f}'
                      f' {nf-of:+8.4f} {ob:+10.4f} {nb:+10.4f}')

    print('\n  All outputs -> clean_full_evaluation.csv')
    print('='*80)

if __name__ == '__main__':
    main()

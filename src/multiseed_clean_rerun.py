"""
multiseed_clean_rerun.py
========================
Trains INDEPENDENT clean occurrence classifiers per seed (42, 123, 456).
Local PRECIP excluded from Stage 1. Val-only threshold. No freq-matching.
Produces publication-quality mean +/- std across seeds.
"""
import sys, io, os, pickle, json, warnings
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except: pass
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, precision_score, recall_score

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
WET_THRESH  = 0.1
P95_THRESH  = 16.74
SEEDS       = [42, 123, 456]
COR_KEY     = 'corrupted_10pct'

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scaler():
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d['scaler'], list(d['meteo_vars'])

def load_npz(split):
    return np.load(os.path.join(OUTPUT_DIR, f'preprocessed_{split}.npz'), allow_pickle=True)

def inv(sc, arr):
    a = np.nan_to_num(arr.copy().astype(np.float64), nan=0.0)
    return sc.inverse_transform(a)

def to_norm(sc, mm, pidx):
    v = np.maximum(mm.astype(np.float64), 0.0)
    return np.clip((v - sc.data_min_[pidx]) / sc.data_range_[pidx], 0.0, 1.0).astype(np.float32)

def build_occ_X(cor, tmp, na, nm, pidx):
    """25-feature matrix — local PRECIP dropped from corrupted block."""
    cols = [i for i in range(cor.shape[1]) if i != pidx]
    parts = [
        np.nan_to_num(cor[:, cols], nan=0.0).astype(np.float32),
        np.nan_to_num(tmp, nan=0.0).astype(np.float32),
        np.nan_to_num(na,  nan=0.0).astype(np.float32),
        np.nan_to_num(nm,  nan=0.0).astype(np.float32),
    ]
    X = np.concatenate(parts, axis=1)
    assert X.shape[1] == 25
    return X

def build_amt_X(sc, z, cor_key, pidx):
    """26-feature matrix — local PRECIP hard-zeroed (Stage 2, unchanged)."""
    cor = inv(sc, z[cor_key].astype(np.float32)).astype(np.float32)
    cor[:, pidx] = 0.0
    return np.concatenate([cor,
                           z['temporal'].astype(np.float32),
                           inv(sc, z['neighbor_avg'].astype(np.float32)).astype(np.float32),
                           z['neighbor_mask'].astype(np.float32)], axis=1)

def apply_qmap(wet_pred_mm, qmap):
    if qmap is None or len(wet_pred_mm) == 0: return wet_pred_mm
    pq = np.argsort(np.argsort(wet_pred_mm)).astype(np.float64)
    pq /= max(len(wet_pred_mm) - 1, 1)
    return np.interp(pq, np.linspace(0, 1, len(qmap)), qmap).astype(np.float32)

def metrics(gt_mm, pred_mm, method, scenario, n_masked):
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
    sel95 = gt_mm >= P95_THRESH
    ma95 = float(np.mean(np.abs(pred_mm[sel95]-gt_mm[sel95]))) if sel95.sum() else np.nan
    rm95 = float(np.sqrt(np.mean((pred_mm[sel95]-gt_mm[sel95])**2))) if sel95.sum() else np.nan
    return dict(method=method, scenario=scenario, n_masked=n_masked,
                freq_gt=round(fg,4), freq_pred=round(fp_,4),
                bias=round(fp_-fg,4),
                precision=round(pr,4), recall=round(rc,4),
                f1=round(f1,4), csi=round(csi,4),
                rmse_wet=round(rw,4) if rw==rw else np.nan,
                mae_wet=round(mw,4) if mw==mw else np.nan,
                mae_p95=round(ma95,2) if ma95==ma95 else np.nan,
                rmse_p95=round(rm95,2) if rm95==rm95 else np.nan,
                tp=tp, fp=fp, fn=fn, tn=tn)

def train_occ(X_tr, y_tr, X_va, y_va, seed):
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    va_p = rf.predict_proba(X_va)[:,1]
    best_f1, best_cut = -1.0, 0.5
    for cut in np.arange(0.20, 0.82, 0.02):
        f = f1_score(y_va, (va_p>=cut).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_cut = f, float(cut)
    cut = round(best_cut, 3)
    yp  = (va_p >= cut).astype(int)
    vm  = dict(cutoff=cut,
               val_f1    =round(float(f1_score(y_va, yp, zero_division=0)),4),
               val_prec  =round(float(precision_score(y_va, yp, zero_division=0)),4),
               val_rec   =round(float(recall_score(y_va, yp, zero_division=0)),4),
               val_bias  =round(float(yp.mean()-y_va.mean()),4),
               wet_frac_gt=round(float(y_va.mean()),4))
    return rf, cut, vm

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print('='*70)
    print('  MULTISEED CLEAN RERUN — INDEPENDENT OCCURRENCE MODELS PER SEED')
    print('='*70)

    sc, mv = load_scaler()
    pidx = mv.index('PRECIP')

    tr = load_npz('train'); va = load_npz('val'); te = load_npz('test')

    # Pre-build static feature matrices
    X_tr_full = build_occ_X(tr[COR_KEY].astype(np.float32),
                            tr['temporal'].astype(np.float32),
                            tr['neighbor_avg'].astype(np.float32),
                            tr['neighbor_mask'].astype(np.float32), pidx)
    X_va_full = build_occ_X(va[COR_KEY].astype(np.float32),
                            va['temporal'].astype(np.float32),
                            va['neighbor_avg'].astype(np.float32),
                            va['neighbor_mask'].astype(np.float32), pidx)
    X_te_full = build_occ_X(te[COR_KEY].astype(np.float32),
                            te['temporal'].astype(np.float32),
                            te['neighbor_avg'].astype(np.float32),
                            te['neighbor_mask'].astype(np.float32), pidx)
    X_te_amt  = build_amt_X(sc, te, COR_KEY, pidx)
    X_tr_amt  = build_amt_X(sc, tr, COR_KEY, pidx)

    tr_gt = inv(sc, tr['data'].astype(np.float32))[:,pidx]
    va_gt = inv(sc, va['data'].astype(np.float32))[:,pidx]
    te_gt = inv(sc, te['data'].astype(np.float32))[:,pidx]

    tr_obs = tr['real_mask'].astype(np.float32)[:,pidx].astype(bool)
    va_obs = va['real_mask'].astype(np.float32)[:,pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs

    X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs]>WET_THRESH).astype(int)
    X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs]>WET_THRESH).astype(int)

    va_wet_mm = va_gt[va_gt > WET_THRESH]
    qmap = np.quantile(va_wet_mm, np.linspace(0,1,201)).astype(np.float32) if len(va_wet_mm)>=10 else None

    SCENARIOS = [('10pct','art_mask_10pct'),('20pct','art_mask_20pct'),
                 ('block7d','art_mask_block7d'),('block30d','art_mask_block30d')]

    all_records = []
    seed_meta   = []

    print(f'\n  Feature dim={X_tr_occ.shape[1]}  Train obs={len(X_tr_occ):,}  Val obs={len(X_va_occ):,}')

    for seed in SEEDS:
        print(f'\n{"="*60}')
        print(f'  SEED = {seed}')
        print('='*60)

        # ── Train seed-specific occurrence RF ─────────────────────────────────
        occ_rf, occ_cut, vm = train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed)
        print(f'  Occ RF  cutoff={occ_cut}  val_F1={vm["val_f1"]}  '
              f'P={vm["val_prec"]}  R={vm["val_rec"]}  bias={vm["val_bias"]:+.4f}')
        vm.update(seed=seed)
        seed_meta.append(vm)

        # Save seed-specific model
        pfx = os.path.join(OUTPUT_DIR, f'precip_occurrence_clean_seed{seed}')
        with open(pfx+'.pkl','wb') as f:
            pickle.dump({'rf':occ_rf,'scaler':sc}, f)
        vm_copy = {k: v for k, v in vm.items() if k != 'seed'}
        meta_out = dict(seed=seed, n_estimators=300, min_samples_leaf=5,
                        n_features=25, local_precip_excluded=True,
                        neighbor_precip_included=True,
                        threshold_strategy='val_F1_maximization',
                        feature_blocks=['corrupted_no_precip(6)','temporal(5)',
                                        'neighbor_avg(7)','neighbor_mask(7)'],
                        **vm_copy)
        with open(pfx+'.json','w',encoding='utf-8') as f:
            json.dump(meta_out, f, indent=2)

        # ── Test occurrence predictions ───────────────────────────────────────
        te_proba   = occ_rf.predict_proba(X_te_full)[:,1]
        te_wet_pred = (te_proba >= occ_cut).astype(bool)
        print(f'  Test: wet_pred={te_wet_pred.mean():.4f}  gt_wet={(te_gt>WET_THRESH).mean():.4f}')

        # ── Load raw GAN base imputation (produced by Step 2: 02_wgan_gp_imputation.py) ──────
        # Note: the legacy name _precip2stage.npy has been replaced with the canonical raw GAN
        # output gan_imputed_test_modeB_seed{seed}.npy.  The Precip2Stage output variant is
        # constructed below inside this script (zeroing dry-predicted positions), not externally.
        p2_path = os.path.join(OUTPUT_DIR, f'gan_imputed_test_modeB_seed{seed}.npy')
        if not os.path.exists(p2_path):
            print(f'  [SKIP] raw GAN base npy not found for seed {seed}')
            print(f'         Expected: {p2_path}')
            print(f'         Produce it by running: python src/02_wgan_gp_imputation.py --seed {seed} --mode B')
            continue
        p2_norm = np.load(p2_path).astype(np.float32)
        p2_mm   = np.clip(p2_norm[:,pidx],0,1)*sc.data_range_[pidx]+sc.data_min_[pidx]

        # ── Build Precip2Stage clean ──────────────────────────────────────────
        imp_p2s = p2_mm.copy()
        imp_p2s[~te_wet_pred] = 0.0
        if qmap is not None and te_wet_pred.sum()>0:
            raw_wet = np.clip(imp_p2s[te_wet_pred],0,None)
            imp_p2s[te_wet_pred] = apply_qmap(raw_wet, qmap)
        imp_p2s = np.clip(imp_p2s, 0, None)

        p2s_full = p2_norm.copy()
        p2s_full[:,pidx] = to_norm(sc, imp_p2s, pidx)
        np.save(os.path.join(OUTPUT_DIR,
            f'gan_imputed_test_modeB_seed{seed}_msclean_precip2stage.npy'),
            p2s_full.astype(np.float32))

        # ── Train seed-specific AmountRF ──────────────────────────────────────
        amt_rf = RandomForestRegressor(n_estimators=400, random_state=seed,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

        # ── Build AmountRF clean (scenario-aware) ────────────────────────────
        amt_mm = imp_p2s.copy()
        for scen_label, mask_key in SCENARIOS:
            if mask_key not in te.files: continue
            m = te[mask_key].astype(np.float32)[:,pidx] > 0.5
            apply_sel = m & te_wet_pred
            if apply_sel.sum() > 0:
                amt_mm[apply_sel] = np.maximum(amt_rf.predict(X_te_amt[apply_sel]), 0.0)
            amt_mm[m & ~te_wet_pred] = 0.0

        amt_full = p2_norm.copy()
        amt_full[:,pidx] = to_norm(sc, amt_mm, pidx)
        np.save(os.path.join(OUTPUT_DIR,
            f'gan_imputed_test_modeB_seed{seed}_msclean_amountrf.npy'),
            amt_full.astype(np.float32))

        # ── Evaluate per scenario ─────────────────────────────────────────────
        for scen_label, mask_key in SCENARIOS:
            if mask_key not in te.files: continue
            m   = te[mask_key].astype(np.float32)[:,pidx] > 0.5
            gt_m = te_gt[m]; n_m = int(m.sum())

            r1 = metrics(gt_m, imp_p2s[m], f'Precip2Stage_clean_seed{seed}', scen_label, n_m)
            r2 = metrics(gt_m, amt_mm[m],  f'AmountRF_clean_seed{seed}',     scen_label, n_m)
            all_records += [r1, r2]
            print(f'  [{scen_label}] P2S F1={r1["f1"]:.4f} bias={r1["bias"]:+.4f}'
                  f'  AMT F1={r2["f1"]:.4f} RMSE_wet={r2["rmse_wet"]} MAE_p95={r2["mae_p95"]}')

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(OUTPUT_DIR,'multiseed_clean_evaluation.csv'), index=False)

    pd.DataFrame(seed_meta).to_csv(
        os.path.join(OUTPUT_DIR,'occurrence_clean_seed_summary.csv'), index=False)

    # ── Aggregate: mean +/- std across seeds ─────────────────────────────────
    SCENS = ['10pct','20pct','block7d','block30d']
    cols  = ['bias','f1','csi','precision','recall','rmse_wet','mae_wet','mae_p95','rmse_p95']

    print('\n' + '='*70)
    print('  SEED TRAINING SUMMARY')
    print('='*70)
    print(f'  {"Seed":>6} {"Cutoff":>8} {"Val_F1":>8} {"Val_P":>7} {"Val_R":>7} {"Val_bias":>10}')
    for m in seed_meta:
        print(f'  {m["seed"]:>6} {m["cutoff"]:>8.3f} {m["val_f1"]:>8.4f}'
              f' {m["val_prec"]:>7.4f} {m["val_rec"]:>7.4f} {m["val_bias"]:>+10.4f}')

    def agg(pat, col):
        sub = df[df['method'].str.contains(pat)]
        g = sub.groupby('scenario')[col]
        return g.mean().round(4), g.std().round(4)

    print('\n' + '='*70)
    print('  TABLE 2 — F1 (mean +/- std across seeds 42, 123, 456)')
    print('='*70)
    p2s_f1_m, p2s_f1_s = agg('Precip2Stage_clean','f1')
    amt_f1_m, amt_f1_s = agg('AmountRF_clean','f1')
    print(f'  {"Method":<32}' + ''.join(f'  {s:<14}' for s in SCENS))
    print('  '+'-'*85)
    for pat, lbl, fm, fs in [
        ('Precip2Stage_clean', '+Precip2Stage (clean)', p2s_f1_m, p2s_f1_s),
        ('AmountRF_clean',     '+AmountRF/DLPIF (clean)', amt_f1_m, amt_f1_s),
    ]:
        row = f'  {lbl:<32}'
        for s in SCENS:
            mu = fm.get(s, np.nan); sd = fs.get(s, np.nan)
            row += f'  {mu:.4f}+/-{sd:.4f}'
        print(row)

    print('\n' + '='*70)
    print('  TABLE 1 — Bias / CSI (mean +/- std)')
    print('='*70)
    for col, cname in [('bias','Bias'), ('csi','CSI')]:
        p2s_m, p2s_s = agg('Precip2Stage_clean', col)
        amt_m, amt_s = agg('AmountRF_clean', col)
        print(f'\n  {cname}:')
        for lbl, fm, fs in [
            ('+Precip2Stage (clean)', p2s_m, p2s_s),
            ('+AmountRF/DLPIF (clean)', amt_m, amt_s),
        ]:
            row = f'  {lbl:<32}'
            for s in SCENS:
                mu = fm.get(s, np.nan); sd = fs.get(s, np.nan)
                row += f'  {mu:+.4f}+/-{sd:.4f}'
            print(row)

    print('\n' + '='*70)
    print('  DIAGNOSTIC — Ablation summary (mean +/- std, 10pct and block30d)')
    print('='*70)
    print(f'  {"Stage":<32} {"F1_10%":>12} {"F1_30d":>12}'
          f' {"Bias_10%":>12} {"RMSE_wet_10%":>14} {"MAE_p95_10%":>13}')
    print('  '+'-'*90)
    for pat, lbl in [('Precip2Stage_clean','+Precip2Stage (clean)'),
                     ('AmountRF_clean','+AmountRF/DLPIF (clean)')]:
        f1m,f1s = agg(pat,'f1'); bim,bis=agg(pat,'bias')
        rwm,rws = agg(pat,'rmse_wet'); mam,mas=agg(pat,'mae_p95')
        def gs(d,s): return d.get(s,np.nan)
        print(f'  {lbl:<32}'
              f' {gs(f1m,"10pct"):>6.4f}+/-{gs(f1s,"10pct"):.4f}'
              f' {gs(f1m,"block30d"):>6.4f}+/-{gs(f1s,"block30d"):.4f}'
              f' {gs(bim,"10pct"):>+7.4f}+/-{gs(bis,"10pct"):.4f}'
              f' {gs(rwm,"10pct"):>8.2f}+/-{gs(rws,"10pct"):.2f}'
              f' {gs(mam,"10pct"):>7.2f}+/-{gs(mas,"10pct"):.2f}')

    # ── Robustness verdict ────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('  ROBUSTNESS VERDICT')
    print('='*70)
    for pat, lbl in [('Precip2Stage_clean','Precip2Stage'),
                     ('AmountRF_clean','AmountRF/DLPIF')]:
        f1m,f1s = agg(pat,'f1')
        max_std = max(f1s.values)
        print(f'  {lbl}: max_seed_std(F1)={max_std:.4f}  '
              + ('STABLE (<0.01)' if max_std<0.01 else
                 'MARGINAL (0.01-0.03)' if max_std<0.03 else 'UNSTABLE (>0.03)'))

    print('\n  Outputs:')
    print('  -> multiseed_clean_evaluation.csv')
    print('  -> occurrence_clean_seed_summary.csv')
    print('  -> precip_occurrence_clean_seed{42,123,456}.pkl/.json')
    print('='*70)

if __name__ == '__main__':
    main()

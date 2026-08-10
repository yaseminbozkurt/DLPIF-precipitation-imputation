"""
direct_two_stage_rf.py
=======================
Backbone-free two-stage Random Forest baseline (classify-then-regress),
matching the design used in prior precipitation-imputation literature
(Chivers et al. 2020; Han et al. 2023; Boukdire et al. 2025): a wet/dry
occurrence classifier followed by a conditional wet-day amount regressor,
trained and applied DIRECTLY on the raw corrupted (gapped) meteorological
records -- with no continuous multivariate base imputer (WGAN-GP / SAITS)
completing the scene first.

Relationship to DLPIF (multiseed_clean_rerun.py)
-------------------------------------------------
DLPIF is a *post-imputation correction layer* that sits on top of a
continuous backbone imputer (manuscript Section 3.2): a WGAN-GP (or SAITS)
model first reconstructs the full multivariate state, and only then does
the occurrence/amount correction run. This script removes that backbone
entirely -- it never reads a `gan_imputed_test_*.npy` file. Every feature
is built straight from the zero-filled raw corrupted array, exactly the
"standalone imputer trained directly on raw gapped records" design that
Section 2.3 of the manuscript attributes to Chivers/Han/Boukdire, as
opposed to DLPIF's own "post-imputation correction layer" framing.

Everything else is held identical to multiseed_clean_rerun.py on purpose,
so any performance gap between the two scripts isolates the contribution
of the backbone-provided multivariate context rather than a difference in
RF hyperparameters, feature layout, leakage guards, training scenario, or
threshold-selection protocol:

  - Stage 1 (occurrence): RandomForestClassifier, 300 trees, min_leaf=5,
    class_weight='balanced'; local PRECIP excluded from the 25-feature
    matrix; threshold chosen by maximising F1 on the validation set only
    (grid 0.20-0.80, step 0.02).
  - Stage 2 (amount): RandomForestRegressor, 400 trees, min_leaf=2;
    26-feature matrix with local PRECIP hard-zeroed; trained on observed
    wet-day precipitation only; applied only where Stage 1 predicts wet.
  - Trained once on the 10% random-missingness partition of the training
    set, applied without retraining to all four test scenarios.
  - Same metrics: bias, precision/recall/F1/CSI, RMSE/MAE on wet-day
    ground truth, RMSE/MAE on extreme (>=16.74 mm) ground truth.

Because this baseline never touches a backbone, it only ever reconstructs
PRECIP -- the six other meteorological variables are left zero-filled in
any saved reconstruction array (they are not evaluated; the manuscript's
comparison, like the cited classify-then-regress literature, is
PRECIP-only).

Outputs
-------
  results/direct_two_stage_rf_evaluation.csv    -- per-seed x scenario metrics
                                                     (same schema as
                                                     multiseed_clean_evaluation.csv)
  results/direct_two_stage_rf_seed_summary.csv  -- per-seed classifier metadata
                                                     (same schema as
                                                     occurrence_clean_seed_summary.csv)
  src/direct_two_stage_rf_test_seed{s}_{scenario}.npy
                                                  -- reconstructed normalised
                                                     (N,7) array, PRECIP column
                                                     only meaningful
"""
import sys, io, os, pickle, json, warnings
try:
    # Guard against double-wrapping: this module is also imported (not just
    # run directly) by check_d2s_dlpif_equivalence.py and
    # canonical_metrics.py. Re-wrapping an already-UTF-8 stdout orphans the
    # previous TextIOWrapper's unflushed buffer, silently dropping every
    # line the importing script printed before the `import` statement.
    if getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, precision_score, recall_score

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

WET_THRESH  = 0.1
P95_THRESH  = 19.2  # train+val p95, post-2005 dataset -- see canonical_metrics.py / compute_p95_threshold.py
SEEDS       = [
    int(s.strip())
    for s in os.environ.get('DLPIF_SEEDS', '42,123,456').split(',')
    if s.strip()
]
TRAIN_SCENARIO = '10pct'
TRAIN_COR_KEY  = 'corrupted_10pct'
SCENARIOS = [
    ('10pct',       'corrupted_10pct',       'art_mask_10pct'),
    ('20pct',       'corrupted_20pct',       'art_mask_20pct'),
    ('block7d',     'corrupted_block7d',     'art_mask_block7d'),
    ('block30d',    'corrupted_block30d',    'art_mask_block30d'),
    # Network-wide, simultaneous, multivariate 30-day block missingness (all
    # 4 stations x all 7 variables masked together) -- see
    # build_network_block_scenario.py. Unlike every scenario above (each
    # (station, variable) pair masked independently), this eliminates the
    # cross-station neighbour context at the same time as the local context.
    ('netblock30d', 'corrupted_netblock30d', 'art_mask_netblock30d'),
    # PRECIP-only missingness-mechanism scenarios -- same 10% PRECIP
    # missing-cell budget as '10pct', but the cells are selected by a
    # MAR/MNAR mechanism instead of uniformly at random. Non-PRECIP columns
    # are identical to '10pct'. See build_mar_mnar_scenarios.py.
    ('mar_meteo',               'corrupted_mar_meteo',               'art_mask_mar_meteo'),
    ('mnar_wet',                'corrupted_mnar_wet',                'art_mask_mnar_wet'),
    ('mnar_intensity_moderate', 'corrupted_mnar_intensity_moderate', 'art_mask_mnar_intensity_moderate'),
    ('mnar_intensity_severe',   'corrupted_mnar_intensity_severe',   'art_mask_mnar_intensity_severe'),
]


# ── I/O helpers (mirrors multiseed_clean_rerun.py) ──────────────────────────

def load_scaler():
    with open(os.path.join(SRC_DIR, 'scaler.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d['scaler'], list(d['meteo_vars'])

def load_npz(split):
    return np.load(os.path.join(SRC_DIR, f'preprocessed_{split}.npz'), allow_pickle=True)

def neighbor_keys(scenario):
    return f'neighbor_avg_{scenario}', f'neighbor_mask_{scenario}'

def require_keys(z, keys, context):
    missing = [k for k in keys if k not in z.files]
    if missing:
        raise KeyError(f'{context}: missing required keys {missing}')

def inv(sc, arr):
    a = np.nan_to_num(arr.copy().astype(np.float64), nan=0.0)
    return sc.inverse_transform(a)

def to_norm(sc, mm, pidx):
    v = np.maximum(mm.astype(np.float64), 0.0)
    return np.clip((v - sc.data_min_[pidx]) / sc.data_range_[pidx], 0.0, 1.0).astype(np.float32)


# ── Feature construction (identical to multiseed_clean_rerun.py) ───────────
# Built entirely from the zero-filled raw CORRUPTED array -- no backbone
# output is ever read here.

def build_occ_X(cor, tmp, na, nm, pidx):
    """25-feature matrix -- local PRECIP dropped from the corrupted block."""
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

def build_amt_X(sc, z, cor_key, pidx, neighbor_avg_key, neighbor_mask_key):
    """26-feature matrix -- local PRECIP hard-zeroed."""
    cor = inv(sc, z[cor_key].astype(np.float32)).astype(np.float32)
    cor[:, pidx] = 0.0
    return np.concatenate([cor,
                           z['temporal'].astype(np.float32),
                           inv(sc, z[neighbor_avg_key].astype(np.float32)).astype(np.float32),
                           z[neighbor_mask_key].astype(np.float32)], axis=1)


# ── Metrics (identical to multiseed_clean_rerun.py) ─────────────────────────

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
    print('  DIRECT TWO-STAGE RF -- BACKBONE-FREE CLASSIFY-THEN-REGRESS BASELINE')
    print('  (no gan_imputed_test_*.npy is read; features come straight from')
    print('   the zero-filled raw corrupted records, per seed x scenario)')
    print('='*70)

    sc, mv = load_scaler()
    pidx = mv.index('PRECIP')

    tr = load_npz('train'); va = load_npz('val'); te = load_npz('test')
    tr_na_key, tr_nm_key = neighbor_keys(TRAIN_SCENARIO)
    require_keys(tr, [TRAIN_COR_KEY, tr_na_key, tr_nm_key], 'train')
    require_keys(va, [TRAIN_COR_KEY, tr_na_key, tr_nm_key], 'val')

    X_tr_full = build_occ_X(tr[TRAIN_COR_KEY].astype(np.float32),
                            tr['temporal'].astype(np.float32),
                            tr[tr_na_key].astype(np.float32),
                            tr[tr_nm_key].astype(np.float32), pidx)
    X_va_full = build_occ_X(va[TRAIN_COR_KEY].astype(np.float32),
                            va['temporal'].astype(np.float32),
                            va[tr_na_key].astype(np.float32),
                            va[tr_nm_key].astype(np.float32), pidx)
    X_tr_amt  = build_amt_X(sc, tr, TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)

    tr_gt = inv(sc, tr['data'].astype(np.float32))[:,pidx]
    va_gt = inv(sc, va['data'].astype(np.float32))[:,pidx]
    te_gt = inv(sc, te['data'].astype(np.float32))[:,pidx]

    tr_obs = tr['real_mask'].astype(np.float32)[:,pidx].astype(bool)
    va_obs = va['real_mask'].astype(np.float32)[:,pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs

    X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs]>WET_THRESH).astype(int)
    X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs]>WET_THRESH).astype(int)

    scen_features = {}
    for scen_label, cor_key, mask_key in SCENARIOS:
        na_key, nm_key = neighbor_keys(scen_label)
        require_keys(te, [cor_key, mask_key, na_key, nm_key], f'test/{scen_label}')
        scen_features[scen_label] = dict(
            X_occ=build_occ_X(te[cor_key].astype(np.float32),
                              te['temporal'].astype(np.float32),
                              te[na_key].astype(np.float32),
                              te[nm_key].astype(np.float32), pidx),
            X_amt=build_amt_X(sc, te, cor_key, pidx, na_key, nm_key),
            cor_norm=te[cor_key].astype(np.float32),
        )

    print(f'\n  Feature dim={X_tr_occ.shape[1]}  Train obs={len(X_tr_occ):,}  Val obs={len(X_va_occ):,}')

    all_records = []
    seed_meta   = []

    for seed in SEEDS:
        print(f'\n{"="*60}')
        print(f'  SEED = {seed}')
        print('='*60)

        occ_rf, occ_cut, vm = train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed)
        print(f'  Occ RF  cutoff={occ_cut}  val_F1={vm["val_f1"]}  '
              f'P={vm["val_prec"]}  R={vm["val_rec"]}  bias={vm["val_bias"]:+.4f}')
        vm.update(seed=seed)
        seed_meta.append(vm)

        pfx = os.path.join(SRC_DIR, f'direct_two_stage_occurrence_seed{seed}')
        with open(pfx+'.pkl','wb') as f:
            pickle.dump({'rf':occ_rf,'scaler':sc}, f)
        vm_copy = {k: v for k, v in vm.items() if k != 'seed'}
        meta_out = dict(seed=seed, n_estimators=300, min_samples_leaf=5,
                        n_features=25, local_precip_excluded=True,
                        backbone='none', feature_source='raw_corrupted_zero_filled',
                        threshold_strategy='val_F1_maximization',
                        training_scenario=TRAIN_SCENARIO,
                        training_corruption_key=TRAIN_COR_KEY,
                        test_corruption_strategy='scenario_specific',
                        neighbor_strategy='scenario_specific',
                        feature_blocks=['corrupted_no_precip(6)','temporal(5)',
                                        'neighbor_avg_scenario(7)','neighbor_mask_scenario(7)'],
                        **vm_copy)
        with open(pfx+'.json','w',encoding='utf-8') as f:
            json.dump(meta_out, f, indent=2)

        amt_rf = RandomForestRegressor(n_estimators=400, random_state=seed,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

        for scen_label, cor_key, mask_key in SCENARIOS:
            sf = scen_features[scen_label]
            m = te[mask_key].astype(np.float32)[:,pidx] > 0.5
            gt_m = te_gt[m]; n_m = int(m.sum())

            te_proba = occ_rf.predict_proba(sf['X_occ'])[:,1]
            te_wet_pred = (te_proba >= occ_cut).astype(bool)

            # -- Direct reconstruction: zero on predicted-dry, RF amount on
            #    predicted-wet. No backbone/continuous base value is used or
            #    available at any point. --
            pred_scen = np.zeros_like(te_gt)
            apply_sel = m & te_wet_pred
            if apply_sel.sum() > 0:
                pred_scen[apply_sel] = np.maximum(
                    amt_rf.predict(sf['X_amt'][apply_sel]), 0.0)

            # Save a full (N,7) normalised reconstruction for record-keeping.
            # Only the PRECIP column is meaningful; other columns are a
            # zero-filled passthrough of the corrupted array (not modelled
            # by this PRECIP-only baseline).
            recon_norm = sf['cor_norm'].copy()
            recon_norm = np.nan_to_num(recon_norm, nan=0.0)
            recon_norm[:, pidx] = to_norm(sc, pred_scen, pidx)
            recon_norm[m, pidx] = to_norm(sc, pred_scen[m], pidx)
            npy_path = os.path.join(
                SRC_DIR, f'direct_two_stage_rf_test_seed{seed}_{scen_label}.npy')
            np.save(npy_path, recon_norm.astype(np.float32))

            r = metrics(gt_m, pred_scen[m], f'DirectTwoStageRF_seed{seed}', scen_label, n_m)
            all_records.append(r)
            print(f'  [{scen_label} | {cor_key}] wet_pred={te_wet_pred[m].mean():.4f} '
                  f'F1={r["f1"]:.4f} bias={r["bias"]:+.4f} '
                  f'RMSE_wet={r["rmse_wet"]} RMSE_p95={r["rmse_p95"]}')

    # ── Save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(RESULTS_DIR, 'direct_two_stage_rf_evaluation.csv'), index=False)
    pd.DataFrame(seed_meta).to_csv(
        os.path.join(RESULTS_DIR, 'direct_two_stage_rf_seed_summary.csv'), index=False)

    # ── Aggregate: mean +/- std across seeds ───────────────────────────────────
    # NOTE: this used to also print an informational comparison against
    # results/multiseed_clean_evaluation.csv (a real DLPIF/WGAN-GP run).
    # That file is a legacy artifact from a different preprocessing snapshot
    # than the one this script runs against, and reading it -- even only for
    # a console printout, never saved to disk -- made the pipeline harder to
    # audit end-to-end. Removed; use check_d2s_dlpif_equivalence.py or a
    # fresh multiseed_clean_rerun.py run against the CURRENT
    # preprocessed_*.npz for that comparison instead.
    SCENS = ['10pct','20pct','block7d','block30d']

    print('\n' + '='*70)
    print('  DIRECT-TWO-STAGE-RF -- F1 / Bias / RMSE_wet / RMSE_p95 (mean +/- std)')
    print('='*70)
    g = df.groupby('scenario')
    for scen in SCENS:
        if scen not in g.groups: continue
        sub = g.get_group(scen)
        print(f'  {scen:<10} F1={sub.f1.mean():.4f}+/-{sub.f1.std():.4f}  '
              f'Bias={sub.bias.mean():+.4f}+/-{sub.bias.std():.4f}  '
              f'RMSE_wet={sub.rmse_wet.mean():.2f}+/-{sub.rmse_wet.std():.2f}  '
              f'RMSE_p95={sub.rmse_p95.mean():.2f}+/-{sub.rmse_p95.std():.2f}')

    print('\n  Outputs:')
    print('  -> results/direct_two_stage_rf_evaluation.csv')
    print('  -> results/direct_two_stage_rf_seed_summary.csv')
    print('  -> src/direct_two_stage_rf_test_seed{seed}_{scenario}.npy')
    print('='*70)

if __name__ == '__main__':
    main()

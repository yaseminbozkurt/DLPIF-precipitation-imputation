# -*- coding: utf-8 -*-
"""
confound_isolation_reduced_features.py
=========================================
R16 follow-up -- isolates whether the Ohio external-validation network's
unusually severe neighbour-loss-alone collapse (F1 0.897 -> 0.074 at zero
neighbours, external_validation_ohio/) is caused by that network's
CLIMATE/TOPOLOGY, or simply by its REDUCED 4-variable feature set (TMIN,
TMEAN, TMAX, PRECIP -- no RH_MEAN/P_MEAN/WIND_MEAN, unavailable in
GHCN-Daily), by re-running the SAME neighbour-loss graded ablation on the
PRIMARY Kutahya network, but with Stage 1 retrained on the SAME reduced
4-variable feature schema instead of the original 7.

If the Kutahya network ALSO collapses sharply under neighbour-loss-alone
once restricted to this 4-variable schema, the Ohio finding is explained
by feature availability, not climate. If it stays close to the original
6-variable result (F1 -> 0.574 at zero neighbours, Table 6.9), the
feature-set explanation is not supported and the Ohio-specific severity
requires a different explanation (left open).

Design
------
- Local features restricted to TMIN, TMEAN, TMAX (columns 0,1,2 of the
  7-variable meteo_vars order) -- RH_MEAN(3), P_MEAN(4), WIND_MEAN(5)
  excluded entirely from Stage 1's input, matching Ohio's exact local
  variable set.
- Neighbour features restricted to the same 4 variables as Ohio's network
  (TMIN, TMEAN, TMAX, PRECIP -- columns 0,1,2,6), i.e. neighbour_avg/mask
  sliced from the existing 7-variable neighbor_avg_10pct/neighbor_mask_10pct
  arrays (computed once from the unmodified raw records; the underlying
  data is untouched, only which columns Stage 1 is ALLOWED to see is
  restricted).
- Everything else identical to graded_context_loss.py's neighbour-loss
  family: PRECIP mask frozen at corrupted_10pct/art_mask_10pct, k=2
  adjacency severed via degrade_neighbours() (reused unmodified from
  graded_context_loss.py), neighbour-averaged features recomputed via
  01_data_preprocessing.compute_neighbor_avg() (reused unmodified) after
  severing, 3 context-mask seeds (101,202,303) x 3 model seeds
  (42,123,456), local context held at its natural (reduced-schema)
  level throughout this family -- i.e. this is a NEW Stage 1 classifier
  (16-feature: 3 local + 5 temporal + 8 neighbour), not the original
  25-feature classifier evaluated on fewer inputs.
- Stage 1 hyperparameters (300 trees, min_leaf=5, balanced) and the
  validation-F1-maximising threshold-selection grid are identical to
  Section 4.2 -- only the feature schema changes.

Output: results/rq4_context_availability/confound_isolation_reduced_features.csv
"""
import os
import sys
import io
import pickle
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import direct_two_stage_rf as d2s

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq4_context_availability')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_SEEDS = [42, 123, 456]
CONTEXT_SEEDS = [101, 202, 303]
WET_THRESH = d2s.WET_THRESH
NEIGHBOUR_LEVELS = [2, 1, 0]

# reduced feature schema, matching external_validation_ohio's 4-variable set
# meteo_vars order: TMIN(0), TMEAN(1), TMAX(2), RH_MEAN(3), P_MEAN(4), WIND_MEAN(5), PRECIP(6)
LOCAL_COLS_REDUCED = [0, 1, 2]          # TMIN, TMEAN, TMAX (non-PRECIP)
NEIGHBOUR_VARS_REDUCED = [0, 1, 2, 6]   # TMIN, TMEAN, TMAX, PRECIP

# reuse graded_context_loss.py's neighbour-severing utility and
# 01_data_preprocessing.py's neighbour-average recomputation, unmodified
_spec_gcl = importlib.util.spec_from_file_location("gcl", os.path.join(SRC_DIR, "graded_context_loss.py"))
_stdout_before = sys.stdout
sys.stdout = io.StringIO()
try:
    gcl = importlib.util.module_from_spec(_spec_gcl)
    _spec_gcl.loader.exec_module(gcl)
finally:
    sys.stdout = _stdout_before


def build_reduced_occ_X(cor7, tmp, na7, nm7):
    """16-feature matrix: 3 local (non-PRECIP) + 5 temporal + 8 neighbour
    (4 vars x avg+mask), matching external_validation_ohio's schema."""
    local = np.nan_to_num(cor7[:, LOCAL_COLS_REDUCED], nan=0.0).astype(np.float32)
    na = np.nan_to_num(na7[:, NEIGHBOUR_VARS_REDUCED], nan=0.0).astype(np.float32)
    nm = nm7[:, NEIGHBOUR_VARS_REDUCED].astype(np.float32)
    return np.concatenate([local, tmp.astype(np.float32), na, nm], axis=1)


def train_occ(X_tr, y_tr, X_va, y_va, seed):
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    va_p = rf.predict_proba(X_va)[:, 1]
    best_f1, best_cut = -1.0, 0.5
    for cut in np.arange(0.20, 0.82, 0.02):
        f = f1_score(y_va, (va_p >= cut).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_cut = f, float(cut)
    return rf, round(best_cut, 3), best_f1


def main():
    print('=' * 70)
    print('  R16 CONFOUND ISOLATION -- Kutahya network, Ohio-matched 4-variable schema')
    print('  Does neighbour-loss collapse as severely once RH/P/WIND are excluded?')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    assert mv[:3] == ['TMIN', 'TMEAN', 'TMAX'], mv
    tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')

    tr_cor = tr[d2s.TRAIN_COR_KEY].astype(np.float32)
    va_cor = va[d2s.TRAIN_COR_KEY].astype(np.float32)
    tr_na, tr_nm = tr[f'neighbor_avg_{d2s.TRAIN_SCENARIO}'], tr[f'neighbor_mask_{d2s.TRAIN_SCENARIO}']
    va_na, va_nm = va[f'neighbor_avg_{d2s.TRAIN_SCENARIO}'], va[f'neighbor_mask_{d2s.TRAIN_SCENARIO}']

    X_tr_full = build_reduced_occ_X(tr_cor, tr['temporal'].astype(np.float32), tr_na, tr_nm)
    X_va_full = build_reduced_occ_X(va_cor, va['temporal'].astype(np.float32), va_na, va_nm)

    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)

    X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs] > WET_THRESH).astype(int)
    X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs] > WET_THRESH).astype(int)
    print(f'  Reduced feature dim: {X_tr_occ.shape[1]} (Ohio external check used 16)')
    print(f'  Train obs={len(X_tr_occ)}  Val obs={len(X_va_occ)}')

    # -- test setup: PRECIP mask frozen at MCAR-10, same population as Table 6.9 --
    with open(os.path.join(SRC_DIR, 'adjacency.pkl'), 'rb') as f:
        adj = pickle.load(f)
    A_knn = adj['A_knn']; stations = adj['stations']
    corrupted_base = te[d2s.TRAIN_COR_KEY].astype(np.float32)
    m = te['art_mask_10pct'].astype(np.float32)[:, pidx] > 0.5
    m_rows = np.where(m)[0]
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
    station_ids = te['station_ids']
    temporal_te = te['temporal'].astype(np.float32)
    print(f'  Frozen MCAR-10 masked-PRECIP evaluation population: {len(m_rows)} positions')

    models = {}
    for seed in MODEL_SEEDS:
        occ_rf, cut, val_f1 = train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed)
        models[seed] = dict(occ_rf=occ_rf, cut=cut)
        print(f'  seed={seed}  cutoff={cut}  val_F1={val_f1:.4f}')

    rows = []
    y_true_eval = te_gt[m_rows]
    is_wet_true = y_true_eval > WET_THRESH

    # baseline: natural (2-of-2 neighbours) reduced-schema context
    na_natural = te[f'neighbor_avg_{d2s.TRAIN_SCENARIO}']
    nm_natural = te[f'neighbor_mask_{d2s.TRAIN_SCENARIO}']
    X_eval_natural = build_reduced_occ_X(corrupted_base, temporal_te, na_natural, nm_natural)[m_rows]
    for seed in MODEL_SEEDS:
        p = models[seed]['occ_rf'].predict_proba(X_eval_natural)[:, 1]
        wet_pred = p >= models[seed]['cut']
        f1 = f1_score(is_wet_true, wet_pred, zero_division=0)
        rows.append(dict(level='2/2 (natural)', context_seed=np.nan, model_seed=seed, f1=round(float(f1), 4)))

    for keep in [1, 0]:
        for cseed in CONTEXT_SEEDS:
            rng = np.random.default_rng(cseed)
            A_mod = gcl.degrade_neighbours(A_knn, stations, keep, rng)
            nbr_avg_mod, nbr_mask_mod = gcl.prep01.compute_neighbor_avg(corrupted_base, station_ids, A_mod, stations)
            X_eval_mod = build_reduced_occ_X(corrupted_base, temporal_te, nbr_avg_mod, nbr_mask_mod)[m_rows]
            for seed in MODEL_SEEDS:
                p = models[seed]['occ_rf'].predict_proba(X_eval_mod)[:, 1]
                wet_pred = p >= models[seed]['cut']
                f1 = f1_score(is_wet_true, wet_pred, zero_division=0)
                rows.append(dict(level=f'{keep}/2', context_seed=cseed, model_seed=seed, f1=round(float(f1), 4)))

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'confound_isolation_reduced_features.csv')
    df.to_csv(out_path, index=False)
    print(f'\n  -> {out_path}')

    agg = df.groupby('level')['f1'].agg(['mean', 'std']).reindex(['2/2 (natural)', '1/2', '0/2'])
    print('\n  Kutahya, Ohio-matched 4-variable schema, neighbour-loss family:')
    print(agg.round(4))
    print('\n  For comparison:')
    print('    Kutahya, original 7-variable schema (Table 6.9): 2/2=0.8117  1/2=0.7841  0/2=0.5742')
    print('    Ohio, 4-variable schema (external_validation_ohio):  2/2=0.8972  1/2=0.8463  0/2=0.0741')
    print('=' * 70)


if __name__ == '__main__':
    main()

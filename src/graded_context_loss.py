"""
graded_context_loss.py
=========================
RQ4a -- Graded Context Loss experiment. Answers what the bimodal
context_availability_diagnostic.py (13,812 pooled rows, EMPTY at
A_total in [.2,.4), n=24 at [.4,.6)) could not: where, between "full
context" and "NetBlock-30d-style total collapse", does DLPIF's
reliability actually break?

CORRECTION vs the originally proposed design: the adjacency structure
built by 01_data_preprocessing.py::build_adjacency() uses k=2 (each of
the 4 stations has exactly 2 nearest neighbours, not 3 -- verified
against adjacency.pkl). Neighbour-loss below is therefore graded over
{2, 1, 0} available neighbours, not {3, 2, 1, 0}.

Design
------
- The PRECIP missingness mask is FROZEN at corrupted_10pct / art_mask_10pct
  (MCAR-10, already used throughout RQ1-RQ3) -- the exact same ~416 masked
  evaluation positions are used at every context level in every family, so
  any metric change is attributable ONLY to context availability, not to a
  different set of positions being evaluated. RQ1 already covers outcome-
  dependent PRECIP missingness; RQ4 does not introduce a new confounder.
- Three independent families, each degrading ONLY what its name says
  (the other context source is left exactly as in corrupted_10pct):
    local-loss      -- 6 non-PRECIP local variables, keep_count in
                        {6,4,3,2,0} (realized fractions ~{1.00,.67,.50,
                        .33,.00}; at each masked row, which columns to
                        keep is drawn from the row's currently-available
                        columns, never adding signal that wasn't there).
    neighbour-loss   -- station-level: for each target station, sever 0,
                        1, or 2 of its 2 real adjacency-graph neighbours
                        (zero their entry in a COPY of A_knn, then
                        recompute neighbor_avg/neighbor_mask via
                        01_data_preprocessing.compute_neighbor_avg() --
                        this removes station j's contribution to i's
                        neighbour features WITHOUT touching j's own local
                        records, unlike NaN-ing j's raw data would).
    joint-loss       -- both simultaneously, 3 paired levels: full
                        (local keep=6, neighbours=2), partial (local
                        keep=3, neighbours=1), none (local keep=0,
                        neighbours=0).
- 3 CONTEXT-MASK seeds (101, 202, 303 -- deliberately distinct from the
  3 MODEL seeds 42/123/456) control which specific columns/neighbours are
  dropped at partial levels, so results are reported as mean +/- SD over
  9 (context_seed x model_seed) combinations per level, not a single
  arbitrary realization.
- Stage-1 occurrence: the RQ2-frozen pretrained RF + Platt mapping +
  frozen threshold (never refit). Stage-2 amount: retrained with
  direct_two_stage_rf.py's own deterministic procedure (as in RQ3, never
  persisted anywhere in this repo). Conformal q: the RQ3-frozen per-
  model-seed quantile (VAL-CAL is untouched by context loss, which only
  modifies TEST features, so reusing q is correct, not a shortcut).
- No gate or fallback is built here. This script only characterizes the
  curve; threshold selection is a separate, later step done on validation
  data, per the user's explicit 5-point pre-registered success criteria
  (see module-level NOTES below).

NOTES (pre-registered before running, per user instruction)
-------------------------------------------------------------
1. Reliability need not be monotonic in availability.
2. A "cliff" is only claimed where at least two intermediate levels have
   adequate n -- here n is fixed at ~416 x 9 = ~3,744 per level (every
   masked-10pct position x 3 context seeds x 3 model seeds), by
   construction, unlike the earlier pooled diagnostic's sparse bins.
3. No collapse threshold is chosen from these results -- that is RQ4b.
4. Local and neighbour effects are reported as separate families, not
   collapsed into one composite score, specifically to test the
   diagnostic's suggestion that neighbour-loss is the dominant driver.

Outputs (results/rq4_context_availability/graded/):
  graded_local_loss.csv
  graded_neighbour_loss.csv
  graded_joint_loss.csv
"""
import os
import sys
import pickle
import io
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import brier_score_loss, f1_score

import direct_two_stage_rf as d2s
from calibrate_occurrence_probability import apply_platt, expected_calibration_error

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
RQ3_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq3_conformal')
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq4_context_availability', 'graded')
os.makedirs(OUT_DIR, exist_ok=True)

_spec = importlib.util.spec_from_file_location("prep01", os.path.join(SRC_DIR, "01_data_preprocessing.py"))
prep01 = importlib.util.module_from_spec(_spec)
# direct_two_stage_rf (imported above as d2s) already wrapped sys.stdout in a
# UTF-8 TextIOWrapper; 01_data_preprocessing.py's module-level code
# unconditionally wraps AGAIN with `io.TextIOWrapper(sys.stdout.buffer, ...)`
# -- both wrappers then share the SAME underlying buffer, and whichever one
# ends up orphaned gets its buffer closed by the other's GC finalizer,
# breaking every later print() with "I/O operation on closed file"
# (documented footgun, see direct_two_stage_rf.py's own comment on this
# exact issue; simply saving/restoring sys.stdout around the exec does NOT
# fix it -- it just moves which wrapper ends up orphaned). Swapping in a
# StringIO (no `.buffer` attribute) makes that re-wrap attempt raise
# AttributeError, silently swallowed by 01_data_preprocessing.py's own
# `except Exception: pass`, so no second TextIOWrapper around the real
# buffer is ever created.
_stdout_before_prep01_import = sys.stdout
sys.stdout = io.StringIO()
_spec.loader.exec_module(prep01)
sys.stdout = _stdout_before_prep01_import

MODEL_SEEDS = [42, 123, 456]
CONTEXT_SEEDS = [101, 202, 303]
WET_THRESH = d2s.WET_THRESH
P95_THRESH = d2s.P95_THRESH
ALPHA = 0.10
LOCAL_LEVELS = [6, 4, 3, 2, 0]        # keep-count out of 6 non-PRECIP local vars
NEIGHBOUR_LEVELS = [2, 1, 0]          # keep-count out of 2 real adjacency neighbours
JOINT_LEVELS = [(6, 2), (3, 1), (0, 0)]  # (local keep, neighbour keep) paired


def degrade_local(corrupted, m_rows, local_cols, keep_count, rng):
    """Row-local: from each masked row's currently-available local columns,
    keep at most `keep_count`, NaN the rest. Never adds signal.
    """
    out = corrupted.copy()
    for row in m_rows:
        avail = [c for c in local_cols if not np.isnan(out[row, c])]
        if len(avail) <= keep_count:
            continue  # already at or below target -- nothing to remove
        drop = rng.choice(avail, size=len(avail) - keep_count, replace=False)
        out[row, drop] = np.nan
    return out


def degrade_neighbours(A_knn, stations, keep_count, rng):
    """Zero out (2 - keep_count) of each station's 2 real neighbours in a
    COPY of A_knn. Returns the modified adjacency matrix only -- raw
    station data is never touched, so severed stations' OWN local rows
    (when they are themselves the target) are unaffected.
    """
    A_mod = A_knn.copy()
    n = len(stations)
    for i in range(n):
        nbr_idx = [j for j in range(n) if A_knn[i, j] > 0]
        if len(nbr_idx) <= keep_count:
            continue
        drop = rng.choice(nbr_idx, size=len(nbr_idx) - keep_count, replace=False)
        A_mod[i, drop] = 0.0
    return A_mod


def compute_metrics(y_true, p_cal, wet_pred, yhat, lo_oracle, hi_oracle, y_oracle,
                    lo_e2e, hi_e2e):
    is_wet_true = y_true > WET_THRESH
    f1 = f1_score(is_wet_true, wet_pred, zero_division=0)
    brier = brier_score_loss(is_wet_true, p_cal)
    ece, _ = expected_calibration_error(p_cal, is_wet_true.astype(int))
    bias = float(wet_pred.mean() - is_wet_true.mean())
    wet_sel = is_wet_true
    rmse_wet = (float(np.sqrt(np.mean((yhat[wet_sel] - y_true[wet_sel]) ** 2)))
               if wet_sel.sum() > 0 else np.nan)
    picp_oracle = float(np.mean((y_oracle >= lo_oracle) & (y_oracle <= hi_oracle))) if len(y_oracle) else np.nan
    picp_e2e = float(np.mean((y_true >= lo_e2e) & (y_true <= hi_e2e)))
    return dict(f1=round(f1, 4), brier=round(brier, 4), ece=round(float(ece), 4),
               bias=round(bias, 4), rmse_wet=round(rmse_wet, 4) if rmse_wet == rmse_wet else np.nan,
               picp_oracle=round(picp_oracle, 4) if picp_oracle == picp_oracle else np.nan,
               picp_e2e=round(picp_e2e, 4))


def main():
    print('=' * 70)
    print('  RQ4a -- GRADED CONTEXT LOSS (PRECIP mask frozen at MCAR-10)')
    print('  local-loss / neighbour-loss (k=2, corrected) / joint-loss')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    local_cols = [i for i in range(len(mv)) if i != pidx]
    tr = d2s.load_npz('train')
    te = d2s.load_npz('test')

    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)
    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs
    X_tr_amt = d2s.build_amt_X(sc, tr, d2s.TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)

    with open(os.path.join(SRC_DIR, 'adjacency.pkl'), 'rb') as f:
        adj = pickle.load(f)
    A_knn = adj['A_knn']
    stations = adj['stations']
    assert adj['k'] == 2, f"expected k=2 adjacency, got k={adj['k']}"

    corrupted_base = te[d2s.TRAIN_COR_KEY].astype(np.float32)  # corrupted_10pct
    m = te['art_mask_10pct'].astype(np.float32)[:, pidx] > 0.5
    m_rows = np.where(m)[0]
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
    station_ids = te['station_ids']
    temporal = te['temporal'].astype(np.float32)
    print(f'  Frozen MCAR-10 masked-PRECIP evaluation population: {len(m_rows)} positions')

    calib = pd.read_csv(os.path.join(RQ3_DIR, 'conformal_calibration_summary.csv'))
    thresholds = pd.read_csv(os.path.join(RQ2_DIR, 'thresholds.csv'))

    models = {}
    for seed in MODEL_SEEDS:
        with open(os.path.join(SRC_DIR, f'direct_two_stage_occurrence_seed{seed}.pkl'), 'rb') as f:
            occ_rf = pickle.load(f)['rf']
        with open(os.path.join(RQ2_DIR, 'calibration_models', f'platt_seed{seed}.pkl'), 'rb') as f:
            platt = pickle.load(f)
        tau = float(thresholds[(thresholds.seed == seed) & (thresholds.variant == 'platt')]['threshold'].iloc[0])
        q = float(calib[calib.seed == seed]['q'].iloc[0])
        amt_rf = RandomForestRegressor(n_estimators=400, random_state=seed,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])
        models[seed] = dict(occ_rf=occ_rf, platt=platt, tau=tau, q=q, amt_rf=amt_rf)
        print(f'  seed={seed}: tau_platt={tau:.3f}  conformal q={q:.3f}mm')

    def evaluate(corrupted_local, neighbor_avg, neighbor_mask, seed):
        mdl = models[seed]
        X_occ = d2s.build_occ_X(corrupted_local, temporal, neighbor_avg, neighbor_mask, pidx)
        cor_inv = d2s.inv(sc, corrupted_local)
        cor_inv[:, pidx] = 0.0
        X_amt = np.concatenate([cor_inv, temporal, d2s.inv(sc, neighbor_avg), neighbor_mask], axis=1)

        y_true = te_gt[m]
        raw_p = mdl['occ_rf'].predict_proba(X_occ[m])[:, 1]
        p_cal = apply_platt(mdl['platt'], raw_p)
        wet_pred = p_cal >= mdl['tau']
        yhat = np.zeros_like(y_true)
        lo_e2e = np.zeros_like(y_true); hi_e2e = np.zeros_like(y_true)
        if wet_pred.sum() > 0:
            pred = mdl['amt_rf'].predict(X_amt[m][wet_pred])
            yhat[wet_pred] = pred
            lo_e2e[wet_pred] = np.maximum(0.0, pred - mdl['q'])
            hi_e2e[wet_pred] = pred + mdl['q']

        oracle_mask = m & (te_gt > WET_THRESH)
        y_oracle = te_gt[oracle_mask]
        yhat_oracle = mdl['amt_rf'].predict(X_amt[oracle_mask])
        lo_oracle = np.maximum(0.0, yhat_oracle - mdl['q'])
        hi_oracle = yhat_oracle + mdl['q']

        return compute_metrics(y_true, p_cal, wet_pred, yhat, lo_oracle, hi_oracle, y_oracle,
                               lo_e2e, hi_e2e)

    # Base (unmodified corrupted_10pct) neighbour features, reused whenever
    # a family doesn't touch the neighbour side.
    base_nbr_avg = te['neighbor_avg_10pct'].astype(np.float32)
    base_nbr_mask = te['neighbor_mask_10pct'].astype(np.float32)

    # ── Local-loss ───────────────────────────────────────────────────────
    print('\n  --- local-loss ---')
    local_rows = []
    for keep in LOCAL_LEVELS:
        for cseed in CONTEXT_SEEDS:
            rng = np.random.default_rng(cseed * 1000 + keep)
            cor_mod = degrade_local(corrupted_base, m_rows, local_cols, keep, rng)
            realized_a_local = float(np.mean(~np.isnan(cor_mod[m][:, local_cols])))
            for mseed in MODEL_SEEDS:
                row = evaluate(cor_mod, base_nbr_avg, base_nbr_mask, mseed)
                row.update(family='local', level=keep, keep_count=keep,
                          realized_availability=round(realized_a_local, 4),
                          context_seed=cseed, model_seed=mseed)
                local_rows.append(row)
        print(f'    keep={keep}/6  realized_A_local~{realized_a_local:.3f}')
    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(os.path.join(OUT_DIR, 'graded_local_loss.csv'), index=False)

    # ── Neighbour-loss ───────────────────────────────────────────────────
    print('\n  --- neighbour-loss (k=2) ---')
    nbr_rows = []
    for keep in NEIGHBOUR_LEVELS:
        for cseed in CONTEXT_SEEDS:
            rng = np.random.default_rng(cseed * 1000 + 500 + keep)
            A_mod = degrade_neighbours(A_knn, stations, keep, rng)
            nbr_avg_mod, nbr_mask_mod = prep01.compute_neighbor_avg(corrupted_base, station_ids, A_mod, stations)
            realized_a_nbr = float(np.mean(nbr_mask_mod[m]))
            for mseed in MODEL_SEEDS:
                row = evaluate(corrupted_base, nbr_avg_mod, nbr_mask_mod, mseed)
                row.update(family='neighbour', level=keep, keep_count=keep,
                          realized_availability=round(realized_a_nbr, 4),
                          context_seed=cseed, model_seed=mseed)
                nbr_rows.append(row)
        print(f'    keep={keep}/2  realized_A_neighbour~{realized_a_nbr:.3f}')
    nbr_df = pd.DataFrame(nbr_rows)
    nbr_df.to_csv(os.path.join(OUT_DIR, 'graded_neighbour_loss.csv'), index=False)

    # ── Joint-loss ───────────────────────────────────────────────────────
    print('\n  --- joint-loss ---')
    joint_rows = []
    for keep_local, keep_nbr in JOINT_LEVELS:
        for cseed in CONTEXT_SEEDS:
            rng_l = np.random.default_rng(cseed * 1000 + 900 + keep_local)
            rng_n = np.random.default_rng(cseed * 1000 + 950 + keep_nbr)
            cor_mod = degrade_local(corrupted_base, m_rows, local_cols, keep_local, rng_l)
            A_mod = degrade_neighbours(A_knn, stations, keep_nbr, rng_n)
            nbr_avg_mod, nbr_mask_mod = prep01.compute_neighbor_avg(cor_mod, station_ids, A_mod, stations)
            realized_a_local = float(np.mean(~np.isnan(cor_mod[m][:, local_cols])))
            realized_a_nbr = float(np.mean(nbr_mask_mod[m]))
            for mseed in MODEL_SEEDS:
                row = evaluate(cor_mod, nbr_avg_mod, nbr_mask_mod, mseed)
                row.update(family='joint', level=f'{keep_local}L+{keep_nbr}N',
                          keep_local=keep_local, keep_nbr=keep_nbr,
                          realized_a_local=round(realized_a_local, 4),
                          realized_a_neighbour=round(realized_a_nbr, 4),
                          context_seed=cseed, model_seed=mseed)
                joint_rows.append(row)
        print(f'    local_keep={keep_local}/6 nbr_keep={keep_nbr}/2  '
              f'realized A_local~{realized_a_local:.3f} A_nbr~{realized_a_nbr:.3f}')
    joint_df = pd.DataFrame(joint_rows)
    joint_df.to_csv(os.path.join(OUT_DIR, 'graded_joint_loss.csv'), index=False)

    print(f'\n  -> {os.path.join(OUT_DIR, "graded_local_loss.csv")}')
    print(f'  -> {os.path.join(OUT_DIR, "graded_neighbour_loss.csv")}')
    print(f'  -> {os.path.join(OUT_DIR, "graded_joint_loss.csv")}')

    for name, df, level_col in [('LOCAL-LOSS', local_df, 'keep_count'),
                                ('NEIGHBOUR-LOSS', nbr_df, 'keep_count'),
                                ('JOINT-LOSS', joint_df, 'level')]:
        print(f'\n  {name} (mean +/- SD over {len(CONTEXT_SEEDS)} context seeds x {len(MODEL_SEEDS)} model seeds):')
        g = df.groupby(level_col)[['f1', 'brier', 'ece', 'rmse_wet', 'picp_oracle', 'picp_e2e']]
        summary = g.agg(['mean', 'std'])
        print(summary.round(4))
    print('=' * 70)


if __name__ == '__main__':
    main()

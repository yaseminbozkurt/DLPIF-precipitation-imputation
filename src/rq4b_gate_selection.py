"""
rq4b_gate_selection.py
=========================
RQ4b, phase 1 -- select a context-availability gate policy USING ONLY
VALIDATION DATA, comparing DLPIF against a Linear-interpolation fallback
across the same graded local/neighbour/joint context-loss regimes RQ4a
characterized on test (graded_context_loss.py). Test scenarios are never
touched by this script -- the winning policy is only APPLIED to test in
the separate, later rq4b_apply_gate.py, so gate selection cannot overfit
test.

Per the user's explicit correction: RQ4a already showed a scalar
availability score cannot distinguish "neighbour-only loss" (F1 stays at
0.574) from "joint loss" (F1=0.000) -- so this script evaluates THREE gate
families side by side, not just a single scalar threshold:
  no-gate         -- always DLPIF (existing baseline, no fallback ever).
  scalar-gate      -- A* = w_L*A_local + w_N*A_neighbor >= tau => DLPIF,
                      else fallback. Grid: w_L/w_N in {(.2,.8),(.3,.7),
                      (.4,.6)} (neighbour-weighted, per RQ4a), tau in
                      [0.05, 0.95] step 0.05.
  interaction-gate -- fallback iff (A_neighbor < tau_N) AND
                      (A_local < tau_L). Grid: tau_N in {0.25, 0.75}
                      (the only two cut points that distinguish this
                      dataset's 3 achievable A_neighbor values 0/.5/1 --
                      finer tau_N values are indistinguishable from one of
                      these two given k=2 adjacency), tau_L in
                      [0.0, 1.0] step 0.1.
  regime-table (reference only, not a deployable parametric rule) -- the
                      per-regime argmax(DLPIF utility, fallback utility);
                      an upper bound on any parametric gate's achievable
                      validation utility, reported for context.

Utility (adapted from the user's proposed U = F1 - lambda*RMSE_wet_norm -
Brier): the Brier term is DROPPED here because Linear interpolation is not
a probabilistic method and has no calibrated probability to score --
comparing a Brier-penalized DLPIF against a Brier-free fallback would not
be a fair comparison. U = F1 - 0.1 * RMSE_wet / std(train PRECIP | wet).
Brier is still reported as a separate DLPIF-only diagnostic column.

Regimes (identical grid to graded_context_loss.py, replicated on
preprocessed_val.npz's own corrupted_10pct/art_mask_10pct -- val's
existing MCAR-10 masked-PRECIP population, ~415 positions, held fixed
across every level exactly as on test):
  local-loss: keep in {6,4,3,2,0} (of 6 non-PRECIP local variables)
  neighbour-loss: keep in {2,1,0} (of 2 real k=2 adjacency neighbours)
  joint-loss: (6,2) / (3,1) / (0,0) paired

Outputs (results/rq4_context_availability/gate/):
  validation_regime_utility.csv   per (family, level, context_seed,
                                   model_seed): A_local, A_neighbor,
                                   DLPIF/fallback F1, RMSE_wet, utility
  gate_candidates_ranked.csv      every no-gate/scalar-gate/interaction-
                                   gate/regime-table candidate x its mean
                                   validation utility, ranked
  selected_gate.json              the frozen, winning policy (parametric
                                   family + parameters), to be applied to
                                   test by rq4b_apply_gate.py
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import direct_two_stage_rf as d2s
from calibrate_occurrence_probability import apply_platt
from graded_context_loss import (degrade_local, degrade_neighbours,
                                 LOCAL_LEVELS, NEIGHBOUR_LEVELS, JOINT_LEVELS,
                                 CONTEXT_SEEDS, MODEL_SEEDS, prep01)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq4_context_availability', 'gate')
os.makedirs(OUT_DIR, exist_ok=True)

WET_THRESH = d2s.WET_THRESH
LAMBDA_RMSE = 0.1
SCALAR_WEIGHTS = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
SCALAR_TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
INTERACTION_TAU_N = [0.25, 0.75]
INTERACTION_TAU_L = np.round(np.arange(0.0, 1.01, 0.1), 2)


def station_wise_linear_interpolation(corrupted, station_ids):
    """Same method as 03_baseline_imputation.py::linear_interpolation()
    (reimplemented locally to avoid importing a numeral-prefixed module
    name), applied here to whatever (possibly context-degraded) corrupted
    array a regime produces.
    """
    station_ids = np.asarray(station_ids)
    out = corrupted.copy()
    n_vars = corrupted.shape[1]
    cols = [f'v{i}' for i in range(n_vars)]
    for sid in pd.unique(station_ids):
        idx = np.where(station_ids == sid)[0]
        df_s = pd.DataFrame(corrupted[idx], columns=cols)
        out[idx] = df_s.interpolate(method='linear', limit_direction='both').values
    return np.clip(out, 0, 1).astype(np.float32)


def dlpif_regime_metrics(corrupted, neighbor_avg, neighbor_mask, temporal, m,
                         y_true, sc, pidx, mdl, rmse_norm):
    X_occ = d2s.build_occ_X(corrupted, temporal, neighbor_avg, neighbor_mask, pidx)
    cor_inv = d2s.inv(sc, corrupted)
    cor_inv[:, pidx] = 0.0
    X_amt = np.concatenate([cor_inv, temporal, d2s.inv(sc, neighbor_avg), neighbor_mask], axis=1)

    raw_p = mdl['occ_rf'].predict_proba(X_occ[m])[:, 1]
    p_cal = apply_platt(mdl['platt'], raw_p)
    wet_pred = p_cal >= mdl['tau']
    yhat = np.zeros_like(y_true)
    if wet_pred.sum() > 0:
        yhat[wet_pred] = mdl['amt_rf'].predict(X_amt[m][wet_pred])

    is_wet_true = y_true > WET_THRESH
    from sklearn.metrics import f1_score, brier_score_loss
    f1 = f1_score(is_wet_true, wet_pred, zero_division=0)
    brier = brier_score_loss(is_wet_true, p_cal)
    wet_sel = is_wet_true
    rmse_wet = (float(np.sqrt(np.mean((yhat[wet_sel] - y_true[wet_sel]) ** 2)))
               if wet_sel.sum() > 0 else np.nan)
    utility = f1 - LAMBDA_RMSE * (rmse_wet / rmse_norm if rmse_wet == rmse_wet else 0.0)
    return dict(f1=round(float(f1), 4), brier=round(float(brier), 4),
               rmse_wet=round(rmse_wet, 4) if rmse_wet == rmse_wet else np.nan,
               utility=round(float(utility), 4))


def linear_regime_metrics(corrupted, station_ids, m, y_true, sc, pidx, rmse_norm):
    lin = station_wise_linear_interpolation(corrupted, station_ids)
    pred_mm = d2s.inv(sc, lin)[:, pidx][m]
    is_wet_true = y_true > WET_THRESH
    wet_pred = pred_mm > WET_THRESH
    from sklearn.metrics import f1_score
    f1 = f1_score(is_wet_true, wet_pred, zero_division=0)
    wet_sel = is_wet_true
    rmse_wet = (float(np.sqrt(np.mean((pred_mm[wet_sel] - y_true[wet_sel]) ** 2)))
               if wet_sel.sum() > 0 else np.nan)
    utility = f1 - LAMBDA_RMSE * (rmse_wet / rmse_norm if rmse_wet == rmse_wet else 0.0)
    return dict(f1=round(float(f1), 4),
               rmse_wet=round(rmse_wet, 4) if rmse_wet == rmse_wet else np.nan,
               utility=round(float(utility), 4))


def main():
    print('=' * 70)
    print('  RQ4b phase 1 -- GATE SELECTION ON VALIDATION (test untouched)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    local_cols = [i for i in range(len(mv)) if i != pidx]
    tr = d2s.load_npz('train')
    va = d2s.load_npz('val')

    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)
    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs
    X_tr_amt = d2s.build_amt_X(sc, tr, d2s.TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)
    rmse_norm = float(np.std(tr_gt[tr_wet]))
    print(f'  RMSE normalizer (train wet-day PRECIP std) = {rmse_norm:.3f} mm')

    with open(os.path.join(SRC_DIR, 'adjacency.pkl'), 'rb') as f:
        adj = pickle.load(f)
    A_knn = adj['A_knn']
    stations = adj['stations']

    corrupted_base = va[d2s.TRAIN_COR_KEY].astype(np.float32)
    m = va['art_mask_10pct'].astype(np.float32)[:, pidx] > 0.5
    m_rows = np.where(m)[0]
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    station_ids = va['station_ids']
    temporal = va['temporal'].astype(np.float32)
    print(f'  Validation MCAR-10 masked-PRECIP population: {len(m_rows)} positions')

    thresholds = pd.read_csv(os.path.join(RQ2_DIR, 'thresholds.csv'))
    models = {}
    for seed in MODEL_SEEDS:
        with open(os.path.join(SRC_DIR, f'direct_two_stage_occurrence_seed{seed}.pkl'), 'rb') as f:
            occ_rf = pickle.load(f)['rf']
        with open(os.path.join(RQ2_DIR, 'calibration_models', f'platt_seed{seed}.pkl'), 'rb') as f:
            platt = pickle.load(f)
        tau = float(thresholds[(thresholds.seed == seed) & (thresholds.variant == 'platt')]['threshold'].iloc[0])
        amt_rf = RandomForestRegressor(n_estimators=400, random_state=seed,
                                       min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])
        models[seed] = dict(occ_rf=occ_rf, platt=platt, tau=tau, amt_rf=amt_rf)

    base_nbr_avg = va['neighbor_avg_10pct'].astype(np.float32)
    base_nbr_mask = va['neighbor_mask_10pct'].astype(np.float32)
    y_true = va_gt[m]

    rows = []

    def run_family(family, cor_fn, nbr_fn, level_label_fn):
        for level in cor_fn['levels']:
            for cseed in CONTEXT_SEEDS:
                cor_mod, nbr_avg_mod, nbr_mask_mod, realized = cor_fn['build'](level, cseed)
                for mseed in MODEL_SEEDS:
                    dlpif_m = dlpif_regime_metrics(cor_mod, nbr_avg_mod, nbr_mask_mod, temporal, m,
                                                   y_true, sc, pidx, models[mseed], rmse_norm)
                    lin_m = linear_regime_metrics(cor_mod, station_ids, m, y_true, sc, pidx, rmse_norm)
                    rows.append(dict(family=family, level=level_label_fn(level),
                                     context_seed=cseed, model_seed=mseed,
                                     a_local=realized[0], a_neighbor=realized[1],
                                     dlpif_f1=dlpif_m['f1'], dlpif_rmse_wet=dlpif_m['rmse_wet'],
                                     dlpif_brier=dlpif_m['brier'], dlpif_utility=dlpif_m['utility'],
                                     fallback_f1=lin_m['f1'], fallback_rmse_wet=lin_m['rmse_wet'],
                                     fallback_utility=lin_m['utility']))
            print(f'    {family} level={level} done')

    print('\n  --- local-loss ---')
    def build_local(keep, cseed):
        rng = np.random.default_rng(cseed * 1000 + keep)
        cor_mod = degrade_local(corrupted_base, m_rows, local_cols, keep, rng)
        a_local = float(np.mean(~np.isnan(cor_mod[m][:, local_cols])))
        a_nbr = float(np.mean(base_nbr_mask[m]))
        return cor_mod, base_nbr_avg, base_nbr_mask, (a_local, a_nbr)
    run_family('local', {'levels': LOCAL_LEVELS, 'build': build_local}, None, lambda k: k)

    print('\n  --- neighbour-loss ---')
    def build_nbr(keep, cseed):
        rng = np.random.default_rng(cseed * 1000 + 500 + keep)
        A_mod = degrade_neighbours(A_knn, stations, keep, rng)
        nbr_avg_mod, nbr_mask_mod = prep01.compute_neighbor_avg(corrupted_base, station_ids, A_mod, stations)
        a_local = float(np.mean(~np.isnan(corrupted_base[m][:, local_cols])))
        a_nbr = float(np.mean(nbr_mask_mod[m]))
        return corrupted_base, nbr_avg_mod, nbr_mask_mod, (a_local, a_nbr)
    run_family('neighbour', {'levels': NEIGHBOUR_LEVELS, 'build': build_nbr}, None, lambda k: k)

    print('\n  --- joint-loss ---')
    def build_joint(pair, cseed):
        keep_local, keep_nbr = pair
        rng_l = np.random.default_rng(cseed * 1000 + 900 + keep_local)
        rng_n = np.random.default_rng(cseed * 1000 + 950 + keep_nbr)
        cor_mod = degrade_local(corrupted_base, m_rows, local_cols, keep_local, rng_l)
        A_mod = degrade_neighbours(A_knn, stations, keep_nbr, rng_n)
        nbr_avg_mod, nbr_mask_mod = prep01.compute_neighbor_avg(cor_mod, station_ids, A_mod, stations)
        a_local = float(np.mean(~np.isnan(cor_mod[m][:, local_cols])))
        a_nbr = float(np.mean(nbr_mask_mod[m]))
        return cor_mod, nbr_avg_mod, nbr_mask_mod, (a_local, a_nbr)
    run_family('joint', {'levels': JOINT_LEVELS, 'build': build_joint},
              None, lambda p: f'{p[0]}L+{p[1]}N')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'validation_regime_utility.csv'), index=False)
    print(f'\n  -> {os.path.join(OUT_DIR, "validation_regime_utility.csv")}  ({len(df)} rows)')

    # ── Aggregate to one row per (family, level) regime ─────────────────
    regime = df.groupby(['family', 'level'])[
        ['a_local', 'a_neighbor', 'dlpif_utility', 'fallback_utility',
        'dlpif_f1', 'fallback_f1', 'dlpif_rmse_wet', 'fallback_rmse_wet']
    ].mean().reset_index()
    print('\n  Regime-level validation utility (mean over 3 context seeds x 3 model seeds):')
    print(regime.round(4).to_string(index=False))

    # ── Gate candidates ──────────────────────────────────────────────────
    candidates = []

    no_gate_util = regime['dlpif_utility'].mean()
    candidates.append(dict(family='no-gate', params='{}', mean_utility=round(float(no_gate_util), 4)))

    regime_table_util = np.maximum(regime['dlpif_utility'], regime['fallback_utility']).mean()
    candidates.append(dict(family='regime-table (reference only)', params='{}',
                           mean_utility=round(float(regime_table_util), 4)))

    for wL, wN in SCALAR_WEIGHTS:
        a_star = wL * regime['a_local'] + wN * regime['a_neighbor']
        for tau in SCALAR_TAUS:
            use_dlpif = a_star >= tau
            util = np.where(use_dlpif, regime['dlpif_utility'], regime['fallback_utility']).mean()
            candidates.append(dict(family='scalar-gate',
                                   params=json.dumps(dict(w_local=wL, w_neighbor=wN, tau=round(float(tau), 2))),
                                   mean_utility=round(float(util), 4)))

    for tau_N in INTERACTION_TAU_N:
        for tau_L in INTERACTION_TAU_L:
            fallback_trig = (regime['a_neighbor'] < tau_N) & (regime['a_local'] < tau_L)
            util = np.where(~fallback_trig, regime['dlpif_utility'], regime['fallback_utility']).mean()
            candidates.append(dict(family='interaction-gate',
                                   params=json.dumps(dict(tau_neighbor=tau_N, tau_local=round(float(tau_L), 2))),
                                   mean_utility=round(float(util), 4)))

    cand_df = pd.DataFrame(candidates).sort_values('mean_utility', ascending=False).reset_index(drop=True)
    cand_df.to_csv(os.path.join(OUT_DIR, 'gate_candidates_ranked.csv'), index=False)
    print(f'\n  -> {os.path.join(OUT_DIR, "gate_candidates_ranked.csv")}  ({len(cand_df)} candidates)')

    print('\n  Top candidate per family:')
    for fam in ['no-gate', 'regime-table (reference only)', 'scalar-gate', 'interaction-gate']:
        sub = cand_df[cand_df.family == fam]
        if len(sub):
            print(f'    {fam:32s} best: {sub.iloc[0]["params"]}  utility={sub.iloc[0]["mean_utility"]:.4f}')

    # Winner selection. This validation regime grid is sparse (11 distinct
    # (a_local, a_neighbor) points), so max utility alone is NOT a unique
    # tie-break: on this run 24 scalar-gate parameterizations and 4
    # interaction-gate parameterizations all tie the regime-table oracle's
    # utility exactly (every one of them correctly separates the single
    # "a_neighbor=0" failure mode from everything else). Reported
    # honestly rather than silently picking whichever happened to sort
    # first. Tie-break policy, decided BEFORE looking at which family
    # would win it (not fit to the outcome):
    #   1. exclude regime-table (reference/upper-bound only, not deployable
    #      as a rule for a_local/a_neighbor combinations it never saw).
    #   2. among ties at max utility, prefer interaction-gate over
    #      scalar-gate -- RQ4a's interaction-effect finding (collapse
    #      needs BOTH local and neighbour loss) is a mechanistic argument
    #      for a conjunctive rule, not a curve-fit one; a scalar boundary
    #      that happens to separate today's 11 points is not guaranteed to
    #      extrapolate the same way to an untested (a_local, a_neighbor)
    #      combination the way the interaction structure does.
    #   3. among interaction-gate ties, prefer tau_local NOT at the grid
    #      edge (0.9 over 1.0) -- the tightest constraint the data
    #      actually pins down is tau_local > 0.894 (the a_local value at
    #      the single observed a_neighbor=0-but-local-intact regime, which
    #      also needed fallback); 0.9 sits just above that with a little
    #      margin instead of exactly at the search grid's ceiling.
    deployable = cand_df[cand_df.family != 'regime-table (reference only)']
    max_util = deployable['mean_utility'].max()
    tied = deployable[deployable['mean_utility'] == max_util]
    if (tied['family'] == 'interaction-gate').any():
        inter_tied = tied[tied.family == 'interaction-gate'].copy()
        inter_tied['tau_local'] = inter_tied['params'].apply(lambda p: json.loads(p)['tau_local'])
        winner = inter_tied.sort_values(['tau_local']).iloc[0].drop(['tau_local']).to_dict()
    else:
        winner = tied.iloc[0].to_dict()
    print(f'\n  {len(tied)} candidates tied at max validation utility={max_util:.4f} '
          f'({tied.family.value_counts().to_dict()})')
    print(f'  SELECTED (tie-break: interaction-gate preferred, mechanistic argument, '
          f'see code comment): {winner}')
    with open(os.path.join(OUT_DIR, 'selected_gate.json'), 'w') as f:
        json.dump(winner, f, indent=2)
    print(f'  -> {os.path.join(OUT_DIR, "selected_gate.json")}')
    print('=' * 70)


if __name__ == '__main__':
    main()

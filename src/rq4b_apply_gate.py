"""
rq4b_apply_gate.py
=====================
RQ4b, phase 2 -- apply the VALIDATION-frozen interaction gate
(results/rq4_context_availability/gate/selected_gate.json:
tau_neighbor=0.75, tau_local=0.9, fallback = Linear interpolation) to
TEST data. No threshold, rule, or fallback choice is touched here -- this
script only evaluates the frozen policy.

  G_t = 1[(A_neighbor(t) < tau_neighbor) AND (A_local(t) < tau_local)]
  yhat_gated(t) = yhat_Linear(t) if G_t else yhat_DLPIF(t)

Part A -- graded test regimes (same local-loss/neighbour-loss/joint-loss
grid graded_context_loss.py used on test): does the validation-fit
boundary transfer to unseen test context combinations? Four systems
compared per regime: DLPIF-only, Linear-only, Gated-DLPIF, and an
Oracle-regime-selector (picks whichever of DLPIF/Linear has the higher
TEST F1 for that specific regime -- a diagnostic UPPER BOUND only, not a
deployable rule, and never fed back into the gate).

Part B -- the original 8 RQ1 test scenarios (10pct, mar_meteo, mnar_wet,
mnar_intensity_{moderate,severe}, block7d, block30d, netblock30d), same
masks/seeds/predictions used throughout RQ1-RQ3. Reports gate_trigger_rate
and DLPIF/Linear/Gated F1, precision, recall, CSI, bias, false_wet_rate,
RMSE_wet, RMSE_p95 per scenario, plus paired block-bootstrap 95% CIs
(reusing canonical_bootstrap_significance.py's exact resampling unit --
contiguous same-station masked-position runs, seed=42 "primary
realization", n_boot=10000) for Delta F1 = F1_gated - F1_DLPIF and
Delta RMSE_wet = RMSE_gated - RMSE_DLPIF.

Deliberately NOT reported here: Brier/ECE/PICP for the gated system.
Linear is a deterministic point imputer with no calibrated probability or
conformal interval, so scoring "gated Brier/ECE/PICP" would silently
average a probabilistic method's calibration against an undefined one.
RQ4b is evaluated as context-aware failure detection + point-reconstruction
recovery; probabilistic end-to-end metrics return once a fallback expert
with its own calibration/UQ exists (future ERA5-Land work).

Pre-registered success criteria (fixed BEFORE running, not fit to results):
  1. Preservation -- under ordinary MCAR/MAR/block conditions, gated-DLPIF
     must not materially degrade vs no-gate DLPIF.
  2. Recovery -- under severe joint-context loss / NetBlock-30d, gated-DLPIF
     must materially improve occurrence skill vs no-gate DLPIF.
  3. Selectivity -- the improvement must come from the gate selectively
     catching bad-context regimes (low gate_trigger_rate under ordinary
     conditions), not from blanket fallback use everywhere.
  4. No post-hoc tuning -- gate parameters are not touched based on what
     follows in this script, regardless of outcome.

Outputs (results/rq4_context_availability/gate_applied/):
  graded_test_local.csv / graded_test_neighbour.csv / graded_test_joint.csv
  original_scenarios_gated.csv
  original_scenarios_bootstrap.csv
  decision_surface_audit.csv
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import direct_two_stage_rf as d2s
import canonical_metrics as cm
from calibrate_occurrence_probability import apply_platt
from graded_context_loss import (degrade_local, degrade_neighbours,
                                 LOCAL_LEVELS, NEIGHBOUR_LEVELS, JOINT_LEVELS,
                                 CONTEXT_SEEDS, MODEL_SEEDS, prep01)
from rq4b_gate_selection import station_wise_linear_interpolation
from canonical_bootstrap_significance import paired_block_bootstrap, contiguous_blocks

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RQ2_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
GATE_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq4_context_availability', 'gate')
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq4_context_availability', 'gate_applied')
os.makedirs(OUT_DIR, exist_ok=True)

WET_THRESH = d2s.WET_THRESH
P95_THRESH = d2s.P95_THRESH
ORIGINAL_SCENARIOS = ['10pct', 'mar_meteo', 'mnar_wet',
                      'mnar_intensity_moderate', 'mnar_intensity_severe',
                      'block7d', 'block30d', 'netblock30d']

with open(os.path.join(GATE_DIR, 'selected_gate.json')) as f:
    _gate = json.load(f)
assert _gate['family'] == 'interaction-gate', f"unexpected frozen gate family: {_gate['family']}"
_gate_params = json.loads(_gate['params'])
TAU_N = _gate_params['tau_neighbor']
TAU_L = _gate_params['tau_local']
print(f'  Frozen gate (validation-selected, NOT touched here): '
      f'fallback if (A_neighbor < {TAU_N}) AND (A_local < {TAU_L})')


def gate_trigger(a_local, a_neighbor):
    return (a_neighbor < TAU_N) & (a_local < TAU_L)


def load_models(sc, mv, tr, pidx):
    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)
    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs
    X_tr_amt = d2s.build_amt_X(sc, tr, d2s.TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)
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
    return models


def dlpif_predict(corrupted, neighbor_avg, neighbor_mask, temporal, m, sc, pidx, mdl):
    X_occ = d2s.build_occ_X(corrupted, temporal, neighbor_avg, neighbor_mask, pidx)
    cor_inv = d2s.inv(sc, corrupted)
    cor_inv[:, pidx] = 0.0
    X_amt = np.concatenate([cor_inv, temporal, d2s.inv(sc, neighbor_avg), neighbor_mask], axis=1)
    raw_p = mdl['occ_rf'].predict_proba(X_occ[m])[:, 1]
    p_cal = apply_platt(mdl['platt'], raw_p)
    wet_pred = p_cal >= mdl['tau']
    yhat = np.zeros(int(m.sum()), dtype=np.float32)
    if wet_pred.sum() > 0:
        yhat[wet_pred] = mdl['amt_rf'].predict(X_amt[m][wet_pred])
    return yhat


def false_wet_rate(tp, fp, fn, tn):
    return round(fp / (fp + tn), 4) if (fp + tn) > 0 else np.nan


def three_system_metrics(y_true, dlpif_yhat, linear_yhat, gate_mask, scenario, seed, n_eval):
    gated_yhat = np.where(gate_mask, linear_yhat, dlpif_yhat)
    rows = {}
    for label, pred in [('DLPIF', dlpif_yhat), ('Linear', linear_yhat), ('Gated', gated_yhat)]:
        r = cm.metrics_row(y_true, pred, method=label, scenario=scenario, seed=seed, n_evaluated=n_eval)
        r['false_wet_rate'] = false_wet_rate(r['tp'], r['fp'], r['fn'], r['tn'])
        rows[label] = r
    rows['gate_trigger_rate'] = round(float(gate_mask.mean()), 4)
    return rows, gated_yhat


def main():
    print('=' * 70)
    print('  RQ4b phase 2 -- APPLY FROZEN GATE TO TEST (no tuning here)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    local_cols = [i for i in range(len(mv)) if i != pidx]
    tr = d2s.load_npz('train')
    te = d2s.load_npz('test')
    models = load_models(sc, mv, tr, pidx)

    with open(os.path.join(SRC_DIR, 'adjacency.pkl'), 'rb') as f:
        adj = pickle.load(f)
    A_knn = adj['A_knn']
    stations = adj['stations']

    corrupted_base = te[d2s.TRAIN_COR_KEY].astype(np.float32)
    m = te['art_mask_10pct'].astype(np.float32)[:, pidx] > 0.5
    m_rows = np.where(m)[0]
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
    station_ids = te['station_ids']
    temporal = te['temporal'].astype(np.float32)
    base_nbr_avg = te['neighbor_avg_10pct'].astype(np.float32)
    base_nbr_mask = te['neighbor_mask_10pct'].astype(np.float32)
    y_true_base = te_gt[m]
    print(f'  Test MCAR-10 masked-PRECIP population: {len(m_rows)} positions')

    decision_surface_rows = []

    # ═══ PART A: graded test regimes ═══════════════════════════════════
    def run_graded(family, levels, build_fn, level_label_fn, out_name):
        print(f'\n  --- Part A: {family}-loss ---')
        detail_rows = []
        for level in levels:
            for cseed in CONTEXT_SEEDS:
                corrupted, nbr_avg, nbr_mask, (a_local, a_neighbor) = build_fn(level, cseed)
                gate_mask = gate_trigger(a_local, a_neighbor)
                for mseed in MODEL_SEEDS:
                    dlpif_yhat = dlpif_predict(corrupted, nbr_avg, nbr_mask, temporal, m, sc, pidx, models[mseed])
                    lin = station_wise_linear_interpolation(corrupted, station_ids)
                    linear_yhat = d2s.inv(sc, lin)[:, pidx][m]
                    metrics, gated_yhat = three_system_metrics(
                        y_true_base, dlpif_yhat, linear_yhat, gate_mask,
                        f'{family}_{level_label_fn(level)}', mseed, len(y_true_base))
                    row = dict(family=family, level=level_label_fn(level),
                              context_seed=cseed, model_seed=mseed,
                              a_local=round(float(np.mean(a_local)), 4),
                              a_neighbor=round(float(np.mean(a_neighbor)), 4),
                              gate_trigger_rate=metrics['gate_trigger_rate'])
                    for sysname in ['DLPIF', 'Linear', 'Gated']:
                        for k in ['f1', 'precision', 'recall', 'csi', 'signed_frequency_bias',
                                 'false_wet_rate', 'rmse_wet', 'rmse_p95']:
                            row[f'{sysname.lower()}_{k}'] = metrics[sysname][k]
                    detail_rows.append(row)
            print(f'    level={level_label_fn(level)} done')

        detail_df = pd.DataFrame(detail_rows)

        # Oracle regime-selector: per (family, level), pick whichever of
        # DLPIF/Linear had the higher mean TEST F1 -- diagnostic upper
        # bound only, computed AFTER the fact, never fed back into the gate.
        oracle_rows = []
        for level_lbl, grp in detail_df.groupby('level'):
            dlpif_f1 = grp['dlpif_f1'].mean()
            linear_f1 = grp['linear_f1'].mean()
            better = 'DLPIF' if dlpif_f1 >= linear_f1 else 'Linear'
            oracle_rows.append(dict(level=level_lbl, oracle_choice=better,
                                    oracle_f1=grp[f'{better.lower()}_f1'].mean(),
                                    oracle_rmse_wet=grp[f'{better.lower()}_rmse_wet'].mean()))
        oracle_df = pd.DataFrame(oracle_rows)

        detail_path = os.path.join(OUT_DIR, out_name)
        detail_df.to_csv(detail_path, index=False)
        print(f'  -> {detail_path}  ({len(detail_df)} rows)')

        summary = detail_df.groupby('level')[
            ['a_local', 'a_neighbor', 'gate_trigger_rate',
            'dlpif_f1', 'linear_f1', 'gated_f1',
            'dlpif_rmse_wet', 'linear_rmse_wet', 'gated_rmse_wet']
        ].agg(['mean', 'std'])
        print(summary.round(4))
        print('  Oracle (diagnostic upper bound, test-informed, NOT a deployable rule):')
        print(oracle_df.round(4).to_string(index=False))

        for _, r in decision_surface_from(detail_df).iterrows():
            decision_surface_rows.append(r.to_dict())

        return detail_df, oracle_df

    def decision_surface_from(detail_df):
        d = detail_df.copy()
        d['a_local_r'] = d['a_local'].round(2)
        d['a_neighbor_r'] = d['a_neighbor'].round(2)
        agg = d.groupby(['a_local_r', 'a_neighbor_r'])['gate_trigger_rate'].agg(['mean', 'count']).reset_index()
        agg['action'] = np.where(agg['mean'] > 0.5, 'Linear (gate triggers)', 'DLPIF (gate does not trigger)')
        agg = agg.rename(columns={'a_local_r': 'a_local', 'a_neighbor_r': 'a_neighbor',
                                  'mean': 'trigger_rate', 'count': 'n_trials'})
        agg['source'] = 'graded_test'
        return agg

    def build_local(keep, cseed):
        rng = np.random.default_rng(cseed * 1000 + keep)
        cor_mod = degrade_local(corrupted_base, m_rows, local_cols, keep, rng)
        a_local = np.mean(~np.isnan(cor_mod[m][:, local_cols]), axis=1)
        a_nbr = np.mean(base_nbr_mask[m], axis=1)
        return cor_mod, base_nbr_avg, base_nbr_mask, (a_local, a_nbr)
    local_detail, local_oracle = run_graded('local', LOCAL_LEVELS, build_local, lambda k: str(k),
                                            'graded_test_local.csv')

    def build_nbr(keep, cseed):
        rng = np.random.default_rng(cseed * 1000 + 500 + keep)
        A_mod = degrade_neighbours(A_knn, stations, keep, rng)
        nbr_avg_mod, nbr_mask_mod = prep01.compute_neighbor_avg(corrupted_base, station_ids, A_mod, stations)
        a_local = np.mean(~np.isnan(corrupted_base[m][:, local_cols]), axis=1)
        a_nbr = np.mean(nbr_mask_mod[m], axis=1)
        return corrupted_base, nbr_avg_mod, nbr_mask_mod, (a_local, a_nbr)
    nbr_detail, nbr_oracle = run_graded('neighbour', NEIGHBOUR_LEVELS, build_nbr, lambda k: str(k),
                                        'graded_test_neighbour.csv')

    def build_joint(pair, cseed):
        keep_local, keep_nbr = pair
        rng_l = np.random.default_rng(cseed * 1000 + 900 + keep_local)
        rng_n = np.random.default_rng(cseed * 1000 + 950 + keep_nbr)
        cor_mod = degrade_local(corrupted_base, m_rows, local_cols, keep_local, rng_l)
        A_mod = degrade_neighbours(A_knn, stations, keep_nbr, rng_n)
        nbr_avg_mod, nbr_mask_mod = prep01.compute_neighbor_avg(cor_mod, station_ids, A_mod, stations)
        a_local = np.mean(~np.isnan(cor_mod[m][:, local_cols]), axis=1)
        a_nbr = np.mean(nbr_mask_mod[m], axis=1)
        return cor_mod, nbr_avg_mod, nbr_mask_mod, (a_local, a_nbr)
    joint_detail, joint_oracle = run_graded('joint', JOINT_LEVELS, build_joint,
                                            lambda p: f'{p[0]}L+{p[1]}N', 'graded_test_joint.csv')

    # ═══ PART B: original 8 RQ1 test scenarios ═════════════════════════
    print(f'\n{"="*70}\n  Part B: original 8 test scenarios\n{"="*70}')
    with open(os.path.join(SRC_DIR, 'baseline_results.pkl'), 'rb') as f:
        bl_all = pickle.load(f)['all_scenarios']

    scenario_rows = []
    bootstrap_rows = []
    scen_key_map = {label: (ck, mk) for label, ck, mk in d2s.SCENARIOS}

    for scen in ORIGINAL_SCENARIOS:
        cor_key, mask_key = scen_key_map[scen]
        na_key, nm_key = d2s.neighbor_keys(scen)
        corrupted = te[cor_key].astype(np.float32)
        nbr_avg = te[na_key].astype(np.float32)
        nbr_mask = te[nm_key].astype(np.float32)
        m_scen = te[mask_key].astype(np.float32)[:, pidx] > 0.5
        y_true = te_gt[m_scen]
        n_eval = int(m_scen.sum())

        a_local = np.mean(~np.isnan(corrupted[m_scen][:, local_cols]), axis=1)
        a_neighbor = np.mean(nbr_mask[m_scen], axis=1)
        gate_mask = gate_trigger(a_local, a_neighbor)

        lin_norm = np.clip(bl_all[('linear', scen)].astype(np.float32), 0, 1)
        linear_yhat = d2s.inv(sc, lin_norm)[:, pidx][m_scen]

        for mseed in MODEL_SEEDS:
            dlpif_yhat = dlpif_predict(corrupted, nbr_avg, nbr_mask, temporal, m_scen, sc, pidx, models[mseed])
            metrics, gated_yhat = three_system_metrics(y_true, dlpif_yhat, linear_yhat, gate_mask,
                                                        scen, mseed, n_eval)
            row = dict(scenario=scen, model_seed=mseed,
                      n_evaluated=n_eval, gate_trigger_rate=metrics['gate_trigger_rate'],
                      mean_a_local=round(float(a_local.mean()), 4),
                      mean_a_neighbor=round(float(a_neighbor.mean()), 4))
            for sysname in ['DLPIF', 'Linear', 'Gated']:
                for k in ['f1', 'precision', 'recall', 'csi', 'signed_frequency_bias',
                         'false_wet_rate', 'rmse_wet', 'rmse_p95']:
                    row[f'{sysname.lower()}_{k}'] = metrics[sysname][k]
            row['delta_f1'] = round(metrics['Gated']['f1'] - metrics['DLPIF']['f1'], 4)
            row['delta_rmse_wet'] = (round(metrics['Gated']['rmse_wet'] - metrics['DLPIF']['rmse_wet'], 4)
                                     if metrics['Gated']['rmse_wet'] == metrics['Gated']['rmse_wet']
                                     and metrics['DLPIF']['rmse_wet'] == metrics['DLPIF']['rmse_wet'] else np.nan)
            scenario_rows.append(row)

            if mseed == 42:
                block_df = pd.DataFrame(dict(
                    row_index=np.where(m_scen)[0], station_id=station_ids[m_scen])).sort_values(
                    ['station_id', 'row_index'])
                order = block_df.index.to_numpy()
                block_id = contiguous_blocks(block_df.reset_index(drop=True))
                gt_ord = y_true[order]
                dlpif_ord = dlpif_yhat[order]
                gated_ord = gated_yhat[order]

                d_f1, f1_lo, f1_hi, f1_p = paired_block_bootstrap(
                    gt_ord, dlpif_ord, gt_ord, gated_ord, block_id,
                    lambda g, p: cm.precip_metrics(g, p)['f1'])
                d_rmse, rmse_lo, rmse_hi, rmse_p = paired_block_bootstrap(
                    gt_ord, dlpif_ord, gt_ord, gated_ord, block_id,
                    lambda g, p: cm.precip_metrics(g, p)['rmse_wet'])
                bootstrap_rows.append(dict(
                    scenario=scen, seed=42,
                    delta_f1=round(d_f1, 4), delta_f1_ci_low=round(f1_lo, 4),
                    delta_f1_ci_high=round(f1_hi, 4), delta_f1_p=round(f1_p, 4),
                    delta_rmse_wet=round(d_rmse, 4), delta_rmse_wet_ci_low=round(rmse_lo, 4),
                    delta_rmse_wet_ci_high=round(rmse_hi, 4), delta_rmse_wet_p=round(rmse_p, 4),
                ))
        print(f'    [{scen:24s}] gate_trigger_rate={metrics["gate_trigger_rate"]:.3f}  '
              f'F1: DLPIF={metrics["DLPIF"]["f1"]:.3f} Gated={metrics["Gated"]["f1"]:.3f}')

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(os.path.join(OUT_DIR, 'original_scenarios_gated.csv'), index=False)
    print(f'\n  -> {os.path.join(OUT_DIR, "original_scenarios_gated.csv")}')

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(os.path.join(OUT_DIR, 'original_scenarios_bootstrap.csv'), index=False)
    print(f'  -> {os.path.join(OUT_DIR, "original_scenarios_bootstrap.csv")}')

    d = scenario_df.copy()
    d['a_local_r'] = d['mean_a_local'].round(2)
    d['a_neighbor_r'] = d['mean_a_neighbor'].round(2)
    agg = d.groupby(['scenario', 'a_local_r', 'a_neighbor_r'])['gate_trigger_rate'].mean().reset_index()
    agg['action'] = np.where(agg['gate_trigger_rate'] > 0.5, 'Linear (gate triggers)', 'DLPIF (gate does not trigger)')
    agg = agg.rename(columns={'a_local_r': 'a_local', 'a_neighbor_r': 'a_neighbor'})
    agg['source'] = 'original_8_scenarios'
    for _, r in agg.iterrows():
        decision_surface_rows.append(r.to_dict())

    surface_df = pd.DataFrame(decision_surface_rows)
    surface_df.to_csv(os.path.join(OUT_DIR, 'decision_surface_audit.csv'), index=False)
    print(f'  -> {os.path.join(OUT_DIR, "decision_surface_audit.csv")}  ({len(surface_df)} rows)')

    print('\n  Summary (mean over 3 model seeds), F1 DLPIF vs Gated:')
    summ = scenario_df.groupby('scenario')[
        ['gate_trigger_rate', 'dlpif_f1', 'gated_f1', 'delta_f1',
        'dlpif_rmse_wet', 'gated_rmse_wet', 'delta_rmse_wet']
    ].mean().reindex(ORIGINAL_SCENARIOS)
    print(summ.round(4))
    print('\n  Bootstrap (seed=42, n_boot=10000):')
    print(bootstrap_df[['scenario', 'delta_f1', 'delta_f1_ci_low', 'delta_f1_ci_high', 'delta_f1_p']].to_string(index=False))
    print('=' * 70)


if __name__ == '__main__':
    main()

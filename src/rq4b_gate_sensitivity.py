# -*- coding: utf-8 -*-
"""
rq4b_gate_sensitivity.py
===========================
R6 follow-up -- Section 5.5 reports that 28 of 81 evaluated gate
candidates (24 scalar-gate, 4 interaction-gate) tie the regime-table
oracle's VALIDATION utility exactly, and that the interaction-gate family
(tau_neighbor=0.75, tau_local=0.9) is selected from this tie on
mechanistic grounds (Section 5.5) -- not because it scored higher than
the tied alternatives. This script answers the natural follow-up
question a reviewer would ask: how much does TEST performance actually
vary across those 28 tied candidates? If it varies a lot, the tie-break
rationale matters a great deal; if the tied candidates behave similarly
on held-out test data too, the tie-break is close to a formality.

This is evaluated ONLY on the eight original RQ1 test scenarios (not the
graded test regimes, which rq4b_apply_gate.py already covers for the
single selected gate) and reports point estimates (gate_trigger_rate,
Gated F1, Delta-F1, Delta-RMSE_wet vs. no-gate DLPIF) per candidate per
scenario, mean over 3 model seeds -- NOT a full block-bootstrap re-run
per candidate (224 candidate x scenario combinations would make that
prohibitively slow for a sensitivity check whose purpose is showing a
range, not re-establishing significance for each one individually; the
single selected gate's own bootstrap CIs are unchanged and remain in
Table 6.10 / gate_applied/original_scenarios_bootstrap.csv).

No new models are trained here -- the frozen Stage 1/Stage 2 models
already used throughout RQ1-RQ4 are reused via rq4b_apply_gate.py's own
load_models()/dlpif_predict()/three_system_metrics() helpers.

Output: results/rq4_context_availability/gate/candidate_sensitivity.csv
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

import direct_two_stage_rf as d2s
import rq4b_apply_gate as rag  # reuses load_models, dlpif_predict, three_system_metrics

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
GATE_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq4_context_availability', 'gate')
OUT_PATH = os.path.join(GATE_DIR, 'candidate_sensitivity.csv')

ORIGINAL_SCENARIOS = rag.ORIGINAL_SCENARIOS
MODEL_SEEDS = [42, 123, 456]


def make_gate_fn(family, params):
    if family == 'scalar-gate':
        w_l, w_n, tau = params['w_local'], params['w_neighbor'], params['tau']
        def gate(a_local, a_neighbor):
            a_star = w_l * a_local + w_n * a_neighbor
            return a_star < tau
        return gate
    elif family == 'interaction-gate':
        tau_n, tau_l = params['tau_neighbor'], params['tau_local']
        def gate(a_local, a_neighbor):
            return (a_neighbor < tau_n) & (a_local < tau_l)
        return gate
    else:
        raise ValueError(family)


def main():
    print('=' * 70)
    print('  R6 FOLLOW-UP -- SENSITIVITY OF TEST PERFORMANCE ACROSS THE 28 TIED GATE CANDIDATES')
    print('=' * 70)

    cand_df = pd.read_csv(os.path.join(GATE_DIR, 'gate_candidates_ranked.csv'))
    cand_df = cand_df[cand_df.family != 'regime-table (reference only)'].copy()
    tied = cand_df[cand_df.mean_utility == cand_df.mean_utility.max()].reset_index(drop=True)
    print(f'  {len(tied)} tied candidates loaded ({(tied.family=="scalar-gate").sum()} scalar-gate, '
         f'{(tied.family=="interaction-gate").sum()} interaction-gate)')

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    local_cols = [i for i in range(len(mv)) if i != pidx]
    tr = d2s.load_npz('train')
    te = d2s.load_npz('test')
    models = rag.load_models(sc, mv, tr, pidx)
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
    temporal = te['temporal'].astype(np.float32)

    with open(os.path.join(SRC_DIR, 'baseline_results.pkl'), 'rb') as f:
        bl_all = pickle.load(f)['all_scenarios']
    scen_key_map = {label: (ck, mk) for label, ck, mk in d2s.SCENARIOS}

    # -- precompute, per scenario, the quantities every candidate reuses --
    scenario_cache = {}
    for scen in ORIGINAL_SCENARIOS:
        cor_key, mask_key = scen_key_map[scen]
        na_key, nm_key = d2s.neighbor_keys(scen)
        corrupted = te[cor_key].astype(np.float32)
        nbr_avg = te[na_key].astype(np.float32)
        nbr_mask = te[nm_key].astype(np.float32)
        m_scen = te[mask_key].astype(np.float32)[:, pidx] > 0.5
        y_true = te_gt[m_scen]
        a_local = np.mean(~np.isnan(corrupted[m_scen][:, local_cols]), axis=1)
        a_neighbor = np.mean(nbr_mask[m_scen], axis=1)
        lin_norm = np.clip(bl_all[('linear', scen)].astype(np.float32), 0, 1)
        linear_yhat = d2s.inv(sc, lin_norm)[:, pidx][m_scen]
        dlpif_yhat_by_seed = {
            seed: rag.dlpif_predict(corrupted, nbr_avg, nbr_mask, temporal, m_scen, sc, pidx, models[seed])
            for seed in MODEL_SEEDS
        }
        scenario_cache[scen] = dict(y_true=y_true, a_local=a_local, a_neighbor=a_neighbor,
                                    linear_yhat=linear_yhat, dlpif_yhat_by_seed=dlpif_yhat_by_seed,
                                    n_eval=int(m_scen.sum()))

    rows = []
    for _, cand in tied.iterrows():
        family = cand['family']
        params = json.loads(cand['params'])
        gate_fn = make_gate_fn(family, params)
        for scen in ORIGINAL_SCENARIOS:
            c = scenario_cache[scen]
            gate_mask = gate_fn(c['a_local'], c['a_neighbor'])
            trigger_rate = round(float(gate_mask.mean()), 4)
            f1_dlpif_list, f1_gated_list, rmse_dlpif_list, rmse_gated_list = [], [], [], []
            for seed in MODEL_SEEDS:
                metrics, gated_yhat = rag.three_system_metrics(
                    c['y_true'], c['dlpif_yhat_by_seed'][seed], c['linear_yhat'], gate_mask, scen, seed, c['n_eval'])
                f1_dlpif_list.append(metrics['DLPIF']['f1']); f1_gated_list.append(metrics['Gated']['f1'])
                rmse_dlpif_list.append(metrics['DLPIF']['rmse_wet']); rmse_gated_list.append(metrics['Gated']['rmse_wet'])
            rows.append(dict(
                family=family, params=cand['params'], scenario=scen,
                gate_trigger_rate=trigger_rate,
                dlpif_f1=round(float(np.mean(f1_dlpif_list)), 4),
                gated_f1=round(float(np.mean(f1_gated_list)), 4),
                delta_f1=round(float(np.mean(f1_gated_list)) - float(np.mean(f1_dlpif_list)), 4),
                dlpif_rmse_wet=round(float(np.nanmean(rmse_dlpif_list)), 4),
                gated_rmse_wet=round(float(np.nanmean(rmse_gated_list)), 4),
                delta_rmse_wet=round(float(np.nanmean(rmse_gated_list)) - float(np.nanmean(rmse_dlpif_list)), 4),
            ))
        print(f'  [{family:16s} {cand["params"]}] done')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f'\n  -> {OUT_PATH}')

    print('\n  Range of Delta-F1 across the 28 tied candidates, per scenario:')
    rng = df.groupby('scenario')['delta_f1'].agg(['min', 'max', 'mean', 'std']).reindex(ORIGINAL_SCENARIOS)
    print(rng.round(4))
    print('\n  Selected (frozen) interaction-gate (tau_neighbor=0.75, tau_local=0.9) row-by-row:')
    sel = df[df.params == '{"tau_neighbor": 0.75, "tau_local": 0.9}']
    print(sel[['scenario', 'gate_trigger_rate', 'delta_f1', 'delta_rmse_wet']].to_string(index=False))
    print('=' * 70)


if __name__ == '__main__':
    main()

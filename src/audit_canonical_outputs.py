# -*- coding: utf-8 -*-
"""
audit_canonical_outputs.py
============================
Independent audit of results/canonical/{predictions,metrics}/, produced by
build_canonical_outputs.py. Wherever a check can be re-derived from the
raw source files (preprocessed_test.npz, scaler.pkl, the .npy/.pkl
prediction artifacts) instead of trusting the canonical CSVs themselves,
it is -- an audit that only re-reads its own output proves nothing.

This script reads ONLY:
  - preprocessed_test.npz, scaler.pkl (the fresh snapshot)
  - baseline_results.pkl, gan_imputed_test_modeB_*.npy,
    direct_two_stage_rf_test_seed*_*.npy, single_stage_rf_test_seed*_*.npy
    (this session's raw prediction artifacts -- the same files
    build_canonical_outputs.py reads)
  - results/canonical/predictions/*.csv, results/canonical/metrics/*.csv
    (the canonical outputs being audited)
No results/*.csv or src/*.csv from any OTHER script's own metric
implementation (multiseed_clean_evaluation.csv, clean_full_evaluation.csv,
metrics_seed*.csv, summary_mean_std*.csv, analysis_*.csv/md,
occurrence_clean_seed_summary.csv, in either results/ or src/) is read --
see the no_legacy_file_read check, which enforces this as a whitelist,
not just a claim.

Output: results/canonical/canonical_audit.csv
Columns: audit_category, check_name, method, scenario, seed,
         status, observed, expected, tolerance, source_file, message

Target: 0 FAIL, 0 WARN.
"""
import os
import re
import warnings
import numpy as np
import pandas as pd

import canonical_metrics as cm
import direct_two_stage_rf as d2s
import build_canonical_outputs as bco

warnings.filterwarnings('ignore')

SRC_DIR     = d2s.SRC_DIR
REPO_ROOT   = d2s.REPO_ROOT
CANON_DIR   = bco.CANON_DIR
PRED_DIR    = bco.PRED_DIR
METRICS_DIR = bco.METRICS_DIR
SEEDS       = bco.SEEDS
SCENARIOS   = bco.SCENARIOS
METHODS     = bco.METHODS

# Whitelist of source_file patterns allowed to appear in predictions/*.csv.
# Anything NOT matching one of these is presumed to be a legacy/archive
# read and fails the audit immediately.
ALLOWED_SOURCE_PATTERNS = [
    re.compile(r'baseline_results\.pkl::all_scenarios\[\("(linear|mean|knn|mice)", "\w+"\)\]$'),
    re.compile(r'gan_imputed_test_modeB_seed\d+_msclean_amountrf_\w+\.npy$'),
    re.compile(r'direct_two_stage_rf_test_seed\d+_\w+\.npy$'),
    re.compile(r'single_stage_rf_test_seed\d+_\w+\.npy$'),
    re.compile(r'missingness_indicator_rf_test_seed\d+_\w+\.npy$'),
    re.compile(r'saits_test_seed\d+_\w+\.npy$'),
    re.compile(r'gan_imputed_test_modeB_\w+_seed\d+\.npy$'),
    re.compile(r'gan_imputed_test_modeB_\w+_seed\d+_precipfix\.npy$'),
]
# Explicit blocklist (defence in depth -- these must never appear even if a
# future pattern edit accidentally widens ALLOWED_SOURCE_PATTERNS).
LEGACY_BASENAMES = {
    'multiseed_clean_evaluation.csv', 'clean_full_evaluation.csv',
    'occurrence_clean_seed_summary.csv', 'summary_mean_std.csv',
    'summary_mean_std_wide.md', 'metrics_seed42.csv', 'metrics_seed123.csv',
    'metrics_seed456.csv', 'analysis_false_wet_rate.csv',
    'analysis_drizzle_distribution.csv', 'analysis_dry_spells.csv',
    'analysis_dry_spells_summary.md', 'analysis_spatial_correlation.csv',
    'analysis_spatial_correlation_summary.md',
}

WET_THRESH = cm.WET_THRESH
ROWS = []


def add(category, check_name, status, method='', scenario='', seed='',
       observed='', expected='', tolerance='', source_file='', message=''):
    ROWS.append(dict(
        audit_category=category, check_name=check_name,
        method=method, scenario=scenario, seed=seed,
        status=status, observed=observed, expected=expected,
        tolerance=tolerance, source_file=source_file, message=message,
    ))


def load_predictions(method):
    path = os.path.join(PRED_DIR, f'{method}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'canonical predictions file missing: {path}')
    return pd.read_csv(path, dtype={'seed': str})


# ── 1/2/3/4: completeness + integrity, per method ───────────────────────────

def audit_completeness_and_integrity(sc, pidx, te):
    for method, (seeds, _loader, method_scenarios) in METHODS.items():
        df = load_predictions(method)
        expected_groups = {(scen, str(s)) for s in seeds for scen, _, _ in method_scenarios}
        actual_groups = set(zip(df['scenario'], df['seed'].astype(str)))

        # 1) all expected seed/scenario groups exist
        missing = expected_groups - actual_groups
        add('completeness', 'expected_groups_exist',
            'PASS' if not missing else 'FAIL',
            method=method, observed=len(actual_groups), expected=len(expected_groups),
            message=('all groups present' if not missing
                     else f'missing groups: {sorted(missing)}'))

        extra = actual_groups - expected_groups
        add('completeness', 'no_unexpected_groups',
            'PASS' if not extra else 'FAIL',
            method=method, observed=len(actual_groups), expected=len(expected_groups),
            message=('no unexpected groups' if not extra
                     else f'unexpected groups: {sorted(extra)}'))

        for scen_label, _cor_key, mask_key in method_scenarios:
            fresh_mask = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            fresh_n = int(fresh_mask.sum())

            for seed in seeds:
                seed_str = str(seed)
                grp = df[(df['scenario'] == scen_label) & (df['seed'].astype(str) == seed_str)]

                # 2) no duplicate method/seed/scenario prediction rows
                dup = grp.duplicated(subset=['row_index']).sum()
                add('completeness', 'no_duplicate_rows',
                    'PASS' if dup == 0 else 'FAIL',
                    method=method, scenario=scen_label, seed=seed_str,
                    observed=int(dup), expected=0,
                    message='no duplicate row_index within group' if dup == 0
                            else f'{dup} duplicate row_index values')

                # 3) prediction and mask lengths match (recomputed fresh)
                add('integrity', 'prediction_mask_length_match',
                    'PASS' if len(grp) == fresh_n else 'FAIL',
                    method=method, scenario=scen_label, seed=seed_str,
                    observed=len(grp), expected=fresh_n,
                    message='row count matches fresh art_mask.sum()' if len(grp) == fresh_n
                            else 'row count does not match freshly recomputed mask count')

                # 4) no NaN/Inf on evaluated cells
                bad = (~np.isfinite(grp['y_true'])).sum() + (~np.isfinite(grp['y_pred'])).sum()
                add('integrity', 'no_nan_inf_evaluated',
                    'PASS' if bad == 0 else 'FAIL',
                    method=method, scenario=scen_label, seed=seed_str,
                    observed=int(bad), expected=0,
                    message='all evaluated cells finite' if bad == 0
                            else f'{bad} non-finite values found')

                # row_index values must all fall inside the fresh mask
                if len(grp):
                    idx_ok = fresh_mask[grp['row_index'].to_numpy()].all()
                    add('integrity', 'row_index_matches_fresh_mask',
                        'PASS' if idx_ok else 'FAIL',
                        method=method, scenario=scen_label, seed=seed_str,
                        observed=bool(idx_ok), expected=True,
                        message='every row_index is a true art_mask position' if idx_ok
                                else 'some row_index values are NOT masked in the fresh art_mask')


# WGAN-GP backbone-integrity checks (5/6/7) removed 2026-07-23 along with
# WGAN-GP itself (no GPU available to retrain on the post-2005 dataset;
# see build_canonical_outputs.py's module docstring).

# ── 8/9/12: metrics consistency + traceability ──────────────────────────────

def audit_metrics_consistency(sc, pidx, te):
    metrics_path = os.path.join(METRICS_DIR, 'canonical_metrics_all.csv')
    df_metrics = pd.read_csv(metrics_path, dtype={'seed': str})

    numeric_fields = ['freq_gt', 'freq_pred', 'signed_frequency_bias',
                      'precision', 'recall', 'f1', 'csi',
                      'rmse', 'mae', 'rmse_wet', 'mae_wet', 'mae_p95', 'rmse_p95',
                      'tp', 'fp', 'fn', 'tn']
    # Predictions are full float64 precision (see build_canonical_outputs.py)
    # and recomputed via the SAME canonical_metrics.metrics_row() call used
    # to build the stored metrics, so an exact match (down to float64
    # rounding noise) is expected for every field -- not just the
    # already-rounded display fields.
    tol_by_field = {f: 1e-9 for f in numeric_fields}

    pred_cache = {}

    for _, mrow in df_metrics.iterrows():
        method, scenario, seed = mrow['method'], mrow['scenario'], str(mrow['seed'])

        # 12) traceability: source_prediction_file exists and matches this method
        src_rel = mrow['source_prediction_file']
        src_abs = os.path.join(REPO_ROOT, src_rel.replace('/', os.sep))
        expected_rel = 'results/canonical/predictions/' + f'{method}.csv'
        traceable = os.path.exists(src_abs) and os.path.normpath(src_rel) == os.path.normpath(expected_rel)
        add('traceability', 'metrics_row_traceability',
            'PASS' if traceable else 'FAIL',
            method=method, scenario=scenario, seed=seed,
            source_file=src_rel,
            message='source_prediction_file exists and matches method' if traceable
                    else 'source_prediction_file missing or mismatched')

        # 8) n_evaluated matches artificial precipitation-mask count (fresh)
        mask_key = dict((s, mk) for s, _c, mk in SCENARIOS)[scenario]
        fresh_n = int((te[mask_key].astype(np.float32)[:, pidx] > 0.5).sum())
        add('integrity', 'n_evaluated_matches_mask_count',
            'PASS' if int(mrow['n_evaluated']) == fresh_n else 'FAIL',
            method=method, scenario=scenario, seed=seed,
            observed=int(mrow['n_evaluated']), expected=fresh_n,
            message='n_evaluated matches freshly recomputed art_mask count'
                    if int(mrow['n_evaluated']) == fresh_n else 'n_evaluated mismatch')

        # 9) metrics recomputed from predictions exactly match saved metrics
        if method not in pred_cache:
            pred_cache[method] = load_predictions(method)
        pdf = pred_cache[method]
        grp = pdf[(pdf['scenario'] == scenario) & (pdf['seed'].astype(str) == seed)]
        recomputed = cm.metrics_row(grp['y_true'].to_numpy(), grp['y_pred'].to_numpy(),
                                    method, scenario, seed=seed, n_evaluated=len(grp))

        worst_field, worst_diff = None, 0.0
        for f in numeric_fields:
            a, b = mrow[f], recomputed[f]
            if pd.isna(a) and pd.isna(b):
                d = 0.0
            elif pd.isna(a) or pd.isna(b):
                d = float('inf')
            else:
                d = abs(float(a) - float(b))
            if d > worst_diff:
                worst_diff, worst_field = d, f
        tol = max(tol_by_field.values())
        status = 'PASS' if worst_diff <= tol else 'FAIL'
        add('consistency', 'metrics_recompute_match', status,
            method=method, scenario=scenario, seed=seed,
            observed=round(worst_diff, 6), expected=0.0, tolerance=tol,
            source_file=os.path.join(PRED_DIR, f'{method}.csv'),
            message=(f'recomputed metrics match saved metrics (worst field: {worst_field})'
                     if status == 'PASS' else
                     f'largest mismatch on field "{worst_field}": diff={worst_diff}'))


# ── 10: no legacy/archive file read ─────────────────────────────────────────

def audit_no_legacy_reads():
    for method in METHODS:
        df = load_predictions(method)
        distinct_sources = df['source_file'].unique().tolist()
        bad = []
        for src in distinct_sources:
            base = os.path.basename(str(src).split('::')[0])
            if base in LEGACY_BASENAMES:
                bad.append(src)
                continue
            if not any(p.search(str(src)) for p in ALLOWED_SOURCE_PATTERNS):
                bad.append(src)
        add('provenance', 'no_legacy_file_read',
            'PASS' if not bad else 'FAIL',
            method=method, observed=len(distinct_sources), expected=len(distinct_sources),
            message=('all source_file values match the allowed whitelist' if not bad
                     else f'non-whitelisted / legacy sources found: {bad}'))


# ── 11: DLPIF vs DirectTwoStageRF equivalence ───────────────────────────────

def audit_dlpif_vs_direct_equivalence():
    a = load_predictions('AmountRF_DLPIF')
    b = load_predictions('DirectTwoStageRF')
    merged = a.merge(b, on=['scenario', 'seed', 'row_index'], suffixes=('_dlpif', '_direct'))

    # AmountRF_DLPIF has no netblock30d reconstruction (Section 5.8's
    # equivalence claim was established, and remains true, only for the
    # scenarios both methods were actually run on); use its own registered
    # scenario list rather than the full SCENARIOS so this check does not
    # spuriously fail on a scenario neither side has data for.
    dlpif_scenarios = METHODS['AmountRF_DLPIF'][2]
    for scen_label, _c, _m in dlpif_scenarios:
        for seed in SEEDS:
            seed_str = str(seed)
            sub = merged[(merged['scenario'] == scen_label) & (merged['seed'].astype(str) == seed_str)]
            if sub.empty:
                add('equivalence', 'dlpif_vs_direct_equivalence', 'FAIL',
                    method='AmountRF_DLPIF_vs_DirectTwoStageRF', scenario=scen_label, seed=seed_str,
                    message='no overlapping rows found -- cannot check equivalence')
                continue
            diff = (sub['y_pred_dlpif'] - sub['y_pred_direct']).abs()
            true_diff = (sub['y_true_dlpif'] - sub['y_true_direct']).abs()
            max_diff = float(diff.max())
            max_true_diff = float(true_diff.max())
            # Predictions are written at full float64 precision (see
            # build_canonical_outputs.py's ROW_DECIMALS docstring note), so
            # two independently-trained-but-architecturally-identical
            # pipelines are expected to match to within ordinary float64
            # noise, not a rounding-derived tolerance.
            tol = 1e-9
            status = 'PASS' if max_diff <= tol and max_true_diff <= tol else 'FAIL'
            add('equivalence', 'dlpif_vs_direct_equivalence', status,
                method='AmountRF_DLPIF_vs_DirectTwoStageRF', scenario=scen_label, seed=seed_str,
                observed=round(max_diff, 8), expected=0.0, tolerance=tol,
                message=(f'{len(sub):,} row_index-matched positions; max|y_pred diff|={max_diff}, '
                         f'max|y_true diff|={max_true_diff} (backbone-discarded equivalence, '
                         f'manuscript Section 3.4 / 5.8)'))


# ── MAIN ──────────────────────────────────────────────────────────────────────

# ── Stage 0 analyses (Tables 4/7/8 sources): completeness + recompute ──────

def audit_stage0_analyses(sc, pidx, te):
    import canonical_false_wet_analysis as cfwa
    import canonical_dry_spell as cds
    import canonical_spatial_correlation as csc

    dates = pd.to_datetime(te['dates'])
    sids = te['station_ids']
    y_true_full = sc.inverse_transform(
        np.nan_to_num(te['data'].astype(np.float64), nan=0.0))[:, pidx]
    analysis_dir = os.path.join(CANON_DIR, 'analysis')

    def check_completeness(df, name, path):
        for method, (seeds, _loader, method_scenarios) in METHODS.items():
            sub = df[df['method'] == method]
            expected_groups = {(scen, str(s)) for s in seeds for scen, _c, _m in method_scenarios}
            actual_groups = set(zip(sub['scenario'], sub['seed'].astype(str)))
            missing = expected_groups - actual_groups
            add('completeness', f'{name}_groups_exist',
                'PASS' if not missing else 'FAIL',
                method=method, observed=len(actual_groups), expected=len(expected_groups),
                source_file=path,
                message='all groups present' if not missing else f'missing: {sorted(missing)}')

    # -- Table 4: false_wet_rate.csv --
    fwr_path = os.path.join(analysis_dir, 'false_wet_rate.csv')
    fwr = pd.read_csv(fwr_path, dtype={'seed': str})
    check_completeness(fwr, 'table4', fwr_path)

    pred_cache = {}
    for _, row in fwr.iterrows():
        method = row['method']
        if method not in pred_cache:
            pred_cache[method] = pd.read_csv(os.path.join(PRED_DIR, f'{method}.csv'),
                                             dtype={'seed': str})
        pdf = pred_cache[method]
        grp = pdf[(pdf['scenario'] == row['scenario']) & (pdf['seed'].astype(str) == str(row['seed']))]
        recomputed = cfwa.compute_group(grp)
        diffs = []
        for k in ('n_true_dry', 'false_wet_rate', 'mean_pred_true_dry', 'p95_pred_true_dry'):
            a, b = row[k], recomputed[k]
            if pd.isna(a) and pd.isna(b):
                continue
            diffs.append(1.0 if (pd.isna(a) or pd.isna(b)) else abs(float(a) - float(b)))
        worst = max(diffs) if diffs else 0.0
        status = 'PASS' if worst < 1e-6 else 'FAIL'
        add('consistency', 'table4_recompute_match', status,
            method=method, scenario=row['scenario'], seed=str(row['seed']),
            observed=round(worst, 8), expected=0.0, tolerance=1e-6, source_file=fwr_path,
            message='recomputed matches stored' if status == 'PASS' else f'diff={worst}')

    # -- Table 7: dry_spells.csv (recompute per group via build_filled_series) --
    ds_path = os.path.join(analysis_dir, 'dry_spells.csv')
    ds = pd.read_csv(ds_path, dtype={'seed': str})
    check_completeness(ds.drop_duplicates(subset=['method', 'scenario', 'seed']), 'table7', ds_path)

    for method, (seeds, _loader, method_scenarios) in METHODS.items():
        for seed in seeds:
            for scen_label, _c, _m in method_scenarios:
                series = bco.build_filled_series(method, seed, scen_label, sc, pidx,
                                                 te, dates, sids, y_true_full)
                series = series.sort_values(['station_id', 'date'])
                stored = ds[(ds['method'] == method) & (ds['scenario'] == scen_label)
                           & (ds['seed'].astype(str) == str(seed))]
                worst = 0.0
                for station_id, sgrp in series.groupby('station_id'):
                    obs_m = cds.compute_spell_metrics(cds.get_dry_spell_lengths(sgrp['y_true_full'].to_numpy()))
                    pred_m = cds.compute_spell_metrics(cds.get_dry_spell_lengths(sgrp['y_filled'].to_numpy()))
                    srow = stored[stored['station_id'] == station_id]
                    if srow.empty:
                        worst = max(worst, 1e9)
                        continue
                    srow = srow.iloc[0]
                    for k, v in (('cdd_obs', obs_m['cdd']), ('cdd_pred', pred_m['cdd']),
                                ('mean_dry_obs', obs_m['mean_dry']), ('mean_dry_pred', pred_m['mean_dry']),
                                ('p95_dry_obs', obs_m['p95_dry']), ('p95_dry_pred', pred_m['p95_dry'])):
                        worst = max(worst, abs(float(srow[k]) - float(v)))
                status = 'PASS' if worst < 1e-6 else 'FAIL'
                add('consistency', 'table7_recompute_match', status,
                    method=method, scenario=scen_label, seed=str(seed),
                    observed=round(worst, 8), expected=0.0, tolerance=1e-6, source_file=ds_path,
                    message=(f'{series["station_id"].nunique()} stations recomputed via '
                             f'build_filled_series -- matches stored') if status == 'PASS'
                            else f'max station-level diff={worst}')

    # -- Table 8: spatial_correlation.csv --
    sp_path = os.path.join(analysis_dir, 'spatial_correlation.csv')
    sp = pd.read_csv(sp_path, dtype={'seed': str})
    check_completeness(sp, 'table8', sp_path)

    for _, row in sp.iterrows():
        method, seed, scen_label = row['method'], row['seed'], row['scenario']
        is_deterministic = METHODS[method][0] == ['deterministic']
        seed_val = seed if is_deterministic else int(seed)
        series = bco.build_filled_series(method, seed_val, scen_label, sc, pidx,
                                         te, dates, sids, y_true_full)
        pearson_mae = csc.compute_corr_mae(series, 'y_filled', 'y_true_full', 'pearson', False)
        spearman_mae = csc.compute_corr_mae(series, 'y_filled', 'y_true_full', 'spearman', False)
        occ_mae = csc.compute_corr_mae(series, 'y_filled', 'y_true_full', 'pearson', True)
        worst = max(abs(row['pearson_amount_corr_mae'] - pearson_mae),
                   abs(row['spearman_amount_corr_mae'] - spearman_mae),
                   abs(row['wet_occurrence_corr_mae'] - occ_mae))
        status = 'PASS' if worst < 1e-6 else 'FAIL'
        add('consistency', 'table8_recompute_match', status,
            method=method, scenario=scen_label, seed=str(seed),
            observed=round(worst, 8), expected=0.0, tolerance=1e-6, source_file=sp_path,
            message='recomputed matches stored' if status == 'PASS' else f'diff={worst}')

    # -- provenance: these 3 analyses must only read predictions/*.csv +
    # preprocessed_test.npz + scaler.pkl (enforced by construction -- the
    # scripts import only canonical_metrics/build_canonical_outputs/
    # direct_two_stage_rf and never open a results/*.csv or src/*.csv by
    # name other than results/canonical/predictions/*.csv). Static check:
    for script_name in ('canonical_false_wet_analysis.py', 'canonical_dry_spell.py',
                        'canonical_spatial_correlation.py'):
        path = os.path.join(SRC_DIR, script_name)
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        bad_refs = [b for b in LEGACY_BASENAMES if b in src]
        add('provenance', 'stage0_no_legacy_file_read',
            'PASS' if not bad_refs else 'FAIL',
            method=script_name, source_file=path,
            message='no legacy filename referenced in source' if not bad_refs
                    else f'legacy filenames referenced: {bad_refs}')


# ── Table 6 (generalisation ablation) + LOSO (Section 5.10) checks ─────────
# Model A of the ablation, and every LOSO fold, are cheap to completeness-
# check but expensive to bit-exactly re-derive (each involves training a
# fresh RandomForestClassifier/Regressor pair; re-running 5 more training
# passes here just to audit the audit is not worth the cost). What IS
# checked: (1) Model A's own numbers ARE bit-exactly recomputed from
# results/canonical/predictions/DirectTwoStageRF.csv, since that requires
# no retraining; (2) both outputs have exactly the expected row/column
# shape and no out-of-range values; (3) provenance -- neither script
# references a legacy filename. Model B / LOSO-fold numbers are trusted as
# "reproducible by re-running canonical_generalization_ablation.py /
# canonical_loso.py" (both scripts are deterministic given a fixed
# random_state) rather than independently re-derived here.

def audit_generalization_and_loso(sc, pidx, te):
    import canonical_generalization_ablation as cga

    analysis_dir = os.path.join(CANON_DIR, 'analysis')
    # canonical_generalization_ablation.py / canonical_loso.py were not
    # extended to netblock30d (a separate, later addition; see
    # build_network_block_scenario.py) -- use the original 4-scenario list,
    # not the current (5-scenario) global SCENARIOS, so this check compares
    # against what those scripts actually produce.
    scenario_order = [s for s, _c, _m in bco.SCENARIOS_NO_NETBLOCK]

    ga_path = os.path.join(analysis_dir, 'generalization_ablation.csv')
    ga = pd.read_csv(ga_path)
    add('completeness', 'table6_groups_exist',
        'PASS' if set(ga['scenario']) == set(scenario_order) and len(ga) == 4 else 'FAIL',
        method='GeneralizationAblation', observed=len(ga), expected=4, source_file=ga_path,
        message='all 4 scenarios present' if len(ga) == 4 else f'got {len(ga)} rows')

    a_metrics, a_source = cga.model_a_metrics()
    worst = 0.0
    for _, row in ga.iterrows():
        a = a_metrics[row['scenario']]
        worst = max(worst, abs(row['model_a_f1'] - a['f1']),
                    abs(row['model_a_rmse_p95'] - a['rmse_p95']))
    status = 'PASS' if worst < 1e-6 else 'FAIL'
    add('consistency', 'table6_model_a_recompute_match', status,
        method='GeneralizationAblation', observed=round(worst, 8), expected=0.0,
        tolerance=1e-6, source_file=a_source,
        message='Model A F1/RMSE_p95 recomputed from DirectTwoStageRF.csv matches stored'
                if status == 'PASS' else f'diff={worst}')

    b_valid = ga['model_b_f1'].between(0, 1).all() and (ga['model_b_rmse_p95'] > 0).all()
    add('completeness', 'table6_model_b_values_in_range',
        'PASS' if b_valid else 'FAIL',
        method='GeneralizationAblation', observed=b_valid, expected=True,
        source_file=ga_path,
        message=('Model B F1 in [0,1], RMSE_p95 > 0 for all scenarios' if b_valid
                 else 'Model B has out-of-range values')
                + ' (Model B is a live 20pct-trained rerun; reproducible via '
                  'canonical_generalization_ablation.py, not independently re-derived here)')

    loso_path = os.path.join(analysis_dir, 'loso.csv')
    loso = pd.read_csv(loso_path)
    expected_groups = {(int(s), scen) for s in np.unique(te['station_ids'])
                       for scen in scenario_order}
    actual_groups = set(zip(loso['held_out_station'], loso['scenario']))
    add('completeness', 'loso_groups_exist',
        'PASS' if expected_groups == actual_groups else 'FAIL',
        method='LOSO', observed=len(actual_groups), expected=len(expected_groups),
        source_file=loso_path,
        message='all 4 stations x 4 scenarios present' if expected_groups == actual_groups
                else f'missing: {sorted(expected_groups - actual_groups)}')

    loso_valid = loso['f1'].between(0, 1).all() and (loso['n_masked'] > 0).all()
    add('completeness', 'loso_values_in_range',
        'PASS' if loso_valid else 'FAIL',
        method='LOSO', observed=loso_valid, expected=True, source_file=loso_path,
        message=('F1 in [0,1], n_masked > 0 for all folds' if loso_valid
                 else 'LOSO has out-of-range values')
                + ' (live per-fold reruns; reproducible via canonical_loso.py, '
                  'not independently re-derived here)')

    for script_name in ('canonical_generalization_ablation.py', 'canonical_loso.py'):
        path = os.path.join(SRC_DIR, script_name)
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        bad_refs = [b for b in LEGACY_BASENAMES if b in src]
        add('provenance', 'table6_loso_no_legacy_file_read',
            'PASS' if not bad_refs else 'FAIL',
            method=script_name, source_file=path,
            message='no legacy filename referenced in source' if not bad_refs
                    else f'legacy filenames referenced: {bad_refs}')


# ── Multi-mask variance experiment (Section 5.9) checks ────────────────────
# Live rerun (10 fresh mask realizations x 2 scenarios x 3 methods, with
# AmountRF_DLPIF/SAITS each repeated across all 3 model seeds); like
# Table 6/LOSO, a full bit-exact recompute here would mean re-running the
# whole experiment again, so this checks completeness/shape/range instead
# and trusts the script's own determinism (fixed mask seeds 9000-9009,
# fixed model seeds 42/123/456) for reproducibility.

def audit_multimask_variance():
    # 2026-08-03: extended to pool RMSE_p95/MAE_p95 across all 3 model
    # seeds (not seed 42 alone) and to add SAITS -- Linear stays a single
    # deterministic row per (scenario, realization); AmountRF_DLPIF and
    # SAITS each get one row per (scenario, realization, seed).
    path = os.path.join(CANON_DIR, 'analysis', 'multimask_variance.csv')
    df = pd.read_csv(path, dtype={'seed': str})
    seeded_methods = ('AmountRF_DLPIF', 'SAITS')
    expected_groups = ({('Linear', scen, r, 'deterministic')
                        for scen in ('10pct', '20pct') for r in range(10)}
                       | {(method, scen, r, str(s)) for method in seeded_methods
                          for scen in ('10pct', '20pct') for r in range(10) for s in (42, 123, 456)})
    actual_groups = set(zip(df['method'], df['scenario'], df['realization'], df['seed']))
    add('completeness', 'multimask_groups_exist',
        'PASS' if expected_groups == actual_groups else 'FAIL',
        method='MultiMaskVariance', observed=len(actual_groups), expected=len(expected_groups),
        source_file=path,
        message='Linear (deterministic) + AmountRF_DLPIF/SAITS (3 seeds each) x 2 scenarios x '
                '10 realizations present'
                if expected_groups == actual_groups
                else f'missing: {sorted(expected_groups - actual_groups)[:10]}')

    valid = df['f1'].between(0, 1).all() and (df['n_masked'] > 0).all()
    add('completeness', 'multimask_values_in_range',
        'PASS' if valid else 'FAIL',
        method='MultiMaskVariance', observed=valid, expected=True, source_file=path,
        message='F1 in [0,1], n_masked > 0 for all realizations' if valid
                else 'out-of-range values found')

    # sanity: every realization's mask seed is unique per scenario (no
    # accidental duplicate/reused mask across the 10 nominally-independent
    # realizations)
    for scen in ('10pct', '20pct'):
        seeds_used = df[(df['scenario'] == scen)]['mask_seed'].unique()
        ok = len(seeds_used) == 10
        add('completeness', 'multimask_seeds_unique',
            'PASS' if ok else 'FAIL',
            method='MultiMaskVariance', scenario=scen,
            observed=len(seeds_used), expected=10, source_file=path,
            message='10 distinct mask seeds' if ok else f'only {len(seeds_used)} distinct seeds')

    script_path = os.path.join(SRC_DIR, 'canonical_multimask_variance.py')
    with open(script_path, 'r', encoding='utf-8') as f:
        src = f.read()
    bad_refs = [b for b in LEGACY_BASENAMES if b in src]
    add('provenance', 'multimask_no_legacy_file_read',
        'PASS' if not bad_refs else 'FAIL',
        method='canonical_multimask_variance.py', source_file=script_path,
        message='no legacy filename referenced in source' if not bad_refs
                else f'legacy filenames referenced: {bad_refs}')


def main():
    print('=' * 70)
    print('  audit_canonical_outputs.py')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    te = d2s.load_npz('test')

    print('\n  [1] completeness + integrity checks ...')
    audit_completeness_and_integrity(sc, pidx, te)

    print('  [3] metrics consistency + traceability checks ...')
    audit_metrics_consistency(sc, pidx, te)

    print('  [4] no-legacy-file-read provenance check ...')
    audit_no_legacy_reads()

    print('  [5] DLPIF vs DirectTwoStageRF equivalence check ...')
    audit_dlpif_vs_direct_equivalence()

    print('  [6] Stage-0 analyses (Tables 4/7/8 sources) checks ...')
    audit_stage0_analyses(sc, pidx, te)

    print('  [7] Table 6 (generalisation ablation) + LOSO checks ...')
    audit_generalization_and_loso(sc, pidx, te)

    print('  [8] Multi-mask variance (Section 5.9) checks ...')
    audit_multimask_variance()

    df = pd.DataFrame(ROWS)
    col_order = ['audit_category', 'check_name', 'method', 'scenario', 'seed',
                'status', 'observed', 'expected', 'tolerance', 'source_file', 'message']
    df = df[col_order]

    # Store source_file as a REPO_ROOT-relative, forward-slash path so
    # canonical_audit.csv is reproducible across platforms (previously
    # stored absolute Windows paths with backslashes, which do not resolve
    # on Linux/Mac -- e.g. every traceability check would fail there even
    # though the underlying audit passed).
    def _relativize(v):
        if isinstance(v, str) and os.path.isabs(v) and v.startswith(REPO_ROOT):
            v = os.path.relpath(v, REPO_ROOT)
        if isinstance(v, str):
            v = v.replace(os.sep, '/')
        return v
    df['source_file'] = df['source_file'].map(_relativize)

    out_path = os.path.join(CANON_DIR, 'canonical_audit.csv')
    df.to_csv(out_path, index=False)

    n_pass = int((df['status'] == 'PASS').sum())
    n_fail = int((df['status'] == 'FAIL').sum())
    n_warn = int((df['status'] == 'WARN').sum())

    print('\n' + '=' * 70)
    print('  AUDIT SUMMARY')
    print('=' * 70)
    print(f'  Total checks : {len(df)}')
    print(f'  PASS         : {n_pass}')
    print(f'  WARN         : {n_warn}')
    print(f'  FAIL         : {n_fail}')

    if n_fail or n_warn:
        print('\n  Non-PASS rows:')
        print(df[df['status'] != 'PASS'].to_string(index=False))
    else:
        print('\n  TARGET MET: FAIL=0, WARN=0')

    print(f'\n  -> {out_path}')
    print('=' * 70)

    if n_fail or n_warn:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

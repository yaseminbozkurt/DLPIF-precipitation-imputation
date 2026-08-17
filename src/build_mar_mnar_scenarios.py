"""
build_mar_mnar_scenarios.py
============================
Adds four new PRECIP-only test scenarios that vary the *missingness
mechanism* while holding the missing-cell budget fixed at the same level as
the existing 'corrupted_10pct' scenario (Section 4.2.1's MCAR baseline):

  mar_meteo                -- Missing-At-Random: PRECIP loss probability
                     increases with RH_MEAN / WIND_MEAN / P_MEAN (observed
                     covariates other than PRECIP itself), not with PRECIP's
                     own value.
  mnar_wet                 -- Missing-Not-At-Random (occurrence-dependent):
                     wet days (PRECIP > 0.1mm) are MNAR_WET_RATIO times more
                     likely to be selected for masking than dry days.
  mnar_intensity_moderate  -- Missing-Not-At-Random (amount-dependent, lower
                     dose): selection weight grows monotonically with raw
                     PRECIP magnitude (weight_t = (1+PRECIP_t)^beta). beta is
                     selected on the VALIDATION partition only (see
                     select_beta_from_validation()) so that VAL's own
                     realized p95-event missing rate lands in a 30-75% band;
                     the frozen beta is then applied to TEST unconditionally
                     -- TEST's own realized rate is reported, never asserted.
  mnar_intensity_severe    -- Same amount-dependent mechanism at a higher
                     dose, again beta-selected on VALIDATION only (smallest
                     grid value whose VAL realized rate exceeds moderate's).
                     Kept alongside `_moderate` to give a dose-response pair
                     (MCAR -> moderate -> severe) rather than a single
                     all-or-nothing extreme scenario -- a reviewer could
                     otherwise object that degraded extreme-event
                     reconstruction is a trivial consequence of having
                     removed every extreme observation by construction.

LEAKAGE FIX (2026-08-11): earlier versions of this script selected/verified
both beta doses by asserting a target p95-missing-rate band computed on the
TEST partition itself -- a test-blindness violation flagged in manuscript
review, since the stress-test's severity was tuned against a held-out-test
outcome. Selection now runs against VALIDATION only; TEST is masked with the
frozen result and its realized rate is reported descriptively.

Design (confirmed with user before implementation):
  - Only the PRECIP column is touched by these three mechanisms. The other
    6 meteorological variables keep EXACTLY the same mask as the existing
    'corrupted_10pct' scenario. This isolates the missingness mechanism as
    the only thing that differs between '10pct' (MCAR) and these three
    scenarios -- local context (temperature/humidity/wind/pressure) stays
    fully observed, unlike block7d/block30d/netblock30d which stress-test
    context *availability* instead of mechanism.
  - Cells are drawn by weighted sampling WITHOUT replacement so each
    scenario hits the exact same PRECIP missing-cell count as '10pct' --
    directly comparable realized missing rates, no separate calibration
    step (e.g. bisection on a logistic intercept) needed.

Method
------
1. Load the existing preprocessed_test.npz (already containing 10pct, 20pct,
   block7d, block30d, netblock30d) and adjacency.pkl -- read-only; nothing
   about the existing scenarios or splits is touched.
2. Recover raw-scale PRECIP / RH_MEAN / WIND_MEAN / P_MEAN via the project's
   saved MinMaxScaler (scaler.pkl) applied with .inverse_transform() to the
   full 7-column normalised 'data' array; NaNs propagate through the linear
   inverse transform unchanged.
3. Candidate pool = rows where real_mask[:, PRECIP] == 1 (naturally observed
   PRECIP), exactly 01_data_preprocessing.random_missingness()'s eligibility
   rule. n_remove = round(len(candidates) * TARGET_RATE), matching '10pct'.
4. Each scenario assigns every candidate a nonnegative weight from its own
   mechanism (see docstrings on the three weight functions below), then
   draws exactly n_remove candidates without replacement using those weights
   as a sampling distribution (RNG seed distinct per scenario).
5. Only the PRECIP column is masked; the other 6 columns / art_mask entries
   are copied verbatim from corrupted_10pct / art_mask_10pct.
6. neighbor_avg_<key> / neighbor_mask_<key> are recomputed from the new
   corrupted array via 01_data_preprocessing.compute_neighbor_avg().
7. The four new keys per scenario are merged into preprocessed_test.npz
   (train/val untouched -- Stage 1/Stage 2 are trained once on 10pct and
   applied without retraining to every test scenario, exactly as for
   block7d/block30d/netblock30d).

Output
------
  preprocessed_test.npz          -- updated in place (original backed up
                                      first to preprocessed_test_pre_marmnar.npz.bak)
  results/mar_mnar_missingness_diagnostics.csv
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prep01", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_data_preprocessing.py"))
prep01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep01)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_RATE = 0.10                # same PRECIP missing-cell budget as '10pct'
MAR_BETAS = (2.0, 2.0, 2.0)       # logistic weights on normalized RH_MEAN, WIND_MEAN, P_MEAN
MNAR_WET_RATIO = 3.0               # wet-cell : dry-cell sampling weight ratio
# weight_t = (1 + raw_PRECIP) ** beta. Two doses, forming a deliberate
# dose-response pair (MCAR -> MODERATE -> SEVERE): a reviewer could
# reasonably object that a single "of course extremes are reconstructed
# poorly, you removed every extreme observation" scenario proves too little.
#
# LEAKAGE FIX (2026-08-11): beta was previously selected/verified by a
# hard-coded assertion checking the realized p95-masking rate on the TEST
# partition's own 47 p95-observed cells -- i.e. the stress-scenario severity
# was tuned against a held-out-test outcome, a test-blindness violation
# flagged in manuscript review. Beta selection now runs against the
# VALIDATION partition only (see select_beta_from_validation() below); the
# frozen values are applied to TEST unconditionally, and whatever p95
# masking rate results on TEST is simply reported, never asserted against.
MNAR_INTENSITY_MODERATE_BAND = (0.30, 0.75)   # validation-partition target band
MNAR_INTENSITY_MODERATE_BETA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
MNAR_INTENSITY_SEVERE_BETA_GRID = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3.0]
WET_THRESH = 0.1                   # mm, matches direct_two_stage_rf.py::WET_THRESH
P95_THRESH = 19.2                  # mm, matches direct_two_stage_rf.py::P95_THRESH
SEED_BASE = 42 + 500               # +1/+2/+3/+4 per scenario below

SCENARIO_SEEDS = {
    'mar_meteo': SEED_BASE + 1,
    'mnar_wet': SEED_BASE + 2,
    'mnar_intensity_severe': SEED_BASE + 3,
    'mnar_intensity_moderate': SEED_BASE + 4,
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def mar_meteo_weights(rh_norm, wind_norm, press_norm, betas=MAR_BETAS):
    """MAR: weight grows with observed RH/WIND/PRESS, not with PRECIP itself.

    Naturally-missing covariate values are replaced with their column mean
    over the candidate pool (a neutral prior) rather than dropped, so every
    PRECIP-observed candidate always receives a finite weight.
    """
    def filled(col):
        col = col.copy()
        m = np.nanmean(col)
        col[np.isnan(col)] = m
        return col

    rh_f, wind_f, press_f = filled(rh_norm), filled(wind_norm), filled(press_norm)
    score = betas[0] * rh_f + betas[1] * wind_f + betas[2] * press_f
    return sigmoid(score)


def mnar_wet_weights(raw_precip, ratio=MNAR_WET_RATIO, wet_thresh=WET_THRESH):
    """MNAR (occurrence-dependent): wet days are `ratio`x more likely than dry days."""
    is_wet = raw_precip > wet_thresh
    return np.where(is_wet, ratio, 1.0)


def mnar_intensity_weights(raw_precip, beta):
    """MNAR (amount-dependent): weight grows monotonically with event size."""
    safe_precip = np.clip(raw_precip, 0, None)
    return np.power(1.0 + safe_precip, beta)


def weighted_sample_without_replacement(candidates, weights, n_remove, seed):
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=np.float64)
    assert np.all(w >= 0) and w.sum() > 0
    p = w / w.sum()
    chosen = rng.choice(candidates, size=n_remove, replace=False, p=p)
    return np.sort(chosen)


def select_beta_from_validation(scaler, target_rate=TARGET_RATE, wet_thresh=WET_THRESH,
                                 p95_thresh=P95_THRESH, seed_base=SEED_BASE + 900):
    """Select MNAR-Intensity beta doses using ONLY the validation partition -- the leakage fix.

    For each beta candidate, simulates the identical weighted-sampling mechanism used later on
    TEST, but on VAL's own naturally-observed PRECIP cells, and measures the realized p95-conditional
    missing rate on VAL. Moderate is the smallest grid value landing inside MNAR_INTENSITY_MODERATE_BAND
    on validation; if none land inside the band, the closest-to-band-center value is used and flagged.
    Severe is the smallest grid value whose validation realized rate exceeds Moderate's (dose-response
    ordering), enforced on validation data only. The frozen (beta, validation-diagnostic) values are
    returned; TEST is never consulted during this selection.
    """
    val_path = os.path.join(SRC_DIR, 'preprocessed_val.npz')
    va = dict(np.load(val_path, allow_pickle=True))
    va_data = va['data'].astype(np.float32)
    va_real_mask = va['real_mask'].astype(np.float32)
    meteo_vars = list(va['meteo_vars'])
    precip_idx = meteo_vars.index('PRECIP')

    va_raw = scaler.inverse_transform(va_data)
    va_raw_precip = va_raw[:, precip_idx]
    va_candidates = np.where(va_real_mask[:, precip_idx] == 1)[0]
    va_n_remove = int(round(len(va_candidates) * target_rate))

    is_wet_all = va_raw_precip > wet_thresh
    is_extreme_all = va_raw_precip >= p95_thresh
    n_extreme_obs = int(is_extreme_all[va_candidates].sum())
    print(f'\n  [BETA SELECTION -- VALIDATION PARTITION ONLY]')
    print(f'  VAL PRECIP candidate pool: {len(va_candidates):,} rows, '
          f'target missing count: {va_n_remove:,}, p95-observed cells: {n_extreme_obs}')

    def realized_rate(beta, seed):
        w = mnar_intensity_weights(va_raw_precip[va_candidates], beta=beta)
        sel = weighted_sample_without_replacement(va_candidates, w, va_n_remove, seed)
        masked = np.zeros(len(va_data), dtype=bool)
        masked[sel] = True
        n_masked = int((masked & is_extreme_all)[va_candidates].sum())
        return n_masked / n_extreme_obs if n_extreme_obs else float('nan')

    lo, hi = MNAR_INTENSITY_MODERATE_BAND
    moderate_beta, moderate_rate, in_band = None, None, False
    for beta in MNAR_INTENSITY_MODERATE_BETA_GRID:
        rate = realized_rate(beta, seed_base + int(beta * 100))
        print(f'    moderate candidate beta={beta:.2f} -> VAL p95 missing rate {rate:.4f}')
        if lo <= rate <= hi:
            moderate_beta, moderate_rate, in_band = beta, rate, True
            break
    if moderate_beta is None:
        band_mid = (lo + hi) / 2
        best = min(MNAR_INTENSITY_MODERATE_BETA_GRID,
                   key=lambda b: abs(realized_rate(b, seed_base + int(b * 100)) - band_mid))
        moderate_beta = best
        moderate_rate = realized_rate(best, seed_base + int(best * 100))
        print(f'    [WARN] no grid value landed inside [{lo},{hi}] on VAL; '
              f'using closest-to-band-center beta={moderate_beta:.2f} (VAL rate {moderate_rate:.4f})')
    else:
        print(f'    [OK] moderate beta={moderate_beta:.2f} selected (VAL p95 missing rate '
              f'{moderate_rate:.4f} inside [{lo},{hi}])')

    severe_beta, severe_rate = None, None
    for beta in MNAR_INTENSITY_SEVERE_BETA_GRID:
        rate = realized_rate(beta, seed_base + int(beta * 100))
        print(f'    severe candidate beta={beta:.2f} -> VAL p95 missing rate {rate:.4f}')
        if rate > moderate_rate:
            severe_beta, severe_rate = beta, rate
            break
    if severe_beta is None:
        severe_beta = MNAR_INTENSITY_SEVERE_BETA_GRID[-1]
        severe_rate = realized_rate(severe_beta, seed_base + int(severe_beta * 100))
        print(f'    [WARN] no grid value exceeded moderate on VAL; using largest grid value '
              f'beta={severe_beta:.2f} (VAL rate {severe_rate:.4f})')
    else:
        print(f'    [OK] severe beta={severe_beta:.2f} selected (VAL p95 missing rate {severe_rate:.4f} '
              f'> moderate VAL rate {moderate_rate:.4f})')

    return dict(moderate_beta=moderate_beta, severe_beta=severe_beta,
                moderate_val_rate=moderate_rate, severe_val_rate=severe_rate,
                moderate_in_band=in_band, val_n_extreme_obs=n_extreme_obs)


def main():
    print('=' * 70)
    print('  BUILD MAR / MNAR PRECIP-ONLY MISSINGNESS-MECHANISM SCENARIOS')
    print(f'  (target rate {TARGET_RATE:.0%}, same PRECIP budget as 10pct;'
          f' PRECIP column only, other variables copied from 10pct)')
    print('=' * 70)

    test_path = os.path.join(SRC_DIR, 'preprocessed_test.npz')
    backup_path = os.path.join(SRC_DIR, 'preprocessed_test_pre_marmnar.npz.bak')
    if not os.path.exists(backup_path):
        shutil.copy2(test_path, backup_path)
        print(f'  Backed up original test npz -> {backup_path}')
    else:
        print(f'  Backup already exists, not overwriting: {backup_path}')

    te = dict(np.load(test_path, allow_pickle=True))
    data = te['data'].astype(np.float32)
    real_mask = te['real_mask'].astype(np.float32)
    station_ids = te['station_ids']
    meteo_vars = list(te['meteo_vars'])

    assert 'corrupted_10pct' in te and 'art_mask_10pct' in te, \
        'corrupted_10pct / art_mask_10pct not found -- run 01_data_preprocessing.py first'
    base_corrupted = te['corrupted_10pct'].astype(np.float32)
    base_art_mask = te['art_mask_10pct'].astype(np.float32)

    with open(os.path.join(SRC_DIR, 'adjacency.pkl'), 'rb') as f:
        import pickle
        adj = pickle.load(f)
    A_knn = adj['A_knn']
    stations = sorted(adj['stations'])

    with open(os.path.join(SRC_DIR, 'scaler.pkl'), 'rb') as f:
        import pickle
        sc_data = pickle.load(f)
    scaler = sc_data['scaler']

    precip_idx = meteo_vars.index('PRECIP')
    rh_idx = meteo_vars.index('RH_MEAN')
    wind_idx = meteo_vars.index('WIND_MEAN')
    press_idx = meteo_vars.index('P_MEAN')

    # Inverse-transform the full normalised (uncorrupted, ground-truth) array
    # back to raw physical units. NaNs (natural missingness) propagate
    # unchanged through the scaler's linear inverse transform.
    raw = scaler.inverse_transform(data)
    raw_precip = raw[:, precip_idx]
    rh_norm = data[:, rh_idx]
    wind_norm = data[:, wind_idx]
    press_norm = data[:, press_idx]

    candidates = np.where(real_mask[:, precip_idx] == 1)[0]
    n_remove = int(round(len(candidates) * TARGET_RATE))
    print(f'  PRECIP candidate pool (naturally observed): {len(candidates):,} rows')
    print(f'  Target missing-cell count per scenario: {n_remove:,} '
          f'({TARGET_RATE:.0%} of candidates, matching corrupted_10pct budget)')

    n_base_10pct = int(base_art_mask[candidates, precip_idx].sum())
    print(f'  (corrupted_10pct actually masks {n_base_10pct:,} of these PRECIP cells, for reference)')

    beta_selection = select_beta_from_validation(scaler)
    MNAR_INTENSITY_MODERATE_BETA = beta_selection['moderate_beta']
    MNAR_INTENSITY_SEVERE_BETA = beta_selection['severe_beta']
    print(f'\n  Frozen doses (selected on VAL, applied unconditionally to TEST): '
          f'moderate beta={MNAR_INTENSITY_MODERATE_BETA:.2f}, severe beta={MNAR_INTENSITY_SEVERE_BETA:.2f}')

    weight_fns = {
        'mar_meteo': lambda: mar_meteo_weights(
            rh_norm[candidates], wind_norm[candidates], press_norm[candidates]),
        'mnar_wet': lambda: mnar_wet_weights(raw_precip[candidates]),
        'mnar_intensity_moderate': lambda: mnar_intensity_weights(
            raw_precip[candidates], beta=MNAR_INTENSITY_MODERATE_BETA),
        'mnar_intensity_severe': lambda: mnar_intensity_weights(
            raw_precip[candidates], beta=MNAR_INTENSITY_SEVERE_BETA),
    }

    diag_rows = []

    for scen_key, weight_fn in weight_fns.items():
        print(f'\n  --- {scen_key} ---')
        weights = weight_fn()
        selected_rows = weighted_sample_without_replacement(
            candidates, weights, n_remove, SCENARIO_SEEDS[scen_key])

        art_mask = base_art_mask.copy()
        corrupted = base_corrupted.copy()
        # Reset PRECIP column to the mechanism-specific selection only
        # (non-PRECIP columns keep the 10pct mask/values untouched).
        art_mask[:, precip_idx] = 0.0
        corrupted[:, precip_idx] = data[:, precip_idx]
        art_mask[selected_rows, precip_idx] = 1.0
        corrupted[selected_rows, precip_idx] = np.nan

        n_hidden = int(art_mask[:, precip_idx].sum())
        assert n_hidden == n_remove, f'{scen_key}: expected {n_remove} masked PRECIP cells, got {n_hidden}'

        # Non-PRECIP columns must be byte-identical to corrupted_10pct.
        other_cols = [i for i in range(data.shape[1]) if i != precip_idx]
        assert np.array_equal(art_mask[:, other_cols], base_art_mask[:, other_cols]), \
            f'{scen_key}: non-PRECIP art_mask diverged from corrupted_10pct'
        np.testing.assert_array_equal(
            np.nan_to_num(corrupted[:, other_cols], nan=-9999.0),
            np.nan_to_num(base_corrupted[:, other_cols], nan=-9999.0),
            err_msg=f'{scen_key}: non-PRECIP corrupted values diverged from corrupted_10pct')

        # No synthetic mask over naturally-missing cells.
        bad_natural = int(((art_mask > 0.5) & (real_mask < 0.5)).sum())
        assert bad_natural == 0, f'{scen_key}: artificial mask covers {bad_natural} naturally-missing cells'

        # Diagnostics: realized wet / dry / p95 missing rates.
        is_wet_all = raw_precip > WET_THRESH
        is_extreme_all = raw_precip >= P95_THRESH
        precip_obs = candidates  # naturally-observed PRECIP rows (denominator)
        masked_bool = np.zeros(len(data), dtype=bool)
        masked_bool[selected_rows] = True

        n_wet_obs = int(is_wet_all[precip_obs].sum())
        n_dry_obs = int((~is_wet_all[precip_obs]).sum())
        n_extreme_obs = int(is_extreme_all[precip_obs].sum())
        n_wet_masked = int((masked_bool & is_wet_all)[precip_obs].sum())
        n_dry_masked = int((masked_bool & ~is_wet_all)[precip_obs].sum())
        n_extreme_masked = int((masked_bool & is_extreme_all)[precip_obs].sum())

        wet_rate = n_wet_masked / n_wet_obs if n_wet_obs else float('nan')
        dry_rate = n_dry_masked / n_dry_obs if n_dry_obs else float('nan')
        extreme_rate = n_extreme_masked / n_extreme_obs if n_extreme_obs else float('nan')
        overall_rate = n_hidden / len(candidates)

        print(f'  Masked {n_hidden:,} PRECIP cells (overall rate {overall_rate:.4f})')
        print(f'  Wet-day missing rate : {wet_rate:.4f}  ({n_wet_masked}/{n_wet_obs})')
        print(f'  Dry-day missing rate : {dry_rate:.4f}  ({n_dry_masked}/{n_dry_obs})')
        print(f'  p95-event missing rate: {extreme_rate:.4f}  ({n_extreme_masked}/{n_extreme_obs})')

        if scen_key == 'mnar_wet':
            assert wet_rate > dry_rate, \
                f'mnar_wet: wet-day missing rate ({wet_rate:.4f}) did not exceed dry-day rate ({dry_rate:.4f})'
            print('  [OK] wet-day missing rate exceeds dry-day missing rate, as intended.')

        if scen_key == 'mnar_intensity_moderate':
            # LEAKAGE FIX: beta was selected/verified on VALIDATION only (see beta_selection above);
            # the realized rate on TEST is reported here descriptively and is NEVER asserted against --
            # test-blind by construction, whatever value results.
            lo, hi = MNAR_INTENSITY_MODERATE_BAND
            band_note = 'inside' if lo <= extreme_rate <= hi else 'outside'
            print(f'  [REPORT-ONLY] TEST p95-event missing rate {extreme_rate:.4f} is {band_note} the '
                  f'[{lo},{hi}] band that guided VAL-based beta selection (VAL rate was '
                  f'{beta_selection["moderate_val_rate"]:.4f}). Not asserted -- reported as-is.')
            moderate_extreme_rate = extreme_rate

        if scen_key == 'mnar_intensity_severe':
            ordering_note = 'held' if extreme_rate > moderate_extreme_rate else 'did NOT hold'
            print(f'  [REPORT-ONLY] TEST dose-response ordering (severe > moderate) {ordering_note}: '
                  f'severe={extreme_rate:.4f} vs moderate={moderate_extreme_rate:.4f} on TEST '
                  f'(ordering was selected on VAL: severe={beta_selection["severe_val_rate"]:.4f} vs '
                  f'moderate={beta_selection["moderate_val_rate"]:.4f}). Not asserted -- reported as-is.')

        print('  Recomputing neighbor_avg / neighbor_mask ...')
        nbr_avg, nbr_mask = prep01.compute_neighbor_avg(corrupted, station_ids, A_knn, stations)

        te[f'corrupted_{scen_key}'] = corrupted.astype(np.float32)
        te[f'art_mask_{scen_key}'] = art_mask.astype(np.float32)
        te[f'neighbor_avg_{scen_key}'] = nbr_avg.astype(np.float32)
        te[f'neighbor_mask_{scen_key}'] = nbr_mask.astype(np.float32)

        row = dict(
            scenario=scen_key,
            n_candidates=len(candidates),
            n_masked=n_hidden,
            overall_missing_rate=overall_rate,
            n_wet_obs=n_wet_obs, n_wet_masked=n_wet_masked, wet_missing_rate=wet_rate,
            n_dry_obs=n_dry_obs, n_dry_masked=n_dry_masked, dry_missing_rate=dry_rate,
            n_extreme_obs=n_extreme_obs, n_extreme_masked=n_extreme_masked,
            extreme_missing_rate=extreme_rate,
        )
        if scen_key == 'mnar_intensity_moderate':
            row.update(beta=MNAR_INTENSITY_MODERATE_BETA,
                       beta_selected_on='validation',
                       val_extreme_missing_rate=beta_selection['moderate_val_rate'],
                       val_n_extreme_obs=beta_selection['val_n_extreme_obs'],
                       val_landed_in_band=beta_selection['moderate_in_band'])
        elif scen_key == 'mnar_intensity_severe':
            row.update(beta=MNAR_INTENSITY_SEVERE_BETA,
                       beta_selected_on='validation',
                       val_extreme_missing_rate=beta_selection['severe_val_rate'],
                       val_n_extreme_obs=beta_selection['val_n_extreme_obs'])
        diag_rows.append(row)

    np.savez_compressed(test_path, **te)
    print(f'\n  -> Updated {test_path} with corrupted_/art_mask_/neighbor_avg_/neighbor_mask_'
          f' for {", ".join(weight_fns.keys())}')

    diag_df = pd.DataFrame(diag_rows)
    diag_path = os.path.join(SRC_DIR, '..', 'results', 'mar_mnar_missingness_diagnostics.csv')
    diag_df.to_csv(diag_path, index=False)
    print(f'  -> {diag_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()

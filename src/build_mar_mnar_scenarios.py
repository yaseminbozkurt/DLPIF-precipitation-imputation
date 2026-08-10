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
                     PRECIP magnitude (weight_t = (1+PRECIP_t)^0.7),
                     calibrated so the realized p95-event missing rate lands
                     in a 30-75% band rather than removing every extreme.
  mnar_intensity_severe    -- Same amount-dependent mechanism at a much
                     higher dose (weight_t = (1+PRECIP_t)^1.5), the original
                     adversarial stress test that removes essentially all
                     p95-observed test cells. Kept alongside `_moderate` to
                     give a dose-response pair (MCAR -> moderate -> severe)
                     rather than a single all-or-nothing extreme scenario --
                     a reviewer could otherwise object that degraded extreme-
                     event reconstruction is a trivial consequence of having
                     removed every extreme observation by construction.

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
# weight_t = (1 + raw_PRECIP) ** beta. Two doses, calibrated empirically
# against this dataset's 47 p95-observed test cells (see the beta sweep in
# the implementation notes): MODERATE lands the realized p95-event missing
# rate near the middle of a 40-70% band, SEVERE is the original adversarial
# stress test that masks essentially all p95 observations. Reviewers could
# reasonably object that a single "of course extremes are reconstructed
# poorly, you removed every extreme observation" scenario proves too little;
# the dose-response pair (MCAR -> MODERATE -> SEVERE) is more convincing.
MNAR_INTENSITY_MODERATE_BETA = 0.7
MNAR_INTENSITY_SEVERE_BETA = 1.5
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
            assert 0.30 <= extreme_rate <= 0.75, \
                f'mnar_intensity_moderate: p95 missing rate {extreme_rate:.4f} outside intended [0.30, 0.75] band'
            print(f'  [OK] p95-event missing rate {extreme_rate:.4f} lands inside the intended moderate-dose band.')
            moderate_extreme_rate = extreme_rate

        if scen_key == 'mnar_intensity_severe':
            assert extreme_rate > moderate_extreme_rate, \
                (f'mnar_intensity_severe: p95 missing rate ({extreme_rate:.4f}) did not exceed '
                 f'mnar_intensity_moderate ({moderate_extreme_rate:.4f}) -- dose-response ordering violated')
            print(f'  [OK] p95-event missing rate {extreme_rate:.4f} exceeds the moderate dose '
                  f'({moderate_extreme_rate:.4f}), confirming MCAR -> moderate -> severe dose-response ordering.')

        print('  Recomputing neighbor_avg / neighbor_mask ...')
        nbr_avg, nbr_mask = prep01.compute_neighbor_avg(corrupted, station_ids, A_knn, stations)

        te[f'corrupted_{scen_key}'] = corrupted.astype(np.float32)
        te[f'art_mask_{scen_key}'] = art_mask.astype(np.float32)
        te[f'neighbor_avg_{scen_key}'] = nbr_avg.astype(np.float32)
        te[f'neighbor_mask_{scen_key}'] = nbr_mask.astype(np.float32)

        diag_rows.append(dict(
            scenario=scen_key,
            n_candidates=len(candidates),
            n_masked=n_hidden,
            overall_missing_rate=overall_rate,
            n_wet_obs=n_wet_obs, n_wet_masked=n_wet_masked, wet_missing_rate=wet_rate,
            n_dry_obs=n_dry_obs, n_dry_masked=n_dry_masked, dry_missing_rate=dry_rate,
            n_extreme_obs=n_extreme_obs, n_extreme_masked=n_extreme_masked,
            extreme_missing_rate=extreme_rate,
        ))

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

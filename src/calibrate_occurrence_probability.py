"""
calibrate_occurrence_probability.py
=====================================
RQ2 -- Stage-1 occurrence probability calibration under missingness
mechanisms. Companion to the RQ1 mechanism-sensitivity experiments
(build_mar_mnar_scenarios.py, results/rq1_mechanism_diagnostics/), frozen
at git tag `rq1-freeze`.

RQ1 showed that occurrence F1 can rise while wet-day RMSE and amount bias
also rise under outcome-dependent (MNAR) missingness -- i.e. F1 alone is
not evidence of reliability. RQ2 asks a narrower, complementary question:
are the Stage-1 occurrence classifier's WET/DRY PROBABILITIES themselves
calibrated (does p=0.8 mean "correct ~80% of the time"), and does that
calibration survive the same mechanism shift RQ1 stress-tested?

Design (agreed with user before implementation)
-------------------------------------------------
1. The pretrained Stage-1 occurrence RandomForestClassifier is NEVER
   refit. It is loaded from direct_two_stage_occurrence_seed{seed}.pkl
   (written by direct_two_stage_rf.py's most recent run, one file per
   seed) -- the exact same model whose predictions produced the RQ1
   table. Only a post-hoc SCALAR mapping from its raw predict_proba
   output to a calibrated probability is fit (Platt/sigmoid via 1-D
   logistic regression on the raw score; isotonic via
   sklearn.isotonic.IsotonicRegression) -- never CalibratedClassifierCV,
   which can silently refit the base estimator depending on sklearn
   version/`cv` setting.
2. The existing validation split (preprocessed_val.npz) is partitioned
   chronologically by UNIQUE DATE (not by row, so all 4 stations on a
   given date stay in the same half) into VAL-CAL (first 50% of unique
   dates) and VAL-SELECT (last 50%). Train and test date boundaries are
   untouched.
3. VAL-CAL fits the three probability variants (Raw / Platt / Isotonic).
   VAL-SELECT scores them (Brier score PRIMARY, ECE with 10 fixed
   equal-width bins SECONDARY) and picks the deployed variant -- test
   scenarios are never looked at before this selection is frozen.
4. Each variant then gets its OWN occurrence threshold, independently
   re-optimised for F1 on VAL-SELECT (calibration changes the probability
   scale, so the old 0.42-0.46 raw-RF thresholds from direct_two_stage_rf.py
   do not carry over).
5. The three frozen (mapping, threshold) pairs are applied ONCE to every
   test scenario -- no further fitting past this point.

Outputs (results/rq2_calibration/, a NEW result family -- RQ1's outputs
under results/ and results/rq1_mechanism_diagnostics/ are untouched):
  validation_split_manifest.csv     VAL-CAL / VAL-SELECT date assignment
  calibration_models/
      platt_seed{s}.pkl, isotonic_seed{s}.pkl
  validation_calibration_metrics.csv   Brier/LogLoss/ECE per variant x seed
                                        on VAL-SELECT, incl. selected flag
  thresholds.csv                       frozen tau per variant x seed
  reliability_bins.csv                 10-bin reliability diagram data
                                        (VAL-SELECT and, per scenario, TEST)
  test_calibration_metrics.csv         Brier/LogLoss/ECE/F1/precision/
                                        recall/bias per variant x scenario
                                        x seed on TEST (masked positions)
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, f1_score, precision_score, recall_score

import direct_two_stage_rf as d2s

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SRC_DIR, '..', 'results', 'rq2_calibration')
MODEL_DIR = os.path.join(OUT_DIR, 'calibration_models')
os.makedirs(MODEL_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
WET_THRESH = d2s.WET_THRESH
N_ECE_BINS = 10
TEST_SCENARIOS = ['10pct', 'mar_meteo', 'mnar_wet',
                   'mnar_intensity_moderate', 'mnar_intensity_severe',
                   'block7d', 'block30d', 'netblock30d']
THRESH_GRID = np.arange(0.02, 0.99, 0.01)  # finer than train_occ's grid; calibrated
                                            # probabilities can concentrate away from 0.5


# ── Calibration variants ────────────────────────────────────────────────────

def fit_platt(raw_p, y):
    """1-D logistic regression on the raw score -- classical Platt scaling,
    fit independently of (and after) the frozen RF; the RF is never touched.
    """
    lr = LogisticRegression(C=1e6, solver='lbfgs')
    lr.fit(raw_p.reshape(-1, 1), y)
    return lr


def apply_platt(lr, raw_p):
    return lr.predict_proba(raw_p.reshape(-1, 1))[:, 1]


def fit_isotonic(raw_p, y):
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(raw_p, y)
    return iso


def apply_isotonic(iso, raw_p):
    return iso.predict(raw_p)


# ── Calibration quality metrics ─────────────────────────────────────────────

def expected_calibration_error(p, y, n_bins=N_ECE_BINS):
    """10 fixed equal-width bins over [0,1] (not equal-frequency) -- chosen
    once here and never changed per scenario/variant, to keep ECE
    comparable across the whole RQ2 table.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    bin_rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            sel = (p >= lo) & (p <= hi)
        else:
            sel = (p >= lo) & (p < hi)
        cnt = int(sel.sum())
        if cnt == 0:
            bin_rows.append(dict(bin_lo=lo, bin_hi=hi, count=0,
                                 mean_pred=np.nan, obs_freq=np.nan))
            continue
        mean_pred = float(p[sel].mean())
        obs_freq = float(y[sel].mean())
        ece += (cnt / n) * abs(mean_pred - obs_freq)
        bin_rows.append(dict(bin_lo=lo, bin_hi=hi, count=cnt,
                             mean_pred=mean_pred, obs_freq=obs_freq))
    return float(ece), pd.DataFrame(bin_rows)


def best_threshold_for_f1(p, y, grid=THRESH_GRID):
    best_f1, best_cut = -1.0, 0.5
    for cut in grid:
        f1 = f1_score(y, (p >= cut).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_cut = f1, float(cut)
    return round(best_cut, 4), round(best_f1, 4)


def occurrence_row(p, y, cut):
    pred = (p >= cut).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return dict(
        n=len(y), freq_gt=round(float(y.mean()), 4), freq_pred=round(float(pred.mean()), 4),
        bias=round(float(pred.mean() - y.mean()), 4),
        precision=round(precision_score(y, pred, zero_division=0), 4),
        recall=round(recall_score(y, pred, zero_division=0), 4),
        f1=round(f1_score(y, pred, zero_division=0), 4),
        brier=round(float(brier_score_loss(y, p)), 4),
        logloss=round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('  RQ2 -- STAGE-1 OCCURRENCE PROBABILITY CALIBRATION')
    print('  (pretrained RF frozen; only Platt/Isotonic post-hoc mappings fit)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    va = d2s.load_npz('val')
    te = d2s.load_npz('test')

    # ── 1. VAL-CAL / VAL-SELECT split by UNIQUE DATE ────────────────────────
    va_dates = pd.to_datetime(va['dates'])
    unique_dates = np.sort(pd.unique(va_dates))
    n_dates = len(unique_dates)
    n_cal_dates = n_dates // 2
    cal_dates = set(unique_dates[:n_cal_dates])
    select_dates = set(unique_dates[n_cal_dates:])
    assert cal_dates.isdisjoint(select_dates), 'VAL-CAL / VAL-SELECT date sets intersect'
    assert len(cal_dates) + len(select_dates) == n_dates

    is_cal_row = pd.Series(va_dates).isin(cal_dates).to_numpy()
    is_select_row = pd.Series(va_dates).isin(select_dates).to_numpy()
    assert not np.any(is_cal_row & is_select_row), 'a validation row is in both halves'
    print(f'  VAL split: {n_dates} unique val dates -> VAL-CAL {len(cal_dates)} dates '
          f'({min(cal_dates)} .. {max(cal_dates)}), '
          f'VAL-SELECT {len(select_dates)} dates ({min(select_dates)} .. {max(select_dates)})')

    manifest = pd.DataFrame({
        'date': pd.to_datetime(unique_dates),
        'split': ['VAL-CAL' if d in cal_dates else 'VAL-SELECT' for d in unique_dates],
    })
    manifest.to_csv(os.path.join(OUT_DIR, 'validation_split_manifest.csv'), index=False)

    tr_na_key, tr_nm_key = d2s.neighbor_keys(d2s.TRAIN_SCENARIO)  # '10pct', same scenario val/test are trained against
    X_va_full = d2s.build_occ_X(va[d2s.TRAIN_COR_KEY].astype(np.float32),
                                va['temporal'].astype(np.float32),
                                va[tr_na_key].astype(np.float32),
                                va[tr_nm_key].astype(np.float32), pidx)
    va_gt_mm = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    y_va_full = (va_gt_mm > WET_THRESH).astype(int)

    val_cal_metrics_rows = []
    reliability_rows = []
    threshold_rows = []
    test_metrics_rows = []

    for seed in SEEDS:
        print(f'\n{"="*70}\n  SEED = {seed}\n{"="*70}')
        pkl_path = os.path.join(SRC_DIR, f'direct_two_stage_occurrence_seed{seed}.pkl')
        with open(pkl_path, 'rb') as f:
            saved = pickle.load(f)
        occ_rf = saved['rf']
        saved_scaler = saved['scaler']
        # Same underlying occurrence model sanity check: scaler used to train
        # the persisted RF must be byte-identical to the one this script uses
        # for every array it feeds the RF -- if they diverged, the frozen RF's
        # feature scale would silently mismatch VAL-CAL / test features.
        assert np.array_equal(saved_scaler.data_min_, sc.data_min_), \
            f'seed={seed}: persisted occurrence model scaler diverges from scaler.pkl'
        assert np.array_equal(saved_scaler.data_range_, sc.data_range_), \
            f'seed={seed}: persisted occurrence model scaler diverges from scaler.pkl'
        rf_probe_before = occ_rf.predict_proba(X_va_full[va_obs][:5])[:, 1].copy()

        mask_cal = va_obs & is_cal_row
        mask_select = va_obs & is_select_row
        assert not np.any(mask_cal & mask_select)

        raw_p_cal = occ_rf.predict_proba(X_va_full[mask_cal])[:, 1]
        y_cal = y_va_full[mask_cal]
        raw_p_select = occ_rf.predict_proba(X_va_full[mask_select])[:, 1]
        y_select = y_va_full[mask_select]
        print(f'  VAL-CAL n={mask_cal.sum()} (wet frac={y_cal.mean():.3f})  '
              f'VAL-SELECT n={mask_select.sum()} (wet frac={y_select.mean():.3f})')

        # RF must be unchanged by anything above (allclose, not array_equal:
        # RandomForestClassifier(n_jobs=-1)'s parallel tree-averaging order
        # is not guaranteed identical across calls, giving ~1e-16-scale
        # float jitter on repeat predict_proba() calls with IDENTICAL input
        # even with zero mutation in between -- verified separately. A real
        # refit/mutation would move predictions far more than 1e-9.)
        rf_probe_after = occ_rf.predict_proba(X_va_full[va_obs][:5])[:, 1]
        assert np.allclose(rf_probe_before, rf_probe_after, atol=1e-9), \
            f'seed={seed}: occurrence RF predict_proba changed -- RF was refit or mutated'

        platt = fit_platt(raw_p_cal, y_cal)
        iso = fit_isotonic(raw_p_cal, y_cal)
        with open(os.path.join(MODEL_DIR, f'platt_seed{seed}.pkl'), 'wb') as f:
            pickle.dump(platt, f)
        with open(os.path.join(MODEL_DIR, f'isotonic_seed{seed}.pkl'), 'wb') as f:
            pickle.dump(iso, f)

        variants = {
            'raw': raw_p_select,
            'platt': apply_platt(platt, raw_p_select),
            'isotonic': apply_isotonic(iso, raw_p_select),
        }
        assert (variants['raw'] is raw_p_select
                and np.array_equal(variants['platt'], apply_platt(platt, raw_p_select))
                and np.array_equal(variants['isotonic'], apply_isotonic(iso, raw_p_select))), \
            'all three variants must derive from the same VAL-SELECT raw probability input'

        brier_by_variant = {}
        for name, p in variants.items():
            brier = float(brier_score_loss(y_select, p))
            ll = float(log_loss(y_select, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
            ece, bins_df = expected_calibration_error(p, y_select)
            brier_by_variant[name] = brier
            bins_df['seed'] = seed; bins_df['variant'] = name; bins_df['split'] = 'VAL-SELECT'
            reliability_rows.append(bins_df)
            val_cal_metrics_rows.append(dict(seed=seed, variant=name,
                                             n_evaluated=len(y_select),
                                             brier=round(brier, 5), logloss=round(ll, 5),
                                             ece=round(ece, 5)))
            print(f'    [{name:8s}] Brier={brier:.5f}  LogLoss={ll:.5f}  ECE={ece:.5f}')

        selected_variant = min(brier_by_variant, key=brier_by_variant.get)
        print(f'  -> selected (min Brier on VAL-SELECT): {selected_variant}')
        for row in val_cal_metrics_rows:
            if row['seed'] == seed:
                row['selected'] = (row['variant'] == selected_variant)

        # Each variant gets its own frozen threshold, fit on VAL-SELECT.
        seed_thresholds = {}
        for name, p in variants.items():
            cut, f1_at_cut = best_threshold_for_f1(p, y_select)
            seed_thresholds[name] = cut
            threshold_rows.append(dict(seed=seed, variant=name, threshold=cut,
                                       val_select_f1=f1_at_cut,
                                       selected_calibration=(name == selected_variant)))
            print(f'    [{name:8s}] tau={cut:.3f}  (VAL-SELECT F1={f1_at_cut:.4f})')

        # ── Frozen application to TEST scenarios (no further fitting) ──────
        for scen_label, cor_key, mask_key in d2s.SCENARIOS:
            if scen_label not in TEST_SCENARIOS:
                continue
            na_key, nm_key = d2s.neighbor_keys(scen_label)
            X_te_occ = d2s.build_occ_X(te[cor_key].astype(np.float32), te['temporal'].astype(np.float32),
                                       te[na_key].astype(np.float32), te[nm_key].astype(np.float32), pidx)
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            te_gt_mm = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]
            y_te = (te_gt_mm[m] > WET_THRESH).astype(int)

            raw_p_te = occ_rf.predict_proba(X_te_occ[m])[:, 1]
            te_variants = {'raw': raw_p_te,
                           'platt': apply_platt(platt, raw_p_te),
                           'isotonic': apply_isotonic(iso, raw_p_te)}
            for name, p in te_variants.items():
                brier = float(brier_score_loss(y_te, p))
                ll = float(log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
                ece, bins_df = expected_calibration_error(p, y_te)
                bins_df['seed'] = seed; bins_df['variant'] = name
                bins_df['split'] = f'TEST:{scen_label}'
                reliability_rows.append(bins_df)

                row = occurrence_row(p, y_te, seed_thresholds[name])
                row.update(seed=seed, scenario=scen_label, variant=name,
                           threshold=seed_thresholds[name],
                           selected_calibration=(name == selected_variant),
                           ece=round(ece, 5))
                # occurrence_row() already computed brier/logloss at threshold-
                # independent level (on p, y directly) -- keep those (not
                # threshold-dependent) rather than recomputing.
                row['brier'] = round(brier, 5)
                row['logloss'] = round(ll, 5)
                test_metrics_rows.append(row)
            print(f'    [{scen_label:24s}] n_masked={int(m.sum())}')

    # ── Save ─────────────────────────────────────────────────────────────
    pd.DataFrame(val_cal_metrics_rows).to_csv(
        os.path.join(OUT_DIR, 'validation_calibration_metrics.csv'), index=False)
    pd.DataFrame(threshold_rows).to_csv(
        os.path.join(OUT_DIR, 'thresholds.csv'), index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, 'reliability_bins.csv'), index=False)
    test_df = pd.DataFrame(test_metrics_rows)
    test_df.to_csv(os.path.join(OUT_DIR, 'test_calibration_metrics.csv'), index=False)

    print(f'\n  -> {OUT_DIR}\\validation_split_manifest.csv')
    print(f'  -> {OUT_DIR}\\validation_calibration_metrics.csv')
    print(f'  -> {OUT_DIR}\\thresholds.csv')
    print(f'  -> {OUT_DIR}\\reliability_bins.csv')
    print(f'  -> {OUT_DIR}\\test_calibration_metrics.csv  ({len(test_df)} rows)')

    print('\n  TEST summary (mean over 3 seeds), RQ1 dose-response scenarios:')
    show = test_df[test_df.scenario.isin(
        ['10pct', 'mar_meteo', 'mnar_wet', 'mnar_intensity_moderate', 'mnar_intensity_severe'])]
    agg = show.groupby(['scenario', 'variant'])[['f1', 'brier', 'ece', 'bias']].mean().round(4)
    print(agg)
    print('=' * 70)


if __name__ == '__main__':
    main()

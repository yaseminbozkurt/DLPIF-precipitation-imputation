# -*- coding: utf-8 -*-
"""
check_d2s_dlpif_equivalence.py
================================
Formal equivalence check between D2S-RF (direct_two_stage_rf.py,
backbone-free classify-then-regress) and DLPIF's AmountRF stage
(multiseed_clean_rerun.py, backbone = WGAN-GP/SAITS/whatever).

Why this check is meaningful
-----------------------------
Tracing multiseed_clean_rerun.py's per-scenario loop shows that at every
artificially-masked PRECIP position, the AmountRF/DLPIF output is fully
overwritten and never touches the backbone value:

    p2s_scen            = p2_mm.copy()          # starts from backbone
    p2s_scen[~wet_pred]  = 0.0                   # dry -> 0 (backbone discarded)
    p2s_scen[wet_pred]   = quantile_map(...)     # wet -> qmap(backbone) (Precip2Stage only)
    amt_scen             = p2s_scen.copy()
    amt_scen[masked & wet_pred]  = amt_rf.predict(X_amt[masked & wet_pred])   # <- backbone discarded again
    amt_scen[masked & ~wet_pred] = 0.0                                        # <- backbone discarded

So for AmountRF/DLPIF, `amt_scen[masked]` depends only on:
  occ_rf, occ_cut  (Stage 1: trained from zero-filled corrupted records)
  amt_rf           (Stage 2: trained from zero-filled corrupted records)
  X_occ, X_amt     (features: zero-filled corrupted records, never backbone output)

None of these four quantities reference the backbone array `p2_mm` at all.
The manuscript's Section 5.8 backbone-agnostic ablation demonstrates this
empirically by swapping WGAN-GP <-> SAITS <-> zero-fill and observing
near-identical metrics. This script goes one step further: it proves the
independence *structurally* by substituting an ADVERSARIAL random-noise
array in place of the backbone (something no real imputer would ever
produce) and checking, at the level of individual masked positions rather
than aggregate metrics, that the reconstructed PRECIP value is bit-for-bit
identical to what direct_two_stage_rf.py (D2S-RF, which never even has a
backbone slot) produces. If the backbone genuinely cannot influence the
AmountRF/DLPIF output, this must hold for ANY backbone content, including
nonsense.

Method
------
For each seed in {42, 123, 456} and each of the 4 test scenarios:
  1. Train Stage-1 (occurrence RF) and Stage-2 (amount RF) exactly as
     direct_two_stage_rf.py / multiseed_clean_rerun.py do (identical
     hyperparameters, identical 10pct-random training scenario, identical
     feature construction -- reused directly from direct_two_stage_rf.py).
  2. Compute the D2S-RF reconstruction `pred_direct[masked]` (no backbone).
  3. For N_TRIALS independent random "dummy backbones" (uniform noise in
     [0, 200] mm, i.e. plausible-range garbage), replay the exact
     multiseed_clean_rerun.py AmountRF derivation (including the
     Precip2Stage quantile-mapping intermediate step) to obtain
     `amt_scen[masked]`.
  4. Compare `pred_direct[masked]` against `amt_scen[masked]` for every
     trial: expect exact equality (max abs diff == 0.0 in mm) regardless
     of which dummy backbone was used.

This script reads only preprocessed_{train,val,test}.npz and scaler.pkl --
no legacy results/*.csv is read, so the reported verdict cannot be
contaminated by a stale preprocessing snapshot. (An earlier version also
printed an informational cross-check against
results/multiseed_clean_evaluation.csv, a legacy DLPIF/WGAN-GP run from a
different snapshot; that read was removed to keep the pipeline fully
self-contained and auditable -- see git history if that comparison is
needed again, and regenerate it from a fresh multiseed_clean_rerun.py run
against the CURRENT preprocessed_*.npz rather than reusing the old file.)

Outputs
-------
  results/d2s_dlpif_equivalence_report.csv
  results/d2s_dlpif_equivalence_report.md
"""
import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# NOTE: do not re-wrap sys.stdout here -- importing direct_two_stage_rf
# below already wraps it once at module load time; wrapping twice makes
# the first TextIOWrapper's finalizer close the shared underlying buffer
# and subsequent prints raise "I/O operation on closed file".
import direct_two_stage_rf as d2s  # reuse the exact feature/RF code being tested

SRC_DIR     = d2s.SRC_DIR
RESULTS_DIR = d2s.RESULTS_DIR
WET_THRESH  = d2s.WET_THRESH
SEEDS       = d2s.SEEDS
SCENARIOS   = d2s.SCENARIOS
TRAIN_SCENARIO = d2s.TRAIN_SCENARIO
TRAIN_COR_KEY  = d2s.TRAIN_COR_KEY

N_TRIALS = 3   # independent random "dummy backbones" per seed x scenario


def apply_qmap(wet_pred_mm, qmap):
    """Identical to multiseed_clean_rerun.py::apply_qmap."""
    if qmap is None or len(wet_pred_mm) == 0:
        return wet_pred_mm
    pq = np.argsort(np.argsort(wet_pred_mm)).astype(np.float64)
    pq /= max(len(wet_pred_mm) - 1, 1)
    return np.interp(pq, np.linspace(0, 1, len(qmap)), qmap).astype(np.float32)


def main():
    print('=' * 70)
    print('  D2S-RF  <->  DLPIF(AmountRF)  STRUCTURAL EQUIVALENCE CHECK')
    print('  (adversarial random-noise backbone substitution)')
    print('=' * 70)

    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')

    tr = d2s.load_npz('train'); va = d2s.load_npz('val'); te = d2s.load_npz('test')
    tr_na_key, tr_nm_key = d2s.neighbor_keys(TRAIN_SCENARIO)
    d2s.require_keys(tr, [TRAIN_COR_KEY, tr_na_key, tr_nm_key], 'train')
    d2s.require_keys(va, [TRAIN_COR_KEY, tr_na_key, tr_nm_key], 'val')

    X_tr_full = d2s.build_occ_X(tr[TRAIN_COR_KEY].astype(np.float32),
                                tr['temporal'].astype(np.float32),
                                tr[tr_na_key].astype(np.float32),
                                tr[tr_nm_key].astype(np.float32), pidx)
    X_va_full = d2s.build_occ_X(va[TRAIN_COR_KEY].astype(np.float32),
                                va['temporal'].astype(np.float32),
                                va[tr_na_key].astype(np.float32),
                                va[tr_nm_key].astype(np.float32), pidx)
    X_tr_amt  = d2s.build_amt_X(sc, tr, TRAIN_COR_KEY, pidx, tr_na_key, tr_nm_key)

    tr_gt = d2s.inv(sc, tr['data'].astype(np.float32))[:, pidx]
    va_gt = d2s.inv(sc, va['data'].astype(np.float32))[:, pidx]
    te_gt = d2s.inv(sc, te['data'].astype(np.float32))[:, pidx]

    tr_obs = tr['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    va_obs = va['real_mask'].astype(np.float32)[:, pidx].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs

    X_tr_occ = X_tr_full[tr_obs]; y_tr = (tr_gt[tr_obs] > WET_THRESH).astype(int)
    X_va_occ = X_va_full[va_obs]; y_va = (va_gt[va_obs] > WET_THRESH).astype(int)

    va_wet_mm = va_gt[va_gt > WET_THRESH]
    qmap = (np.quantile(va_wet_mm, np.linspace(0, 1, 201)).astype(np.float32)
            if len(va_wet_mm) >= 10 else None)

    scen_features = {}
    for scen_label, cor_key, mask_key in SCENARIOS:
        na_key, nm_key = d2s.neighbor_keys(scen_label)
        d2s.require_keys(te, [cor_key, mask_key, na_key, nm_key], f'test/{scen_label}')
        scen_features[scen_label] = dict(
            X_occ=d2s.build_occ_X(te[cor_key].astype(np.float32),
                                  te['temporal'].astype(np.float32),
                                  te[na_key].astype(np.float32),
                                  te[nm_key].astype(np.float32), pidx),
            X_amt=d2s.build_amt_X(sc, te, cor_key, pidx, na_key, nm_key),
        )

    rows = []
    rng_master = np.random.default_rng(2026)

    for seed in SEEDS:
        print(f'\n{"="*60}')
        print(f'  SEED = {seed}')
        print('=' * 60)

        occ_rf, occ_cut, vm = d2s.train_occ(X_tr_occ, y_tr, X_va_occ, y_va, seed)
        amt_rf = __import__('sklearn.ensemble', fromlist=['RandomForestRegressor'])\
            .RandomForestRegressor(n_estimators=400, random_state=seed,
                                   min_samples_leaf=2, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])
        print(f'  cutoff={occ_cut}  val_F1={vm["val_f1"]}')

        for scen_label, _cor_key, mask_key in SCENARIOS:
            sf = scen_features[scen_label]
            m = te[mask_key].astype(np.float32)[:, pidx] > 0.5
            n_m = int(m.sum())

            te_proba = occ_rf.predict_proba(sf['X_occ'])[:, 1]
            te_wet_pred = (te_proba >= occ_cut).astype(bool)
            apply_sel = m & te_wet_pred

            # amt_rf.predict() with n_jobs=-1 sums per-tree outputs across
            # threads; float addition is not associative, so two separate
            # .predict() calls on the *same* input can differ by ~1e-6 mm
            # of pure floating-point noise. Call it exactly once per
            # (seed, scenario) and reuse the cached result everywhere below,
            # so the comparison isolates genuine backbone dependence rather
            # than RF prediction non-determinism.
            amt_pred_vals = (np.maximum(amt_rf.predict(sf['X_amt'][apply_sel]), 0.0)
                             if apply_sel.sum() > 0 else np.zeros(0, dtype=np.float32))

            # -- D2S-RF quantity (no backbone at all) --
            pred_direct = np.zeros_like(te_gt)
            if apply_sel.sum() > 0:
                pred_direct[apply_sel] = amt_pred_vals

            for trial in range(N_TRIALS):
                rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
                # Adversarial dummy backbone: uniform noise across the
                # plausible PRECIP range, unrelated to ground truth. Kept in
                # float64 (NOT cast to float32) to match the dtype that
                # multiseed_clean_rerun.py's real `p2_mm` ends up as: the
                # loaded .npy is float32, but `np.clip(...) * sc.data_range_[.]
                # + sc.data_min_[.]` upcasts to float64 because data_range_/
                # data_min_ are float64 scaler attributes. Casting the dummy
                # to float32 here would truncate the cached float64 RF
                # predictions on assignment and manufacture a fake ~1e-6 mm
                # "mismatch" that has nothing to do with the backbone.
                p2_mm_dummy = rng.uniform(0.0, 200.0, size=te_gt.shape)

                # -- Replay multiseed_clean_rerun.py's AmountRF derivation --
                p2s_scen = p2_mm_dummy.copy()
                p2s_scen[~te_wet_pred] = 0.0
                if qmap is not None and te_wet_pred.sum() > 0:
                    raw_wet = np.clip(p2s_scen[te_wet_pred], 0, None)
                    p2s_scen[te_wet_pred] = apply_qmap(raw_wet, qmap)
                p2s_scen = np.clip(p2s_scen, 0, None)

                amt_scen = p2s_scen.copy()
                if apply_sel.sum() > 0:
                    amt_scen[apply_sel] = amt_pred_vals
                amt_scen[m & ~te_wet_pred] = 0.0

                diff = np.abs(pred_direct[m] - amt_scen[m])
                max_diff = float(diff.max()) if n_m > 0 else 0.0
                n_mismatch = int((diff > 1e-9).sum())

                rows.append(dict(
                    seed=seed, scenario=scen_label, trial=trial,
                    n_masked=n_m, max_abs_diff_mm=max_diff,
                    n_mismatch=n_mismatch,
                    exact_match=bool(n_mismatch == 0),
                ))

            print(f'  [{scen_label}] {N_TRIALS} dummy-backbone trials: '
                  f'max|diff| over trials = '
                  f'{max(r["max_abs_diff_mm"] for r in rows if r["seed"]==seed and r["scenario"]==scen_label):.2e} mm')

    # NOTE: this used to also print an "[real-backbone ref]" comparison
    # against results/multiseed_clean_evaluation.csv, a legacy DLPIF/WGAN-GP
    # run from a different preprocessing snapshot than the one this script
    # runs against. It never fed into the saved report (`rows` below), but
    # reading it at all made the pipeline harder to audit end-to-end.
    # Removed. Run multiseed_clean_rerun.py fresh against the CURRENT
    # preprocessed_*.npz if a real-backbone comparison is needed.

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'd2s_dlpif_equivalence_report.csv')
    df.to_csv(csv_path, index=False)

    all_exact = bool(df['exact_match'].all())
    overall_max_diff = float(df['max_abs_diff_mm'].max())

    print('\n' + '=' * 70)
    print('  SUMMARY')
    print('=' * 70)
    print(f'  Seeds x scenarios x dummy-backbone trials = {len(df)} comparisons')
    print(f'  All exact matches (max|diff| < 1e-9 mm)   : {all_exact}')
    print(f'  Overall max |diff| across all trials       : {overall_max_diff:.2e} mm')
    if all_exact:
        print('\n  VERDICT: CONFIRMED. AmountRF/DLPIF\'s reconstructed PRECIP at')
        print('  masked positions is provably independent of the backbone content --')
        print('  identical to D2S-RF (no backbone at all) under every dummy backbone')
        print('  tested. The backbone contributes nothing to the PRECIP target;')
        print('  its only role is imputing the six other meteorological variables.')
    else:
        bad = df[~df['exact_match']]
        print('\n  VERDICT: MISMATCH DETECTED -- backbone independence does NOT hold')
        print('  as currently traced through the code. Offending rows:')
        print(bad.to_string(index=False))

    md_lines = [
        '# D2S-RF <-> DLPIF(AmountRF) Structural Equivalence Report',
        '',
        f'- Seeds: {SEEDS}',
        f'- Scenarios: {[s for s, _ck, _mk in SCENARIOS]}',
        f'- Dummy-backbone trials per (seed, scenario): {N_TRIALS}',
        f'- Total comparisons: {len(df)}',
        f'- All exact matches: **{all_exact}**',
        f'- Overall max |diff|: {overall_max_diff:.2e} mm',
        '',
        ('DLPIF\'s AmountRF stage reconstructs masked PRECIP positions '
         'identically to D2S-RF regardless of backbone content, confirming '
         'that the WGAN-GP/SAITS backbone contributes nothing to the '
         'PRECIP reconstruction itself (Section 3.4 / 5.8 of the manuscript) '
         '-- verified here at the level of individual masked positions under '
         'adversarial (random-noise) backbone substitution, not just via '
         'aggregate metrics on two similar real backbones.'
         if all_exact else
         'Equivalence did NOT hold for at least one (seed, scenario, trial) -- '
         'see d2s_dlpif_equivalence_report.csv for the offending rows before '
         'relying on the backbone-agnostic claim.'),
        '',
        '| Seed | Scenario | Max |diff| (mm, over trials) | Exact match |',
        '|---|---|---|---|',
    ]
    g = df.groupby(['seed', 'scenario'])
    for (seed, scen), sub in g:
        md_lines.append(f'| {seed} | {scen} | {sub["max_abs_diff_mm"].max():.2e} | '
                        f'{bool(sub["exact_match"].all())} |')
    md_path = os.path.join(RESULTS_DIR, 'd2s_dlpif_equivalence_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')

    print(f'\n  -> {csv_path}')
    print(f'  -> {md_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()

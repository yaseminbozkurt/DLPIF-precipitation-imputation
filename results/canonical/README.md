# Canonical Predictions & Metrics

This directory is the **single authoritative location** for row-level
predictions and evaluation metrics in this repository. It exists because
at least six scripts (`multiseed_clean_rerun.py`, `generate_clean_tables.py`,
`06_audit_predictions.py`, `direct_two_stage_rf.py`,
`baselines/train_saits_v2.py`, `04_evaluation.py`) each carried their own
copy of the wet-day / extreme-event metric formula, with small,
drift-prone differences between them (strict `>` vs `>=` at the 0.1 mm
wet-day threshold; a fixed 16.74 mm extreme threshold vs one recomputed
per run). Those differences are part of why the manuscript's Table 2,
Table 5, and Supplementary Table S2 do not agree with each other on every
row (see the DLPIF_Hydrology.docx review and
`audit_linear_interpolation.md` for concrete examples).

Everything under `results/canonical/` is produced by
`src/build_canonical_outputs.py`, which computes metrics exclusively
through `src/canonical_metrics.py` -- one implementation, imported, not
retyped -- and independently checked by `src/audit_canonical_outputs.py`
(see `canonical_audit.csv`, currently **841/841 checks PASS, 0 FAIL, 0
WARN**).

## Layout

```
results/canonical/
├── README.md                            (this file)
├── canonical_audit.csv                  independent audit, 841 checks
├── predictions/
│   └── {method}.csv                     row-level, ALL seeds x scenarios
│                                        for that method in a single file
│       columns: method, scenario, seed, row_index, station_id, date,
│                y_true, y_pred, evaluation_mask, source_file
└── metrics/
    ├── canonical_metrics_all.csv        long-form, one row per
    │                                    (method, scenario, seed)
    ├── canonical_metrics_summary.csv    mean +/- std across seeds, one
    │                                    row per (method, scenario)
    └── canonical_metrics_all.md         canonical_metrics_all.csv,
                                         human-readable
```

`y_true`/`y_pred` in `predictions/*.csv` are written at **full float64
precision** (pandas' default float serialisation, no rounding). This is
deliberate, not an oversight: a small number of ground-truth PRECIP
observations are exactly 0.1 mm, and the mm -> normalise -> mm round-trip
through the scaler perturbs them by ~1e-10, landing a few of them a hair
on one side of the 0.1 mm wet/dry threshold. Rounding to 6 decimals (tried
and rejected during development) collapses these back to exactly
`0.100000`, silently flipping their wet/dry classification and breaking
the `metrics_recompute_match` audit check for a handful of positions per
scenario. Full precision keeps the predictions CSV and
`canonical_metrics_all.csv` bit-for-bit consistent with each other.

## Registered methods (9/9)

All nine evaluated against the **same** fresh
`preprocessed_{train,val,test}.npz` / `scaler.pkl` snapshot in `src/`
(01_data_preprocessing.py rerun 2026-07-20).

| Method | Seeds | Source |
|---|---|---|
| `Mean` | deterministic (1) | `baseline_results.pkl` -- per-station training-mean fill |
| `Linear` | deterministic (1) | `baseline_results.pkl` -- station-wise linear interpolation (see `audit_linear_interpolation.md` for the cross-station-leakage fix this replaced) |
| `KNN` | deterministic (1) | `baseline_results.pkl` -- k-nearest-neighbour imputation |
| `MICE` | deterministic (1) | `baseline_results.pkl` -- multiple imputation by chained equations |
| `AmountRF_DLPIF` | 42, 123, 456 | `gan_imputed_test_modeB_seed{seed}_msclean_amountrf_{scenario}.npy` -- full DLPIF (Stage-1 occurrence RF + Stage-2 amount RF) |
| `DirectTwoStageRF` | 42, 123, 456 | `direct_two_stage_rf_test_seed{seed}_{scenario}.npy` -- backbone-free Stage-1+Stage-2 RF baseline |
| `SAITS` | 42, 123, 456 | `saits_test_seed{seed}_{scenario}.npy` -- self-attention imputer, produced by `baselines/train_saits_v2.py` + `baselines/repackage_saits_outputs.py` |
| `WGANGP_raw` | 42, 123, 456 | `gan_imputed_test_modeB_{scenario}_seed{seed}.npy` -- WGAN-GP continuous backbone, raw generator output for the PRECIP channel, no post-hoc calibration |
| `WGANGP_PrecipFix` | 42, 123, 456 | `gan_imputed_test_modeB_{scenario}_seed{seed}_precipfix.npy` -- same WGAN-GP run, with quantile-mapping / occurrence-threshold precip calibration (`precip_calibration.py`) applied |

**WGAN-GP reinstated in the registry (2026-08-02).** `WGANGP_raw` and
`WGANGP_PrecipFix` were removed on 2026-07-23 because the only trained
WGAN-GP checkpoints at the time (2026-07-21, `gpu_transfer_package_scenario_wgan.zip`)
predated the 2005-2023 dataset restriction and produced an (11180, 7)
test-shape array incompatible with the current (4168, 7) canonical test
set -- see `gpu_transfer_package_scenario_wgan_v2.zip`'s `README_GPU_RUN.md`
for the full incompatibility writeup. A fresh GPU run against the
corrected snapshot (Tesla T4, `torch==2.11.0+cu128`, all 3 seeds) produced
output at the correct `(4168, 7)` shape, hash-verified against its own
manifest before being copied into `src/`; both methods are now registered,
built, and independently audited exactly like every other method here.
`Precip2Stage` ("WGAN-GP raw + occurrence correction + quantile mapping")
was never reconstructed: its exact definition is not recoverable from
this file's own history (it is untracked in git; no prior committed
version exists) and it is not referenced anywhere in the current
manuscript text, so it remains deliberately unregistered rather than
guessed at; its old `predictions/Precip2Stage.csv` is an orphaned
artifact from an earlier pipeline version and should not be cited.

Every loader reads only `preprocessed_test.npz`, `scaler.pkl`, and one of
the raw artifacts above. No `results/*.csv` or `src/*.csv` written by any
other script's own metric implementation is read as a value source --
enforced, not just claimed, by the audit's `no_legacy_file_read` check
(a whitelist of allowed `source_file` patterns).

**Notable empirical result:** `AmountRF_DLPIF` and `DirectTwoStageRF`
produce **bit-identical** `y_pred` at every one of the 8,787
seed x scenario x masked-position rows (2,929 masked positions -- 416
at 10pct, 833 at 20pct, 840 at block7d, 840 at block30d -- x 3 seeds;
`dlpif_vs_direct_equivalence` audit check, max\|diff\| = 0.0, matching
`canonical_audit.csv` and manuscript Section 3.4 / 5.8 / Table 9
exactly). This is the real-production-data confirmation of the
architectural proof in
`check_d2s_dlpif_equivalence.py` (which used an adversarial random-noise
backbone substitution): the WGAN-GP backbone contributes nothing to
DLPIF's PRECIP reconstruction at masked positions -- Stage-1/Stage-2 RF
features are built entirely from the zero-filled raw corrupted records,
never from the backbone's output. Registering a **real, current** WGAN-GP
backbone (`WGANGP_raw`/`WGANGP_PrecipFix` above) changes none of
`AmountRF_DLPIF`'s numbers -- as this invariance predicts -- and adds a
second, independent real-data confirmation of it, alongside two new
comparison-baseline rows in their own right: `WGANGP_raw` shows a large
positive wet-day frequency bias (+0.63 to +0.67 across scenarios,
mean-across-seed F1 0.49-0.54, i.e. a severe drizzle artefact -- see
`metrics/canonical_metrics_summary.csv`), and `WGANGP_PrecipFix`
over-corrects into a negative bias instead (-0.04 to -0.10, F1 0.31-0.41),
neither approaching `AmountRF_DLPIF`'s occurrence-level accuracy (F1
0.77-0.81 across scenarios).

## Audit

`src/audit_canonical_outputs.py` re-derives what it can from the raw
source files rather than trusting the canonical CSVs it's checking --
re-reading your own output proves nothing. Checks, by category:

| Category | Checks |
|---|---|
| `completeness` (129 checks) | all expected (method, scenario, seed) groups exist; no unexpected extras; equivalent group-existence checks for the Table 4/6/7/8, LOSO, and multi-mask-variance analyses (`table4_groups_exist`, `table6_groups_exist`, `table7_groups_exist`, `table8_groups_exist`, `loso_groups_exist`, `multimask_groups_exist`, `multimask_seeds_unique`) |
| `integrity` (304 checks) | no duplicate prediction rows; prediction row count matches a freshly-recomputed `art_mask` count; no NaN/Inf; every `row_index` is a genuine masked position; `n_evaluated` matches the fresh mask count |
| `consistency` (305 checks) | metrics recomputed from `predictions/*.csv` match `canonical_metrics_all.csv` to 1e-9; the false-wet-rate/dry-spell/spatial-correlation tables (Table 4/7/8), the cross-scenario generalisation ablation (Table 6, Model A), and the LOSO/multi-mask-variance value ranges are independently recomputed from the same raw sources and checked for an exact match or valid range |
| `provenance` (15 checks) | every `source_file` in every predictions CSV matches the fresh-artifact whitelist; no legacy/archive filename appears anywhere, including a static source-code grep over the Table-4/7/8 and LOSO/multi-mask analysis scripts themselves (`no_legacy_file_read`, `stage0_no_legacy_file_read`, `table6_loso_no_legacy_file_read`, `multimask_no_legacy_file_read`) |
| `equivalence` (12 checks) | `AmountRF_DLPIF` vs `DirectTwoStageRF`, row-index-matched, max\|diff\| -- one check per (scenario, seed) |
| `traceability` (76 checks) | every `canonical_metrics_all.csv` row's `source_prediction_file` points to an existing, matching `predictions/*.csv` |

Check counts above are from the live `canonical_audit.csv` (841 rows
total; up from 589 after `WGANGP_raw`/`WGANGP_PrecipFix` were reinstated
2026-08-02, adding their own completeness/integrity/consistency/
provenance/traceability checks across all 3 seeds x 4 scenarios). An
earlier `backbone_integrity` category -- WGAN-GP observed-cell
byte-for-byte preservation, per-scenario array uniqueness -- applied only
to the pre-2026-07-23 WGAN-GP run and no longer runs; it is not one of
the six categories currently produced.

Current result: **841/841 PASS, 0 FAIL, 0 WARN** (`canonical_audit.csv`).
Re-run via `python audit_canonical_outputs.py` after any change to
`build_canonical_outputs.py` or its inputs.

## How to add a method

In `src/build_canonical_outputs.py`, add a loader function with signature

```python
def _load_my_method(seed, scen_label, mask_key, sc, pidx, te, dates, sids):
    # Raise MissingCanonicalInput if the source artifact isn't present --
    # fail fast, never silently skip or fall back to an archived file.
    ...
    return _extract_masked_rows(sc, pidx, te, dates, sids, mask_key,
                                arr_norm, source_file)
```

and register it (with its seed list -- `SEEDS` or `['deterministic']`) in
`METHODS`. Everything else (row-level CSV, `canonical_metrics_all.csv`,
`canonical_metrics_summary.csv`, the `.md` rendering) is generated
generically. Re-run `audit_canonical_outputs.py` afterwards -- it will
automatically pick up the new method from `METHODS` and check it the same
way as the other eight.

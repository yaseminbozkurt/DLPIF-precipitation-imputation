# Pipeline Audit History

This file is a git-tracked provenance record for post-hoc corrections made to
the analysis pipeline after initial results were generated. It exists so that
"before" states referenced only by filename convention (e.g. `*_PRE_BETAFIX*`)
have a corresponding entry in version-controlled history, not just a filename.

## 2026-08-11 — MNAR-Intensity beta-dose recalibration

**What changed.** The `beta` values used for the two MNAR-Intensity stress
doses in `src/build_mar_mnar_scenarios.py` (Section 5.1.3 of the manuscript)
were recalibrated. Artifacts from the run prior to this fix are preserved,
untouched, in `src/_mnar_beta_fix_backup_20260811/` (`preprocessed_test_PRE_BETAFIX.npz`,
`preprocessed_val_PRE_BETAFIX.npz`, and the corresponding pre-fix pickled
models) and in `results/rq1_mechanism_diagnostics/rq1_master_table_all_methods_PRE_BETAFIX.csv`.

**Numeric effect on the DLPIF RQ1 table** (`rq1_master_table_all_methods.csv`,
mean over 3 seeds):

| Scenario | Metric | Pre-fix | Post-fix (current, cited in manuscript) |
|---|---|---|---|
| MNAR-Intensity-Moderate | F1 | 0.895 | 0.876 |
| MNAR-Intensity-Moderate | RMSE_wet (mm) | 7.559 | 6.421 |
| MNAR-Intensity-Moderate | RMSE_p95 (mm) | 17.717 | 16.130 |
| MNAR-Intensity-Severe | F1 | 0.921 | 0.938 |
| MNAR-Intensity-Severe | RMSE_wet (mm) | 9.974 | 9.794 |
| MNAR-Intensity-Severe | RMSE_p95 (mm) | 21.520 | 21.393 |

MCAR-10, MAR-Meteo, and MNAR-Wet are unaffected (the beta parameter only
enters the MNAR-Intensity weighting function, Section 5.1.3).

**Why this is not post-hoc test tuning.** Per Section 5.1.3, both Moderate
and Severe beta values are selected and verified exclusively on the
validation partition's own p95-observed cells (a pre-specified 30-75%
validation p95-conditional-missing-rate band for Moderate; the smallest
grid value exceeding Moderate's rate for Severe), then applied to the test
partition unconditionally. The recalibration corrected the beta-selection
procedure itself (i.e., which grid values satisfied the pre-specified
validation band), not a selection informed by test-partition outcomes; the
post-fix run was, like the pre-fix run, evaluated on test data only once,
after the doses were frozen from validation alone.

**Caveat.** This record is reconstructed from file timestamps and the
diff between the two CSVs; no separate commit-by-commit log of the fix
itself was kept at the time. Going forward, corrections of this kind
should be committed with an explicit message referencing this file rather
than relying on a `_PRE_*`/`_BACKUP_*` filename convention alone.

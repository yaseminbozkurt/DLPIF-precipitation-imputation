# DLPIF — Decoupled Learning–Physical Imputation Framework

> *"Beyond RMSE: Occurrence–Amount Decoupling for Hydrologically Consistent Precipitation Imputation"*  
> Target journal: *MDPI Hydrology*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

Standard multivariate imputation models optimise for RMSE across all variables and time steps. When applied to precipitation, this produces **drizzle-like artefacts**: low-intensity nonzero values persist during structurally dry periods, inflating wet-day frequency while underestimating extreme events.

**DLPIF** addresses this mismatch through a two-stage post-processing layer on top of a continuous multivariate base imputer:

1. **Stage 1 — Occurrence Classification** — A Random Forest classifier predicts wet/dry state for each missing day using 25 meteorological, temporal, and spatial-context features. Local precipitation is **excluded** from Stage 1 inputs to prevent circular target leakage.

2. **Stage 2 — Wet-Day Amount Estimation** — A Random Forest regressor estimates precipitation amounts only at positions predicted wet in Stage 1. Local precipitation is **hard-zeroed** in the feature matrix as a leakage guard.

---

## Key Results

Source of truth: `results/canonical/metrics/canonical_metrics_summary.csv`
(11 registered methods, audited by `audit_canonical_outputs.py`, 1253/1253
checks PASS — see `results/canonical/README.md`). Seeded methods (DLPIF,
SAITS, WGAN-GP) are reported as **mean ± std** across three independent
seeds (42, 123, 456); Linear is deterministic.

| Scenario | Method | Bias | F1 | Wet RMSE (mm) | Extreme RMSE (mm) |
|---|---|---|---|---|---|
| Random 10% | **DLPIF** | +0.038 ± 0.013 | **0.806 ± 0.002** | **5.40 ± 0.03** | **17.51 ± 0.16** |
| Random 10% | Linear (station-wise) | +0.156 | 0.674 | 6.92 | 24.55 |
| Random 10% | SAITS | +0.060 ± 0.056 | 0.695 ± 0.009 | 7.27 ± 0.06 | 25.77 ± 0.12 |
| Random 10% | WGAN-GP (calibrated) | −0.043 ± 0.051 | 0.414 ± 0.094 | 7.57 ± 0.10 | 26.27 ± 0.12 |
| Block 30d | **DLPIF** | +0.033 ± 0.013 | **0.794 ± 0.011** | **6.08 ± 0.02** | **20.36 ± 0.09** |
| Block 30d | Linear (station-wise) | +0.099 | 0.412 | 8.60 | 27.58 |
| Block 30d | SAITS | −0.030 ± 0.066 | 0.615 ± 0.043 | 8.31 ± 0.03 | 28.59 ± 0.07 |
| Block 30d | WGAN-GP (calibrated) | −0.100 ± 0.047 | 0.311 ± 0.023 | 8.51 ± 0.03 | 28.92 ± 0.09 |

**DLPIF is the only method on the Pareto front** (low wet RMSE + high occurrence F1) across all four missingness scenarios. Full per-scenario table for all 11 methods (Mean, Linear, KNN, MICE, DLPIF, DirectTwoStageRF, SingleStageRF, MissingnessIndicatorRF, SAITS, WGAN-GP raw, WGAN-GP calibrated): `results/canonical/metrics/canonical_metrics_summary.csv`.

**WGAN-GP** is included in two roles. First, and primarily, as the
optional continuous base imputer for the six *non*-precipitation
variables (see Architecture below) — DLPIF's PRECIP output does not
depend on it (proven in `results/canonical/README.md`, "Notable
empirical result"). Second, as of 2026-08-02, as two real
comparison-baseline rows in the table above (`WGANGP_raw`,
`WGANGP_PrecipFix`): raw WGAN-GP output for the PRECIP channel is a
severe drizzle artefact (bias +0.63 to +0.67, F1 0.49-0.54 across all
four scenarios — worse than every classical baseline except Mean); the
post-hoc quantile-mapping calibration (`precip_calibration.py`,
`WGANGP_PrecipFix` above) corrects the sign of the bias but overshoots
into under-prediction and still trails DLPIF by a wide margin on every
metric. Full per-scenario numbers for both:
`results/canonical/metrics/canonical_metrics_summary.csv`. Any WGAN-GP
PRECIP numbers you may find in `results/summary_mean_std_wide.md` or
other pre-canonical files predate the 2005–2023 restriction, the
metric-threshold fixes below, and this reinstated GPU run — treat them
as historical, not current.

---

## Architecture

DLPIF is **two parallel, non-communicating branches** operating on the
same raw corrupted record — not a sequential pipeline. This was a
deliberate redesign (see `figures/fig1_dlpif_architecture.png`): an
earlier sequential version fed the base imputer's PRECIP output into
Stage 1, until an architectural-invariance test proved Stage 1/Stage 2
never actually use it (`check_d2s_dlpif_equivalence.py`; real-data
confirmation in `results/canonical/README.md`, 8,787/8,787 masked
positions bit-identical to a backbone-free reimplementation).

```
Raw corrupted meteorological records (7 variables, incl. PRECIP)
         |
         +-------------------------------+-------------------------------+
         |                                                                |
         v                                                                v
[Base imputer -- any continuous model]                [DLPIF occurrence-amount pathway]
 e.g. WGAN-GP Mode B: dual-branch spatio-        reads ONLY the raw corrupted record --
 temporal BiLSTM (temporal) + FFN (spatial       never the base imputer's output
 neighbour context)                              (proven independent, not just designed to be)
 Reconstructs the 6 NON-PRECIP variables                            |
 (TMIN, TMEAN, TMAX, RH_MEAN, P_MEAN,                    [Stage 1: Occurrence RF]
  WIND_MEAN) only -- never touches PRECIP                25-feature classifier; local PRECIP
         |                                                excluded; threshold = validation-F1-
         |                                                optimal (grid 0.20-0.80, step 0.02)
         |                                                                |
         |                                                    [Stage 2: Amount RF]
         |                                                26-feature regressor, wet-predicted
         |                                                positions only; local PRECIP hard-
         |                                                zeroed to prevent leakage
         |                                                                |
         |                                                [Hydrological constraints]
         |                                                dry -> 0.0 mm; negatives clipped
         |                                                                |
         v                                                                v
   6-variable reconstruction  ─────────────────────────────────▶  merged into the final
                                                                   7-variable state --
                                                                   PRECIP comes entirely
                                                                   from the DLPIF pathway
```

---

## Repository Structure

```
DLPIF-precipitation-imputation/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── 01_data_preprocessing.py       # Data loading, validation, adjacency, normalisation, missingness scenarios
│   ├── 02_wgan_gp_imputation.py       # Base imputer (WGAN-GP Mode B) for the 6 non-PRECIP variables (optional, GPU-only)
│   ├── 03_baseline_imputation.py      # Mean, linear interpolation, KNN, MICE baselines
│   ├── multiseed_clean_rerun.py       # DLPIF Stage 1/Stage 2 training (per-seed occurrence + amount RFs)
│   ├── direct_two_stage_rf.py         # Backbone-free Stage-1+Stage-2 RF baseline (proves DLPIF's backbone-independence)
│   ├── check_d2s_dlpif_equivalence.py # Adversarial backbone-substitution invariance test
│   ├── compute_p95_threshold.py       # Derives the extreme-event threshold from train+val only
│   ├── canonical_metrics.py           # Single metric implementation (imported everywhere, not retyped)
│   ├── build_canonical_outputs.py     # Builds results/canonical/ — the authoritative predictions + metrics
│   ├── audit_canonical_outputs.py     # Independent re-derivation audit of results/canonical/ (1253 checks)
│   ├── canonical_bootstrap_significance.py  # Block-bootstrap significance testing
│   ├── canonical_loso.py              # Leave-one-station-out spatial cross-validation
│   ├── canonical_generalization_ablation.py # Cross-scenario (10%- vs 20%-trained) generalisation
│   ├── canonical_stage_ablation.py    # Stage-2 marginal-contribution ablation
│   ├── canonical_multimask_variance.py      # Multi-mask sampling-variance experiment
│   ├── canonical_false_wet_analysis.py, canonical_dry_spell.py, canonical_spatial_correlation.py
│   │                                   # Canonical drizzle / dry-spell / spatial-correlation tables
│   │
│   ├── build_mar_mnar_scenarios.py    # RQ1 -- MAR-Meteo / MNAR-Wet / MNAR-Intensity mechanism-shift masks
│   ├── build_network_block_scenario.py # netblock30d network-wide simultaneous-missingness mask
│   ├── calibrate_occurrence_probability.py # RQ2 -- Platt/Isotonic Stage-1 probability calibration
│   ├── audit_model_identity.py        # RQ2 -- confirms calibration doesn't mutate the Stage-1 classifier (hash check)
│   ├── stage2_conformal_uq.py         # RQ3 -- Stage-2 split-conformal prediction intervals (primary, symmetric)
│   ├── stage2_conformal_uq_cqr.py     # RQ3 robustness check -- QRF-based conformalized quantile regression
│   ├── stage2_conformal_uq_saits.py   # RQ3 backbone-generalisation check -- same construction applied to SAITS
│   ├── context_availability_diagnostic.py # RQ4a -- A_local/A_neighbour availability diagnostics
│   ├── graded_context_loss.py         # RQ4a -- graded local/neighbour/joint context-loss ablation
│   ├── rq4b_gate_selection.py         # RQ4b -- interaction-gate selection on validation data only
│   ├── rq4b_apply_gate.py             # RQ4b -- frozen gate applied once to all test scenarios
│   ├── confound_isolation_reduced_features.py # isolates whether external_validation_ohio's sharper
│   │                                   #   neighbour-loss severity is a feature-set or a network effect
│   ├── rq4b_gate_sensitivity.py       # applies all 28 validation-tied gate candidates to the test
│   │                                   #   scenarios -- how much does the tie-break choice matter?
│   ├── rf_hyperparameter_sensitivity.py # +/-50% n_estimators sweep for Stage 1/Stage 2 -- are the
│   │                                   #   fixed, manually-chosen tree counts fragile?
│   │
│   └── baselines/
│       ├── saits_data_adapter.py
│       ├── train_saits_v2.py
│       ├── repackage_saits_outputs.py
│       ├── repackage_saits_val_10pct.py # inference-only SAITS VAL-CAL predictions for stage2_conformal_uq_saits.py
│       ├── evaluate_saits.py
│       └── run_multiseed_saits_v2.py
│
├── figures/
│   ├── generate_fig1_architecture.py  # Figure 1 — parallel-branch DLPIF architecture diagram
│   ├── fig1_dlpif_architecture.png/.pdf
│   └── figures/                       # Additional output figures (PNG, PDF, SVG)
│
└── results/
    ├── canonical/                     # ★ single authoritative source — see results/canonical/README.md
    │   ├── canonical_audit.csv        #   independent audit, 1253/1253 PASS
    │   ├── predictions/{method}.csv   #   row-level predictions, all 11 registered methods
    │   ├── metrics/canonical_metrics_summary.csv  #   mean ± std per (method, scenario) — cite this
    │   └── analysis/                  #   bootstrap significance, LOSO, ablations, multi-mask variance
    ├── audit_linear_interpolation.md  # Cross-station-leakage fix for the Linear baseline
    ├── d2s_dlpif_equivalence_report.md # Backbone-independence equivalence proof (numerical log)
    │
    ├── rq1_mechanism_diagnostics/     # RQ1 -- occurrence/amount performance under MAR/MNAR mechanism shift
    ├── rq2_calibration/               # RQ2 -- Stage-1 calibration metrics, reliability bins, VAL-CAL/VAL-SELECT split
    ├── rq3_conformal/                 # RQ3 primary -- symmetric split-conformal oracle/end-to-end coverage
    ├── rq3_conformal_cqr/             # RQ3 robustness check -- QRF-CQR oracle/end-to-end coverage (comparable
    │                                   #   cell-for-cell with rq3_conformal/; see stage2_conformal_uq_cqr.py)
    ├── rq3_conformal_saits/           # RQ3 backbone-generalisation check -- SAITS oracle coverage, same
    │                                   #   construction as rq3_conformal/; see stage2_conformal_uq_saits.py
    └── rq4_context_availability/      # RQ4a/b -- graded context-loss ablation, gate selection, gate applied to test

Files directly under `results/` (`clean_full_evaluation.csv`,
`multiseed_clean_evaluation.csv`, `summary_mean_std.csv`,
`analysis_false_wet_rate.csv`, etc.) are **pre-canonical, legacy
outputs** kept for provenance; they predate the 2005–2023 restriction,
the Linear station-wise fix, and the p95-threshold correction described
below, and should not be cited as current results. Use
`results/canonical/` for anything you report externally.
```

---

## Setup

```bash
git clone https://github.com/yaseminbozkurt/DLPIF-precipitation-imputation.git
cd DLPIF-precipitation-imputation
pip install -r requirements.txt
```

### Dependencies

| Package | Version |
|---|---|
| scikit-learn | 1.8.0 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| matplotlib | 3.10.9 |
| torch | >= 2.0 |
| joblib | >= 1.3 |

---

## Data

### Study Area

- **Region:** Kütahya, Turkey
- **Stations:** 4 meteorological stations (KÜTAHYA, TAVŞANLI, SİMAV, GEDİZ)
- **Raw MGM archive:** 1973–2023 (daily resolution)
- **Analysis period:** **2005–2023 only** (19 years, 27,756 station-day records). An audit found a structural discontinuity in PRECIP recording around 2005 — observed-value rates rise from ~30% pre-2005 to >97% from 2005 onward, with wet-day frequency among observed values falling correspondingly — consistent with a change in station recording convention rather than a real climatic shift. Mixing the two regimes would bias any imputation-difficulty comparison, so all splits, thresholds, and results below use 2005–2023 only.
- **Variables (7):** `TMIN`, `TMEAN`, `TMAX`, `RH_MEAN`, `P_MEAN`, `WIND_MEAN`, `PRECIP`

### Data Availability

The meteorological observations used in this study were obtained from the Turkish State Meteorological Service (MGM) under institutional permission. Raw data cannot be publicly redistributed. Users who obtain the required data through MGM's permission procedures should place the following files in `src/`:

```
preprocessed_train.npz
preprocessed_val.npz
preprocessed_test.npz
scaler.pkl
adjacency.pkl
baseline_results.pkl                                        (Step 3 output)
gan_imputed_test_modeB_{scenario}_seed{42,123,456}.npy               (Step 2 output -- full 7-variable reconstruction; non-PRECIP columns optional for DLPIF, PRECIP column required for the WGANGP_raw baseline)
gan_imputed_test_modeB_{scenario}_seed{42,123,456}_precipfix.npy     (Step 2 output -- required for the WGANGP_PrecipFix baseline)
direct_two_stage_rf_test_seed{42,123,456}_{scenario}.npy    (Step 4 output -- required for DLPIF/PRECIP)
saits_test_seed{42,123,456}_{scenario}.npy                  (Step 4 output -- required for the SAITS baseline)
```

### Data Splits

Strict chronological split of the 2005–2023 dataset (no shuffling); zero date overlap verified by assertion.

| Split | Fraction | Dates | Station-days | PRECIP observed |
|---|---|---|---|---|
| Train | 70% | 2005-01-01 – 2018-04-19 | 19,428 | 19,275 (99.2%) |
| Validation | 15% | 2018-04-20 – 2021-02-22 | 4,160 | 4,159 (100.0%) |
| Test | 15% | 2021-02-23 – 2023-12-31 | 4,168 | 4,168 (100.0%) |

Role: Train fits models and the scaler; Validation selects thresholds, does early stopping, and fits quantile mapping; Test is evaluated once, with no decisions made on it.

### Missingness Scenarios

| Scenario | Type | Rate |
|---|---|---|
| `10pct` | Random uniform | 10% |
| `20pct` | Random uniform | 20% |
| `block7d` | Consecutive 7-day blocks | 20% |
| `block30d` | Consecutive 30-day blocks | 20% |

---

## Reproduction Steps

### Step 1 — Preprocessing

```bash
python src/01_data_preprocessing.py
```

Loads data, validates physical bounds, builds kNN-Gaussian adjacency (Haversine + elevation, k=2), adds temporal features, splits chronologically, fits MinMaxScaler on training data, generates all missingness masks.

**Outputs:** `preprocessed_{train,val,test}.npz`, `scaler.pkl`, `adjacency.pkl`, `missingness_report.csv`

### Step 2 — WGAN-GP Base Imputer (optional for DLPIF; required for the WGAN-GP comparison rows)

```bash
python src/02_wgan_gp_imputation.py --seed 42  --mode B
python src/02_wgan_gp_imputation.py --seed 123 --mode B
python src/02_wgan_gp_imputation.py --seed 456 --mode B
```

Mode B uses a dual-branch BiLSTM (temporal) + FFN (spatial). Training: 60 epochs, batch=128, n_critic=3, λ_GP=10, λ_recon=10, Adam lr=1e-4, early stopping patience=10. **Requires a CUDA GPU** — this repository's own environment has none, so this step was run externally (Tesla T4, via `gpu_transfer_package_scenario_wgan_v2.zip`) and its outputs copied back into `src/`.

This step reconstructs all 7 variables, but serves two independent roles:
reconstructing the **six non-precipitation variables**
(TMAX, TMIN, TMEAN, RH_MEAN, P_MEAN, WIND_MEAN) is not required to
reproduce any DLPIF PRECIP result in Key Results above or in the
manuscript — DLPIF's Stage 1/Stage 2 read only the raw corrupted record
(Step 1's output), never this step's output (see Architecture); the
PRECIP column of this same output, however, **is** required for the
`WGANGP_raw`/`WGANGP_PrecipFix` comparison-baseline rows in Key Results.
Skip this step entirely if you only want DLPIF/PRECIP reconstruction and
downstream hydrological metrics and don't need the WGAN-GP baseline rows.

**Outputs per seed:** `gan_model_modeB_seed{s}.pt`, `gan_imputed_test_modeB_{scenario}_seed{s}.npy` (+ `_precipfix.npy`) per scenario

### Step 3 — Baseline Methods

```bash
python src/03_baseline_imputation.py
```

Fits Mean, Linear Interpolation, KNN (k=5), and MICE on all scenarios. **Output:** `baseline_results.pkl`

### Step 4 — DLPIF Stage 1/Stage 2 Training

```bash
python src/multiseed_clean_rerun.py
```

Per seed: trains Stage 1 RF classifier (300 trees, balanced class weights, validation-tuned threshold), trains Stage 2 RF regressor (400 trees, wet-day subsetting), evaluates over all scenarios. Also run `src/direct_two_stage_rf.py` (backbone-free equivalent) and `src/baselines/train_saits_v2.py` + `repackage_saits_outputs.py` (SAITS), plus Step 2 (WGAN-GP) if you want the `WGANGP_raw`/`WGANGP_PrecipFix` rows, to populate all 9 canonical methods.

**Outputs:** `multiseed_clean_evaluation.csv`, `occurrence_clean_seed_summary.csv` (legacy-format; see Step 5 for the authoritative rebuild).

### Step 5 — Build and Audit the Canonical Results (authoritative)

```bash
python src/build_canonical_outputs.py     # rebuilds results/canonical/predictions + metrics
python src/audit_canonical_outputs.py     # independently re-derives and checks every number
```

This is the step that produces the numbers in **Key Results** above and
in the manuscript. It reads only `preprocessed_test.npz`, `scaler.pkl`,
and the raw per-method prediction artefacts from Steps 3–4 — never any
other script's own metric computation — through the single
`canonical_metrics.py` implementation. The audit should report
`1253/1253 PASS, 0 FAIL, 0 WARN`; investigate before trusting any result
if it does not.

### Step 6 — Statistical Robustness Suite

```bash
python src/canonical_bootstrap_significance.py    # block-bootstrap significance (Table 4 refs)
python src/canonical_loso.py                      # leave-one-station-out spatial CV
python src/canonical_generalization_ablation.py   # 10%-trained vs 20%-trained
python src/canonical_stage_ablation.py            # Stage-2 marginal contribution
python src/canonical_multimask_variance.py        # sampling-variance across 10 mask draws
python src/canonical_false_wet_analysis.py        # drizzle / false-wet-rate table
python src/canonical_dry_spell.py                 # dry-spell / CDD reconstruction table
python src/canonical_spatial_correlation.py       # inter-station correlation preservation
```

All eight write to `results/canonical/analysis/` and are covered by the
same `audit_canonical_outputs.py` run in Step 5.

### Step 7 — Figure 1

```bash
python figures/generate_fig1_architecture.py
```

---

## Methodological Notes

### Leakage Prevention

| Stage | Guard |
|---|---|
| Stage 1 | Local `PRECIP` removed from feature matrix before concatenation |
| Stage 2 | Local `PRECIP` hard-zeroed (set to 0.0 mm) |
| Threshold selection | Validation F1 only — no test-set information used |
| Scaler | Fitted on training data only |
| Quantile map | Built from validation wet-day ground truth |

### Wet-Day Threshold

**0.1 mm/day** (WMO standard). The occurrence model's decision threshold is tuned per seed on the validation set (grid 0.20–0.80, step 0.02), maximising F1.

### Extreme-Event Threshold

**p95 = 19.20 mm/day**, computed from the training+validation wet-day
precipitation distribution only (2005–2023 dataset). Extreme metrics
(MAE p95, RMSE p95) are evaluated at masked positions where ground-truth
precipitation is at or above this threshold.

An earlier version of this pipeline used a fixed 16.74 mm threshold that
an internal audit found had been inadvertently calibrated on the *test*
partition rather than train+validation as documented, and computed on a
now-superseded pre-restriction (1973–2023) dataset. It was corrected to
19.20 mm, recomputed strictly from train+validation data on the current
2005–2023 dataset (`compute_p95_threshold.py`; single implementation in
`canonical_metrics.py`). 16.74 mm should not appear in any current table
or figure — if you see it, it is from a pre-audit file.

### Occurrence Model

```python
RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                       class_weight='balanced', random_state=seed, n_jobs=-1)
```

### Amount Model

```python
RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                      random_state=seed, n_jobs=-1)
```

---

## External Validation

[`external_validation_ohio/`](external_validation_ohio/) — a first,
scoped-down replication of RQ1 and RQ4a on four independent GHCN-Daily
stations in Ohio, USA, with no institutional data-use restriction. See
that directory's README for scope, results, and reproduction steps.

## Provenance

Post-hoc corrections to the analysis pipeline (e.g. the 2026-08-11
MNAR-Intensity beta-dose recalibration) are documented in
[`PIPELINE_AUDIT_HISTORY.md`](PIPELINE_AUDIT_HISTORY.md).

## Citation

```bibtex
@article{bozkurt2026dlpif,
  title   = {Beyond RMSE: Occurrence--Amount Decoupling for Hydrologically Consistent Precipitation Imputation},
  author  = {Bozkurt, Yasemin and Serttaş, Soydan and Bakır, Çiğdem},
  journal = {Hydrology},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

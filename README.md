# DLPIF — Decoupled Latent–Physical Imputation Framework

**Official clean, leakage-safe codebase for the EMS manuscript submission.**

> *"Beyond RMSE: Decoupling Occurrence and Amount for Physically Consistent Precipitation Imputation"*  
> *Target journal: Environmental Modelling & Software*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

Standard deep-learning imputation models optimise for RMSE over all variables and time steps. Applied to precipitation, this objective produces **drizzle-like artefacts**: the model spreads precipitation mass across many days rather than concentrating it on true wet days. The result is realistic mean RMSE but severely distorted wet/dry intermittency and underestimated extremes.

**DLPIF** addresses this by introducing a two-stage physical post-processing layer on top of a continuous multivariate base imputation model:

1. **Stage 1 — Wet/Dry Occurrence Classification** — A Random Forest classifier predicts whether each missing day is wet or dry, using 25 meteorological, temporal, and spatial-context features. Local precipitation is **explicitly excluded** from Stage 1 inputs to prevent circular target leakage.

2. **Stage 2 — Wet-Day Amount Estimation** — A Random Forest regressor estimates precipitation amounts only for days predicted wet in Stage 1. Local precipitation is **hard-zeroed** in the feature matrix as a leakage guard.

Validation-derived calibration utilities are provided for precipitation-aware ablation variants and distributional adjustment where applicable.

---

## Key Results

Results are reported as **mean ± std** across three independent random seeds (42, 123, 456).

The numerical results reported in the manuscript are generated from `results/multiseed_clean_evaluation.csv` and summarized by `src/generate_clean_tables.py`.

> **DLPIF substantially reduces extreme-precipitation reconstruction error relative to continuous baselines.**

### Seed Robustness (Occurrence Model Thresholds)

| Seed | Val Cutoff | Val F1 | Val Precision | Val Recall |
|---|---|---|---|---|
| 42 | 0.600 | 0.786 | 0.781 | 0.791 |
| 123 | 0.620 | 0.785 | 0.789 | 0.781 |
| 456 | 0.580 | 0.785 | 0.777 | 0.793 |

Max seed std(F1) < 0.005 — results are **stable** across all seeds.

---

## Architecture

```
Incomplete meteorological data
         |
         v
   [WGAN-GP Base Imputation]   <-- Mode B: dual-branch spatio-temporal BiLSTM
   Temporal branch: BiLSTM([corrupted | combined_mask | temporal_features])
   Spatial branch:  FFN([neighbor_avg | neighbor_mask])
   Fusion:          Concatenate + Linear + Sigmoid
         |
         v  (raw GAN output — drizzle artefacts present)
         |
   [Stage 1: Occurrence RF]    <-- 25-feature Random Forest Classifier
   Feature space: local meteo (no PRECIP) + temporal + neighbor avg + neighbor mask
   Threshold: val-only F1 maximisation (grid 0.20-0.80, step 0.02)
         |
         v  (binary wet/dry mask)
         |
   [Stage 2: Amount RF]        <-- 26-feature Random Forest Regressor
   Applied only to wet-predicted positions
   Local PRECIP hard-zeroed to prevent leakage
   + quantile mapping to val wet-day distribution
         |
         v
   [Physical Consistency]
   Dry positions forced to 0.0 mm
   Negative values clipped
         |
         v
   Final physically-consistent precipitation reconstruction
```

---

## Repository Structure

```
DLPIF-clean-repo/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── 01_data_preprocessing.py       # Data loading, validation, adjacency, normalisation, missingness scenarios
│   ├── 02_wgan_gp_imputation.py       # WGAN-GP base imputation — Mode A (temporal) & Mode B (spatio-temporal)
│   ├── 03_baseline_imputation.py      # Mean, linear, KNN, MICE baselines
│   ├── 04_evaluation.py               # Full evaluation: RMSE/MAE, seasonal, station-wise, distribution metrics
│   │
│   ├── multiseed_clean_rerun.py       # ★ OFFICIAL DLPIF pipeline (leakage-safe, 3-seed)
│   ├── generate_clean_tables.py       # ★ Generates manuscript Tables 1–2
│   ├── precip_calibration.py          # PrecipFix baseline (threshold + quantile mapping on val)
│   │
│   ├── (see _quarantine/precip_occurrence.py) # Occurrence model class (legacy reference, quarantined)
│   ├── audit_numbers.py               # Numerical audit to verify reported metrics
│   │
│   ├── baselines/
│   │   ├── saits_data_adapter.py      # Data adapter for SAITS model
│   │   ├── train_saits_v2.py          # SAITS training (optional deep baseline)
│   │   ├── evaluate_saits.py          # SAITS evaluation
│   │   └── run_multiseed_saits_v2.py  # SAITS multi-seed run
│   │
│   └── figures/
│       ├── select_diagnostic_window.py #Diagnostic window selection for Figure 6
│       ├── generate_fig1_EMS.py       # Figure 1 — DLPIF workflow
│       ├── generate_fig2_EMS.py       # Figure 2 — wet-day frequency
│       ├── generate_fig3_EMS.py       # Figure 3 — wet-day F1 performance
│       ├── generate_fig4_EMS.py       # Figure 4 — bias vs F1 relationship
│       ├── generate_fig5_EMS.py       # Figure 5 — extreme-event errors
│       ├── generate_fig6_EMS.py       # Figure 6 — qualitative reconstruction window
│       ├── generate_fig7_EMS.py       # Figure 7 — near-zero distribution comparison
│       └── generate_graphical_abstract.py  # Graphical abstract (2000x800 px)
│
├── results/
│   ├── multiseed_clean_evaluation.csv      # ★ Official DLPIF metrics (source of Tables 1–2)
│   ├── clean_full_evaluation.csv           # All methods combined
│   ├── occurrence_clean_seed_summary.csv   # Per-seed val cutoff, F1, precision, recall, bias
│   └── baseline_precip_classification.csv  # Baseline reference metrics
│
├── figures/                           # Output figures (regenerate with generate_fig*.py)
├── data/                              # Place preprocessed *.npz and scaler.pkl here
└── _quarantine/                       # Legacy / exploratory scripts (not used for results)
```

---

## Setup

```bash
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

- **Region:** Kutahya, Turkey
- **Stations:** 4 meteorological stations (KUTAHYA, TAVSANLI, SIMAV, GEDIZ)
- **Period:** 1973–2023 (daily resolution)
- **Variables (7):** `TMIN`, `TMEAN`, `TMAX`, `RH_MEAN`, `P_MEAN`, `WIND_MEAN`, `PRECIP`

### Spatial Adjacency

Station proximity is encoded using a **kNN-Gaussian adjacency matrix** (k=2 nearest neighbours). Distance combines geographic (Haversine) and elevation separation with a 70/30 weighting, then applies a Gaussian kernel. This adjacency drives the spatial branch of the WGAN-GP generator and the neighbour-average features used by the occurrence and amount models.

### Data Splits (temporal, no shuffle)

| Split | Fraction | Role |
|---|---|---|
| Train | 70% | Model fitting, scaler fitting |
| Validation | 15% | Threshold selection, early stopping, quantile map |
| Test | 15% | Final evaluation only — no decisions made here |

Strict temporal ordering is enforced; an assertion verifies zero date overlap between splits.

### Missingness Scenarios

| Scenario | Type | Rate |
|---|---|---|
| `10pct` | Random uniform | 10% of observed values |
| `20pct` | Random uniform | 20% of observed values |
| `block7d` | Consecutive blocks (7 days) | 20% of observed values |
| `block30d` | Consecutive blocks (30 days) | 20% of observed values |

Block missingness scenarios are generated for the **test split only**. Random scenarios are generated for all splits.

### Required Data Files

The meteorological observations used in this study were obtained from the Turkish State Meteorological Service (MGM) under institutional permission and access restrictions. Therefore, the raw and processed meteorological datasets are not publicly redistributed in this repository.

Users who obtain the required data through MGM’s permission procedures can place the following files in `data/` or update the script paths accordingly:
```
preprocessed_train.npz
preprocessed_val.npz
preprocessed_test.npz
scaler.pkl
gan_imputed_test_modeB_seed42.npy       (from Step 2 — raw GAN output)
gan_imputed_test_modeB_seed123.npy      (from Step 2 — raw GAN output)
gan_imputed_test_modeB_seed456.npy      (from Step 2 — raw GAN output)
baseline_results.pkl                    (from Step 3)
```

> **Note:** `gan_imputed_test_modeB_seed*_msclean_precip2stage.npy` and
> `gan_imputed_test_modeB_seed*_msclean_amountrf.npy` are **produced** by
> `multiseed_clean_rerun.py` (Step 5); they are not external prerequisites.

---

## Reproduction Steps

### Step 1 — Data Preprocessing

```bash
python src/01_data_preprocessing.py
```

**What it does:**
- Loads `dataset_birlestirilmis.csv` (combined 4-station dataset)
- Validates physical bounds (TMIN <= TMEAN <= TMAX, RH in [0,100], P_MEAN in [850,1050])
- Analyses real missingness (gap statistics per variable and per station)
- Builds the kNN-Gaussian adjacency matrix from station coordinates (Haversine + elevation, k=2)
- Adds temporal features: `DOY_sin`, `DOY_cos`, `MON_sin`, `MON_cos`, `SEASON`
- Splits chronologically: 70% train / 15% val / 15% test
- Fits `MinMaxScaler` on training data only, transforms all splits
- Computes neighbour-average features (`neighbor_avg`, `neighbor_mask`) using kNN adjacency
- Generates artificial missingness masks for all scenarios (10%, 20%, block-7d, block-30d)

**Outputs:** `preprocessed_{train,val,test}.npz`, `scaler.pkl`, `adjacency.pkl`, `missingness_report.csv`

---

### Step 2 — WGAN-GP Base Imputation

```bash
python src/02_wgan_gp_imputation.py --seed 42  --mode B
python src/02_wgan_gp_imputation.py --seed 123 --mode B
python src/02_wgan_gp_imputation.py --seed 456 --mode B
```

**Architecture (Mode B — dual-branch spatio-temporal):**

| Component | Details |
|---|---|
| Temporal branch | Bidirectional LSTM, hidden=64, 2 layers, on `[corrupted \| combined_mask \| temporal]` |
| Spatial branch | 2-layer feedforward network on `[neighbor_avg \| neighbor_mask]` |
| Fusion | Concatenate + Linear(128→64) + ReLU + Dropout + Linear(64→7) + Sigmoid |
| Discriminator | Unidirectional LSTM + Linear head (Wasserstein, no sigmoid) |
| Training | 60 epochs, batch=128, n_critic=3, lambda_GP=10, lambda_recon=10, Adam lr=1e-4 |
| Early stopping | Patience=10 on validation RMSE |
| Inference | Overlapping sliding-window (step = seq_len // 2), outputs averaged |

After training, `PrecipCalibrator` is fitted on the validation imputation to produce the `_precipfix.npy` variant (PrecipFix baseline).

**Outputs per seed:**
```
gan_model_modeB_seed{s}.pt
training_history_modeB_seed{s}.csv
gan_imputed_test_modeB_seed{s}.npy         (raw GAN output)
gan_imputed_test_modeB_seed{s}_precipfix.npy  (PrecipFix baseline)
```

---

### Step 3 — Baseline Imputation Methods

```bash
python src/03_baseline_imputation.py
```

Fits and evaluates: **mean**, **linear interpolation**, **KNN** (k=5), and **MICE** on all missingness scenarios.

**Output:** `baseline_results.pkl`

---

### Step 4 — PrecipFix Baseline (standalone re-run)

```bash
python src/precip_calibration.py
```

Refits `PrecipCalibrator` independently using only validation data. Selects the wet-day threshold that minimises |bias| on validation, with F1 as a tiebreaker. Optionally applies quantile mapping from the validation wet-day distribution.

---

### Step 5 — DLPIF Pipeline (Official Manuscript Results)

```bash
python src/multiseed_clean_rerun.py
```

This is the **single authoritative command** for all DLPIF results. Running it reproduces every number reported in Tables 1–2 of the manuscript.

**What it does (per seed):**

**Occurrence model (Stage 1):**
- Builds a 25-feature matrix from training observations — local `PRECIP` excluded
- Trains `RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced')`
- Selects the classification threshold that maximises F1 on the validation set (grid 0.20–0.80, step 0.02)
- No frequency matching; no test-distribution-informed thresholding
- Saves model as `precip_occurrence_clean_seed{s}.pkl` + metadata as `.json`

**Amount model (Stage 2):**
- Builds a 26-feature matrix — local `PRECIP` hard-zeroed
- Trains `RandomForestRegressor(n_estimators=400, min_samples_leaf=2)` on training wet-day observations
- Applies quantile mapping from validation wet-day distribution

**Inference:**
- Predicts wet/dry on test set using the occurrence model
- Zeros out dry-predicted positions (`Precip2Stage` variant)
- Fills wet-predicted positions with AmountRF predictions (`DLPIF/AmountRF` variant)
- Evaluates over all four missingness scenarios

**Outputs:**
```
multiseed_clean_evaluation.csv              (all metrics per seed x scenario)
occurrence_clean_seed_summary.csv           (val cutoff, F1, P, R, bias per seed)
precip_occurrence_clean_seed{42,123,456}.pkl/.json
gan_imputed_test_modeB_seed{s}_msclean_precip2stage.npy
gan_imputed_test_modeB_seed{s}_msclean_amountrf.npy
```

---

### Step 6 — Generate Manuscript Tables and Supporting Result Summaries

```bash
python src/generate_clean_tables.py
```

| Table | Content |
|---|---|
| Table 1 | Benchmark comparison under representative 10% random and 30-day block missingness scenarios |
| Table 2 | Ablation analysis of precipitation-specific correction stages under the 10% random missingness scenario |

Additional full-scenario and full-metric summaries are written to `clean_full_evaluation.csv`.

### Step 7 — Full Evaluation (Optional)

```bash
python src/04_evaluation.py
```

Computes extended diagnostics: per-variable RMSE/MAE in original units, standardised RMSE, seasonal breakdown (DJF/MAM/JJA/SON), station-wise RMSE, Wasserstein and KS distribution distances, physical consistency checks (TMIN <= TMEAN <= TMAX violation rate), and wet-day classification metrics for all discovered `.npy` files.

**Outputs:** `evaluation_results_orig.csv`, `evaluation_stdrmse.csv`, `evaluation_extreme.csv`, `seasonal_rmse.csv`, `distribution_metrics.csv`, `station_rmse.csv`, `physical_check_v2.csv`, `fig_timeseries.png`, `fig_scatter.png`, `fig_rmse_bar.png`

---

### Step 8 — SAITS Baseline (Optional)

```bash
python src/baselines/train_saits_v2.py
python src/baselines/run_multiseed_saits_v2.py
python src/baselines/evaluate_saits.py
```

---

### Step 9 — Generate Figures

All figure scripts must be **run from the repository root** (the directory containing `src/` and `figures/`):

```bash
# Run from the repository root:
python src/figures/generate_fig1_EMS.py
python src/figures/generate_fig2_EMS.py
# ... and so on through generate_fig7_EMS.py
python src/figures/generate_graphical_abstract.py
```

`select_diagnostic_window.py` is a diagnostic helper that prints candidate windows for qualitative visualisation; it is not required to reproduce submitted figures.

The time-series window shown in Figure 6 is selected using a predefined diagnostic heuristic implemented in `src/figures/select_diagnostic_window.py`. This window is used only for qualitative visualization. All reported quantitative metrics and conclusions are computed over the full masked test set, not over the selected illustrative window.

---

## Methodological Notes

### Leakage Prevention

| Stage | Guard |
|---|---|
| Stage 1 | Local `PRECIP` removed from the corrupted-data block (column index `pidx`) before feature concatenation |
| Stage 2 | Local `PRECIP` **hard-zeroed** in the feature matrix (set to 0.0 in physical units) |
| Threshold selection | Uses **validation F1 only** — no information from the test set is used at any point |
| Scaler | Fitted on **training data only**; applied to validation and test without refitting |
| Quantile map | Built from **validation wet-day ground truth** |

### Stage 1 Feature Space (25 features — no local PRECIP)

| Block | Features | Dim | Notes |
|---|---|---|---|
| Local meteorological | TMIN, TMEAN, TMAX, RH_MEAN, P_MEAN, WIND_MEAN | 6 | MinMax [0,1]; PRECIP **excluded** |
| Temporal encoding | DOY_sin, DOY_cos, MON_sin, MON_cos, SEASON | 5 | Cyclic; SEASON in {0,1,2,3} |
| Neighbour average | All 7 vars from k=2 nearest stations | 7 | MinMax [0,1]; neighbour PRECIP **retained** |
| Neighbour mask | Data-availability indicator per neighbour variable | 7 | Binary {0, 1} |

### Stage 2 Feature Space (26 features)

Same block structure as Stage 1 but: (1) features are in **original physical units** (inverse-transformed), (2) local PRECIP is **hard-zeroed** to prevent amount-leakage. The 26th feature arises because the corrupted block retains all 7 variables (PRECIP forced to 0 rather than dropped).

### Wet-Day Threshold

WMO standard threshold of **0.1 mm** is used for ground-truth wet/dry classification. The occurrence model's decision threshold is tuned independently per seed on the validation set (grid search over 0.20–0.80 in steps of 0.02), maximising F1.

### Occurrence Model Hyperparameters

```python
RandomForestClassifier(
    n_estimators    = 300,
    min_samples_leaf= 5,
    class_weight    = 'balanced',
    random_state    = seed,
    n_jobs          = -1,
)
```

### Amount Model Hyperparameters

```python
RandomForestRegressor(
    n_estimators    = 400,
    min_samples_leaf= 2,
    random_state    = seed,
    n_jobs          = -1,
)
```

Only training observations where `gt_mm > 0.1` are used for fitting (wet-day subsetting).

### Extreme-Event Threshold

**p95 = 16.74 mm/day** — computed from the training-set precipitation distribution. All extreme metrics (MAE p95, RMSE p95) are evaluated at masked positions where ground-truth precipitation exceeds this threshold.

---

## Results Files

| File | Description |
|---|---|
| `results/multiseed_clean_evaluation.csv` | All metrics per seed × scenario (primary source for Tables 1–2) |
| `results/occurrence_clean_seed_summary.csv` | Val cutoff, F1, precision, recall, bias per seed |
| `results/clean_full_evaluation.csv` | All methods (baselines + DLPIF) aggregated |
| `results/baseline_precip_classification.csv` | Baseline reference precipitation metrics |

---

## Citation

```bibtex
@misc{bozkurt2026dlpif,
  title  = {Beyond RMSE: Decoupling Occurrence and Amount for Physically Consistent Precipitation Imputation},
  author = {Bozkurt, Yasemin and Serttaş, Soydan and Bakır, Çiğdem},
  note   = {Manuscript submitted to Environmental Modelling & Software},
  year   = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

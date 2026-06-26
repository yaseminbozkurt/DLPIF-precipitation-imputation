# DLPIF — Decoupled Latent–Physical Imputation Framework

> Repository for the manuscript:
> **"Beyond RMSE: Decoupling Occurrence and Amount for Physically Consistent Precipitation Imputation"**
> Submitted to *Journal of Hydrology*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

Standard multivariate imputation models are commonly optimised using aggregate error metrics such as RMSE. However, when they are applied to precipitation, they may produce **drizzle-like artefacts**, where small nonzero rainfall amounts persist during physically dry periods. This behaviour can inflate wet-day frequency, distort dry-spell characteristics, and underestimate extreme precipitation events.

To address this issue, **DLPIF** introduces a precipitation-specific correction framework that explicitly decouples wet/dry occurrence reconstruction from conditional wet-day amount estimation. The framework is applied as a two-stage post-processing layer on top of a continuous multivariate base imputer.

DLPIF consists of two main stages:

1. **Stage 1 — Occurrence Classification**
   A Random Forest classifier predicts the wet/dry state for each missing precipitation value using meteorological, temporal, and spatial-context features. Local precipitation is excluded from the Stage 1 feature matrix to prevent circular target leakage.

2. **Stage 2 — Wet-Day Amount Estimation**
   A Random Forest regressor estimates precipitation amounts only for positions predicted as wet in Stage 1. Local precipitation is hard-zeroed in the Stage 2 feature matrix as an additional leakage-prevention guard.

The final reconstruction enforces physical consistency by setting predicted dry positions to exactly 0.0 mm and clipping negative values.

---

## Key Results

All reported values are averaged across three independent seeds: **42, 123, and 456**.

| Scenario   |      Method |    Bias |    F1 | Wet RMSE | Extreme RMSE |
| ---------- | ----------: | ------: | ----: | -------: | -----------: |
| Random 10% |       DLPIF | −0.0009 | 0.742 |     5.12 |        16.73 |
| Random 10% | WGAN-GP raw |  +0.678 | 0.487 |     6.77 |        26.16 |
| Block 30d  |       DLPIF |  +0.012 | 0.759 |     5.61 |        15.88 |

Across the evaluated missingness scenarios, DLPIF provides a more physically consistent precipitation reconstruction by reducing wet-day frequency bias, improving occurrence classification, and preserving high-intensity precipitation behaviour more effectively than raw continuous imputation outputs.

---

## Architecture

```text
Incomplete meteorological observations
         |
         v
[WGAN-GP Base Imputation]
Mode B: dual-branch spatio-temporal BiLSTM
Temporal branch: BiLSTM([corrupted | combined_mask | temporal_features])
Spatial branch:  FFN([neighbor_avg | neighbor_mask])
Fusion:          Concatenate → Linear → Sigmoid
         |
         v
Raw continuous precipitation output
         |
         v
[Stage 1: Occurrence Classification]
Random Forest Classifier
Features: local meteorology excluding PRECIP + temporal + neighbour context
Threshold: validation-F1 maximisation
         |
         v
Binary wet/dry mask
         |
         v
[Stage 2: Wet-Day Amount Estimation]
Random Forest Regressor
Applied only to wet-predicted positions
Local PRECIP hard-zeroed to prevent leakage
Quantile mapping to validation wet-day distribution
         |
         v
[Physical Consistency Layer]
Dry positions forced to 0.0 mm
Negative values clipped
         |
         v
Physically consistent precipitation reconstruction
```

---

## Repository Structure

```text
DLPIF-precipitation-imputation/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── 01_data_preprocessing.py
│   ├── 02_wgan_gp_imputation.py
│   ├── 03_baseline_imputation.py
│   ├── 04_evaluation.py
│   ├── 05_generate_seed_outputs.py
│   ├── 06_audit_predictions.py
│   ├── 07_summary_table.py
│   ├── 08_drizzle_false_wet.py
│   ├── 09_rmse_f1_tradeoff.py
│   ├── 10_dry_spell_cdd.py
│   ├── 11_spatial_correlation.py
│   ├── multiseed_clean_rerun.py
│   ├── precip_calibration.py
│   └── baselines/
│       ├── saits_data_adapter.py
│       ├── train_saits_v2.py
│       ├── evaluate_saits.py
│       └── run_multiseed_saits_v2.py
│
├── figures/
│   ├── generate_fig1_architecture.py
│   └── figures/
│       ├── Figure_1_DLPIF_Workflow.png
│       ├── Figure_2_WetDayFrequency.png
│       ├── Figure_3_F1_Performance.png
│       ├── Figure_4_RMSE_F1_Tradeoff.png
│       ├── Figure_5_ExtremeEvents.png
│       ├── Figure_6_TimeSeries.png
│       └── Figure_7_DistributionComparison.png
│
└── results/
    ├── metrics_seed42.csv
    ├── metrics_seed123.csv
    ├── metrics_seed456.csv
    ├── summary_mean_std.csv
    ├── summary_mean_std_wide.md
    ├── analysis_drizzle_distribution.csv
    ├── analysis_false_wet_rate.csv
    ├── analysis_dry_spells.csv
    ├── analysis_dry_spells_summary.md
    ├── analysis_spatial_correlation.csv
    └── analysis_spatial_correlation_summary.md
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/yaseminbozkurt/DLPIF-precipitation-imputation.git
cd DLPIF-precipitation-imputation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dependencies

| Package      | Version |
| ------------ | ------: |
| Python       |   3.10+ |
| scikit-learn |   1.8.0 |
| numpy        |   2.4.4 |
| pandas       |   3.0.2 |
| scipy        |  1.17.1 |
| matplotlib   |  3.10.9 |
| PyTorch      |  >= 2.0 |
| joblib       |  >= 1.3 |

---

## Data

### Study Area

* **Region:** Kütahya, Turkey
* **Stations:** KÜTAHYA, TAVŞANLI, SİMAV, GEDİZ
* **Temporal coverage:** 1973–2023
* **Resolution:** Daily
* **Variables:**
  `TMIN`, `TMEAN`, `TMAX`, `RH_MEAN`, `P_MEAN`, `WIND_MEAN`, `PRECIP`

### Data Availability

The meteorological observations used in this study were obtained from the Turkish State Meteorological Service (MGM) under institutional permission. Due to data-use restrictions, the raw and processed meteorological datasets cannot be publicly redistributed by the authors.

Researchers who obtain the required meteorological observations through MGM's official permission procedures can reproduce the workflow by placing the required preprocessed files in the `src/` directory.

Expected local files:

```text
preprocessed_train.npz
preprocessed_val.npz
preprocessed_test.npz
scaler.pkl
adjacency.pkl
gan_imputed_test_modeB_seed42.npy
gan_imputed_test_modeB_seed123.npy
gan_imputed_test_modeB_seed456.npy
baseline_results.pkl
```

The repository provides the source code, evaluation scripts, result summaries, figure-generation scripts, and documentation required to reproduce the modelling and evaluation workflow, subject to access to the restricted MGM data.

---

## Data Splits

| Split      | Fraction | Role                                                      |
| ---------- | -------: | --------------------------------------------------------- |
| Train      |      70% | Model fitting and scaler fitting                          |
| Validation |      15% | Threshold selection, early stopping, and quantile mapping |
| Test       |      15% | Final evaluation                                          |

The data are split chronologically. No test-set information is used for threshold selection, scaler fitting, model selection, or calibration.

---

## Missingness Scenarios

| Scenario   | Type                                 | Rate |
| ---------- | ------------------------------------ | ---: |
| `10pct`    | Random uniform missingness           |  10% |
| `20pct`    | Random uniform missingness           |  20% |
| `block7d`  | Consecutive 7-day block missingness  |  20% |
| `block30d` | Consecutive 30-day block missingness |  20% |

---

## Reproduction Workflow

### Step 1 — Data Preprocessing

```bash
python src/01_data_preprocessing.py
```

This script loads the meteorological data, validates physical bounds, constructs the spatial adjacency matrix, adds temporal features, applies chronological splitting, fits the scaler on the training split, and generates artificial missingness masks.

Expected outputs:

```text
preprocessed_train.npz
preprocessed_val.npz
preprocessed_test.npz
scaler.pkl
adjacency.pkl
missingness_report.csv
```

---

### Step 2 — WGAN-GP Base Imputation

```bash
python src/02_wgan_gp_imputation.py --seed 42 --mode B
python src/02_wgan_gp_imputation.py --seed 123 --mode B
python src/02_wgan_gp_imputation.py --seed 456 --mode B
```

Mode B uses a dual-branch architecture combining temporal and spatial information.

Main training configuration:

| Parameter                  | Value |
| -------------------------- | ----: |
| Epochs                     |    60 |
| Batch size                 |   128 |
| Critic steps               |     3 |
| Gradient penalty           |    10 |
| Reconstruction loss weight |    10 |
| Optimizer                  |  Adam |
| Learning rate              |  1e-4 |
| Early stopping patience    |    10 |

Expected outputs:

```text
gan_model_modeB_seed42.pt
gan_model_modeB_seed123.pt
gan_model_modeB_seed456.pt
gan_imputed_test_modeB_seed42.npy
gan_imputed_test_modeB_seed123.npy
gan_imputed_test_modeB_seed456.npy
```

---

### Step 3 — Baseline Imputation

```bash
python src/03_baseline_imputation.py
```

This script evaluates conventional baseline methods, including mean imputation, linear interpolation, KNN, and MICE.

Expected output:

```text
baseline_results.pkl
```

---

### Step 4 — DLPIF Pipeline

```bash
python src/multiseed_clean_rerun.py
```

This is the main script for reproducing the DLPIF evaluation results across all seeds and missingness scenarios.

The script performs:

* Stage 1 wet/dry occurrence classification
* Validation-based threshold selection
* Stage 2 wet-day amount estimation
* Quantile mapping using validation wet-day distribution
* Final physical consistency correction
* Evaluation across all missingness scenarios

Expected outputs:

```text
results/metrics_seed42.csv
results/metrics_seed123.csv
results/metrics_seed456.csv
results/summary_mean_std.csv
results/summary_mean_std_wide.md
```

---

### Step 5 — Additional Analyses

```bash
python src/07_summary_table.py
python src/08_drizzle_false_wet.py
python src/09_rmse_f1_tradeoff.py
python src/10_dry_spell_cdd.py
python src/11_spatial_correlation.py
```

These scripts generate summary tables and additional diagnostic analyses, including drizzle artefacts, false wet rate, RMSE–F1 trade-off, dry-spell reconstruction, and spatial correlation preservation.

---

### Step 6 — Figure Generation

```bash
python figures/generate_fig1_architecture.py
```

This script generates the DLPIF workflow figure used to illustrate the proposed architecture.

---

## Methodological Notes

### Leakage Prevention

| Component                | Leakage-prevention guard                        |
| ------------------------ | ----------------------------------------------- |
| Stage 1 occurrence model | Local `PRECIP` excluded from the feature matrix |
| Stage 2 amount model     | Local `PRECIP` hard-zeroed                      |
| Threshold selection      | Performed using validation F1 only              |
| Scaler                   | Fitted on the training split only               |
| Quantile mapping         | Built from validation wet-day observations only |
| Final evaluation         | Conducted on the held-out test split only       |

---

### Wet-Day Threshold

A threshold of **0.1 mm/day** is used to define wet-day occurrence. The occurrence classifier's decision threshold is tuned on the validation set using an F1-maximisation grid search.

---

### Extreme-Event Threshold

The extreme-event threshold is defined as the **95th percentile** of the training-set precipitation distribution.

Extreme-event metrics are evaluated at masked positions where the ground-truth precipitation exceeds this threshold.

---

### Occurrence Model

```python
RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=seed,
    n_jobs=-1
)
```

---

### Amount Model

```python
RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=2,
    random_state=seed,
    n_jobs=-1
)
```

---

## Reproducibility Statement

This repository contains the code and supporting outputs required to reproduce the modelling workflow, evaluation summaries, and manuscript figures, provided that the user has authorised access to the underlying MGM meteorological observations.

The raw meteorological data are not included in this repository because they are subject to institutional data-use restrictions.

---

## Citation

```bibtex
@article{bozkurt2026dlpif,
  title   = {Beyond RMSE: Decoupling Occurrence and Amount for Physically Consistent Precipitation Imputation},
  author  = {Bozkurt, Yasemin and Serttaş, Soydan and Bakır, Çiğdem},
  journal = {Journal of Hydrology},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

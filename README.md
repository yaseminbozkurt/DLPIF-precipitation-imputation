# DLPIF — Decoupled Learning–Physical Imputation Framework

> *"Beyond RMSE: Decoupling Occurrence and Amount for Physically Consistent Precipitation Imputation"*  
> Submitted to *Journal of Hydrology*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

Standard multivariate imputation models optimise for RMSE across all variables and time steps. When applied to precipitation, this produces **drizzle-like artefacts**: low-intensity nonzero values persist during physically dry periods, inflating wet-day frequency while underestimating extreme events.

**DLPIF** addresses this mismatch through a two-stage post-processing layer on top of a continuous multivariate base imputer:

1. **Stage 1 — Occurrence Classification** — A Random Forest classifier predicts wet/dry state for each missing day using 25 meteorological, temporal, and spatial-context features. Local precipitation is **excluded** from Stage 1 inputs to prevent circular target leakage.

2. **Stage 2 — Wet-Day Amount Estimation** — A Random Forest regressor estimates precipitation amounts only at positions predicted wet in Stage 1. Local precipitation is **hard-zeroed** in the feature matrix as a leakage guard.

---

## Key Results

All metrics are reported as **mean ± std** across three independent seeds (42, 123, 456).

| Scenario | Method | Bias | F1 | Wet RMSE | Extreme RMSE |
|---|---|---|---|---|---|
| Random 10% | DLPIF | −0.0009 | 0.742 | 5.12 | 16.73 |
| Random 10% | WGAN-GP (raw) | +0.678 | 0.487 | 6.77 | 26.16 |
| Block 30d | DLPIF | +0.012 | 0.759 | 5.61 | 15.88 |

**DLPIF is the only method on the Pareto front** (low wet RMSE + high occurrence F1) across all four missingness scenarios.

---

## Architecture

```
Incomplete meteorological observations
         |
         v
   [WGAN-GP Base Imputation]   <-- Mode B: dual-branch spatio-temporal BiLSTM
   Temporal branch: BiLSTM([corrupted | combined_mask | temporal_features])
   Spatial branch:  FFN([neighbor_avg | neighbor_mask])
   Fusion:          Concatenate → Linear → Sigmoid
         |
         v  (raw continuous output — drizzle artefacts present)
         |
   [Stage 1: Occurrence RF]    <-- 25-feature Random Forest Classifier
   Features: local meteo (no PRECIP) + temporal + neighbour avg + neighbour mask
   Threshold: validation-F1 maximisation (grid 0.20–0.80, step 0.02)
         |
         v  (binary wet/dry mask)
         |
   [Stage 2: Amount RF]        <-- 26-feature Random Forest Regressor
   Applied only to wet-predicted positions
   Local PRECIP hard-zeroed to prevent leakage
   + quantile mapping to validation wet-day distribution
         |
         v
   [Physical Consistency]
   Dry positions forced to 0.0 mm; negatives clipped
         |
         v
   Physically consistent precipitation reconstruction
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
│   ├── 02_wgan_gp_imputation.py       # WGAN-GP base imputation (Mode B: spatio-temporal BiLSTM)
│   ├── 03_baseline_imputation.py      # Mean, linear interpolation, KNN, MICE baselines
│   ├── 04_evaluation.py               # Full evaluation: RMSE/MAE, seasonal, station-wise, distribution metrics
│   ├── 05_generate_seed_outputs.py    # Generates per-seed metrics and prediction CSV files
│   ├── 06_audit_predictions.py        # Numerical audit to verify reported metrics
│   ├── 07_summary_table.py            # Mean ± std summary tables (manuscript Tables 2–3)
│   ├── 08_drizzle_false_wet.py        # Drizzle distribution and false wet rate analysis
│   ├── 09_rmse_f1_tradeoff.py         # RMSE–F1 Pareto trade-off figure (Figure 4)
│   ├── 10_dry_spell_cdd.py            # Hydrological dry-spell and CDD analysis (Table 6)
│   ├── 11_spatial_correlation.py      # Inter-station spatial correlation preservation (Table 7)
│   ├── multiseed_clean_rerun.py       # DLPIF pipeline — reproduces all manuscript results
│   ├── precip_calibration.py          # PrecipFix baseline (threshold + quantile mapping on val)
│   └── baselines/
│       ├── saits_data_adapter.py
│       ├── train_saits_v2.py
│       ├── evaluate_saits.py
│       └── run_multiseed_saits_v2.py
│
├── figures/
│   ├── generate_fig1_architecture.py  # Figure 1 — DLPIF workflow diagram
│   └── figures/                       # Output figures (PNG, PDF, SVG)
│       ├── Figure_1_DLPIF_Workflow.png
│       ├── Figure_2_WetDayFrequency.png
│       ├── Figure_3_F1_Performance.png
│       ├── Figure_4_Bias_vs_F1.png
│       ├── Figure_5_ExtremeEvents.png
│       ├── Figure_6_TimeSeries.png
│       └── Figure_7_DistributionComparison.png
│
└── results/
    ├── clean_full_evaluation.csv           # All methods — aggregated metrics
    ├── multiseed_clean_evaluation.csv      # Per-seed metrics (source of manuscript tables)
    ├── occurrence_clean_seed_summary.csv   # Val threshold, F1, precision, recall per seed
    ├── summary_mean_std.csv                # Mean ± std across seeds
    ├── analysis_false_wet_rate.csv         # False wet rate and drizzle analysis
    ├── analysis_dry_spells.csv             # CDD and dry-spell length reconstruction
    └── analysis_spatial_correlation.csv    # Inter-station correlation MAE
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
- **Period:** 1973–2023 (daily resolution)
- **Variables (7):** `TMIN`, `TMEAN`, `TMAX`, `RH_MEAN`, `P_MEAN`, `WIND_MEAN`, `PRECIP`

### Data Availability

The meteorological observations used in this study were obtained from the Turkish State Meteorological Service (MGM) under institutional permission. Raw data cannot be publicly redistributed. Users who obtain the required data through MGM's permission procedures should place the following files in `src/`:

```
preprocessed_train.npz
preprocessed_val.npz
preprocessed_test.npz
scaler.pkl
adjacency.pkl
gan_imputed_test_modeB_seed{42,123,456}.npy   (Step 2 outputs)
baseline_results.pkl                           (Step 3 output)
```

### Data Splits

| Split | Fraction | Role |
|---|---|---|
| Train | 70% | Model fitting, scaler fitting |
| Validation | 15% | Threshold selection, early stopping, quantile mapping |
| Test | 15% | Final evaluation — no decisions made here |

Strict temporal ordering; zero date overlap verified by assertion.

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

### Step 2 — WGAN-GP Base Imputation

```bash
python src/02_wgan_gp_imputation.py --seed 42  --mode B
python src/02_wgan_gp_imputation.py --seed 123 --mode B
python src/02_wgan_gp_imputation.py --seed 456 --mode B
```

Mode B uses a dual-branch BiLSTM (temporal) + FFN (spatial). Training: 60 epochs, batch=128, n_critic=3, λ_GP=10, λ_recon=10, Adam lr=1e-4, early stopping patience=10.

**Outputs per seed:** `gan_model_modeB_seed{s}.pt`, `gan_imputed_test_modeB_seed{s}.npy`

### Step 3 — Baseline Methods

```bash
python src/03_baseline_imputation.py
```

Fits Mean, Linear Interpolation, KNN (k=5), and MICE on all scenarios. **Output:** `baseline_results.pkl`

### Step 4 — DLPIF Pipeline (Official Results)

```bash
python src/multiseed_clean_rerun.py
```

Reproduces all manuscript tables. Per seed: trains Stage 1 RF classifier (300 trees, balanced class weights, validation-tuned threshold), trains Stage 2 RF regressor (400 trees, wet-day subsetting), applies quantile mapping, evaluates over all scenarios.

**Outputs:** `results/multiseed_clean_evaluation.csv`, `results/occurrence_clean_seed_summary.csv`

### Step 5 — Analysis Scripts

```bash
python src/07_summary_table.py          # Mean ± std table
python src/08_drizzle_false_wet.py      # Drizzle / false wet analysis
python src/09_rmse_f1_tradeoff.py       # RMSE–F1 figure
python src/10_dry_spell_cdd.py          # Dry-spell / CDD analysis
python src/11_spatial_correlation.py    # Spatial correlation analysis
```

### Step 6 — Figure 1

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

**p95 = 16.74 mm/day**, computed from the training-set precipitation distribution. Extreme metrics (MAE p95, RMSE p95) are evaluated at masked positions where ground-truth precipitation exceeds this threshold.

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

MIT License — see [LICENSE](LICENSE).

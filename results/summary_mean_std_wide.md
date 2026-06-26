# Precipitation Imputation — Mean ± Std Summary Table

Metrics computed across seeds 42, 123, 456 (seeded methods) or single run
(deterministic methods: Mean, Linear, KNN, MICE).

| Symbol | Definition |
|---|---|
| Bias | freq_pred − freq_gt (wet-day frequency bias) |
| F1 | Harmonic mean of precision and recall (wet-day classification) |
| CSI | Critical Success Index = TP / (TP + FP + FN) |
| RMSE$_{wet}$ | RMSE on ground-truth wet-day positions (mm) |
| RMSE$_{p95}$ | RMSE on positions where ground truth ≥ 16.74 mm (extreme events, mm) |

*Values shown as mean ± std (std computed with ddof=1 across seeds).*
*Single values indicate deterministic methods or only one seed available.*

### Random 10%

| Method | Bias | F1 | CSI | RMSE$_{wet}$ | RMSE$_{p95}$ |
|---|---|---|---|---|---|
| Mean | 0.6780 | 0.4871 | 0.3220 | 5.94 | 23.43 |
| Linear | 0.0430 | 0.7311 | 0.5761 | 5.75 | 19.61 |
| KNN | 0.6753 | 0.4881 | 0.3228 | 6.95 | 23.65 |
| MICE | 0.6753 | 0.4881 | 0.3228 | 6.26 | 21.36 |
| WGAN-GP (raw) | 0.6780† | 0.4871† | 0.3220† | 6.77 ± 0.198 | 26.16 ± 0.475 |
| PrecipFix | 0.0045 ± 0.00324 | 0.3560 ± 0.01981 | 0.2167 ± 0.01457 | 8.86 ± 0.186 | 25.24 ± 0.148 |
| Precip2Stage | -0.0009† | 0.7420† | 0.5898† | 7.78 ± 0.124 | 23.38 ± 0.656 |
| DLPIF | -0.0009† | 0.7420† | 0.5898† | 5.12 ± 0.022 | 16.73 ± 0.045 |

### Random 20%

| Method | Bias | F1 | CSI | RMSE$_{wet}$ | RMSE$_{p95}$ |
|---|---|---|---|---|---|
| Mean | 0.6547 | 0.5133 | 0.3453 | 5.85 | 20.52 |
| Linear | 0.0480 | 0.7359 | 0.5821 | 5.50 | 15.60 |
| KNN | 0.6507 | 0.5135 | 0.3455 | 6.53 | 20.15 |
| MICE | 0.6543 | 0.5135 | 0.3454 | 6.40 | 18.55 |
| WGAN-GP (raw) | 0.6547† | 0.5133† | 0.3453† | 6.78 ± 0.223 | 23.46 ± 0.449 |
| PrecipFix | -0.0290 ± 0.01638 | 0.3973 ± 0.02474 | 0.2481 ± 0.01911 | 8.07 ± 0.145 | 23.10 ± 0.146 |
| Precip2Stage | -0.0054† | 0.7801† | 0.6395† | 9.02 ± 0.181 | 20.56 ± 0.149 |
| DLPIF | -0.0054† | 0.7801† | 0.6395† | 4.81 ± 0.021 | 13.49 ± 0.134 |

### Block 7d

| Method | Bias | F1 | CSI | RMSE$_{wet}$ | RMSE$_{p95}$ |
|---|---|---|---|---|---|
| Mean | 0.6831 | 0.4813 | 0.3169 | 5.96 | 20.94 |
| Linear | 0.0976 | 0.6579 | 0.4902 | 6.47 | 21.52 |
| KNN | 0.6778 | 0.4818 | 0.3173 | 6.49 | 18.98 |
| MICE | 0.6807 | 0.4822 | 0.3177 | 6.14 | 17.74 |
| WGAN-GP (raw) | 0.6831† | 0.4813† | 0.3169† | 6.84 ± 0.228 | 23.73 ± 0.597 |
| PrecipFix | 0.0015 ± 0.01439 | 0.3587 ± 0.02466 | 0.2187 ± 0.01810 | 8.82 ± 0.035 | 23.70 ± 0.603 |
| Precip2Stage | -0.0130† | 0.7704† | 0.6266† | 9.50 ± 0.146 | 20.20 ± 0.797 |
| DLPIF | -0.0130† | 0.7704† | 0.6266† | 4.60 ± 0.020 | 11.26 ± 0.065 |

### Block 30d

| Method | Bias | F1 | CSI | RMSE$_{wet}$ | RMSE$_{p95}$ |
|---|---|---|---|---|---|
| Mean | 0.6896 | 0.4737 | 0.3104 | 6.54 | 21.13 |
| Linear | 0.0485 | 0.4749 | 0.3113 | 7.71 | 23.03 |
| KNN | 0.6817 | 0.4759 | 0.3122 | 6.94 | 19.98 |
| MICE | 0.6856 | 0.4752 | 0.3116 | 6.80 | 19.28 |
| WGAN-GP (raw) | 0.6896† | 0.4737† | 0.3104† | 7.55 ± 0.221 | 24.11 ± 0.481 |
| PrecipFix | 0.0140 ± 0.00786 | 0.3424 ± 0.02603 | 0.2068 ± 0.01877 | 9.12 ± 0.149 | 24.31 ± 0.435 |
| Precip2Stage | 0.0119† | 0.7590† | 0.6116† | 9.52 ± 0.172 | 21.55 ± 0.592 |
| DLPIF | 0.0119† | 0.7590† | 0.6116† | 5.61 ± 0.023 | 15.88 ± 0.058 |

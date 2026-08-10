# Multi-Mask x Multi-Seed Extreme-Event (p95) Variance (Section 5.4/5.9 source)

10 independent random mask realizations per scenario x 3 model seeds (42, 123, 456) for AmountRF_DLPIF and SAITS -- 30 independent extreme-event evaluations pooled per method per scenario, versus 1 in the single-mask production tables (Table 2, Table S1). Linear is deterministic (10 realizations, no seed variation). Extreme threshold: PRECIP >= 19.2 mm (canonical_metrics.P95_THRESH).

| Method | Scenario | Mean RMSE_p95 | Std RMSE_p95 | Mean MAE_p95 | Std MAE_p95 | N (realizations x seeds) | Pooled n_extreme |
|---|---|---|---|---|---|---|---|
| AmountRF_DLPIF | 10pct | 18.77 | 4.35 | 16.68 | 4.12 | 27 | 144 |
| AmountRF_DLPIF | 20pct | 16.22 | 2.30 | 14.46 | 1.95 | 30 | 273 |
| Linear | 10pct | 26.15 | 7.98 | 23.84 | 7.31 | 9 | 48 |
| Linear | 20pct | 23.46 | 2.18 | 22.01 | 1.58 | 10 | 91 |
| SAITS | 10pct | 29.23 | 6.54 | 27.73 | 5.39 | 27 | 144 |
| SAITS | 20pct | 25.99 | 2.43 | 25.01 | 1.85 | 30 | 273 |

Paired Wilcoxon signed-rank test on RMSE_p95. The independent unit is the mask realization (n=10): for AmountRF_DLPIF vs SAITS, the 3 model seeds are averaged within each realization first (since seeds share the same masked positions and ground truth, they are repeated measures, not independent samples), then the test is run across the 10 resulting per-realization differences. Linear has no seed dimension and was already one value per realization.

| Comparison | Scenario | N (independent masks) | Mean diff (A - B) | p-value |
|---|---|---|---|---|
| AmountRF_DLPIF vs SAITS | 10pct | 9 | -10.467 | 3.906e-03 |
| AmountRF_DLPIF vs Linear | 10pct | 9 | -7.378 | 3.906e-03 |
| AmountRF_DLPIF vs SAITS | 20pct | 10 | -9.763 | 1.953e-03 |
| AmountRF_DLPIF vs Linear | 20pct | 10 | -7.241 | 1.953e-03 |

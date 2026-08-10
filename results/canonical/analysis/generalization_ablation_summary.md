# Cross-Scenario Generalisation Ablation (Table 6 source)

Model A: published pipeline, trained on 10% random missingness (seed 42). Model B: ablation, trained on 20pct random missingness (seed 42). Both evaluated on all four test scenarios using the identical canonical metric implementation (canonical_metrics.precip_metrics).

Maximum |ΔF1| across all four scenarios: **0.0202**

| Scenario | Model A F1 | Model B F1 | ΔF1 | Model A RMSE_p95 | Model B RMSE_p95 | ΔRMSE_p95 |
|---|---|---|---|---|---|---|
| 10pct | 0.8082 | 0.8079 | -0.0003 | 17.62 | 17.98 | +0.36 |
| 20pct | 0.7674 | 0.7876 | +0.0202 | 14.99 | 14.53 | -0.46 |
| block7d | 0.7875 | 0.7986 | +0.0111 | 15.45 | 16.37 | +0.92 |
| block30d | 0.8014 | 0.7979 | -0.0035 | 20.26 | 21.12 | +0.86 |

# Leave-One-Station-Out (LOSO) Spatial Cross-Validation Summary (Section 5.10 source)

Seed 42. Each fold trains Stage 1/Stage 2 on 3 stations (10% random-missingness partition) and evaluates purely out-of-sample on the 4th held-out station, across all four test scenarios. The scaler and neighbour-averaging adjacency are refit per fold, excluding the held-out station, so no information about it can leak into the other 3 stations' training features. Values below are averaged across the 4 station-folds.

| Scenario | Mean F1 | Mean Wet RMSE (mm) |
|---|---|---|
| 10pct | 0.8236 | 5.68 |
| 20pct | 0.7692 | 4.78 |
| block7d | 0.7914 | 4.94 |
| block30d | 0.7720 | 6.24 |

# Spatial Correlation Analysis Summary

Inter-station correlation matrix Mean Absolute Error (MAE) compared to observed values.

| Scenario | Method | Pearson Amount MAE | Spearman Amount MAE | Wet Occurrence MAE |
|---|---|---|---|---|
| 10pct | mean | 0.0733 | 0.1846 | 0.1466 |
| 10pct | linear | 0.0244 | 0.0235 | 0.0286 |
| 10pct | knn | 0.0463 | 0.1187 | 0.1456 |
| 10pct | mice | 0.0386 | 0.1234 | 0.1458 |
| 10pct | WGAN-GP_raw | 0.0525 | 0.1303 | 0.1466 |
| 10pct | SAITS | 0.0386 | 0.0327 | 0.0196 |
| 10pct | Precip2Stage | 0.0646 | 0.0177 | 0.0304 |
| 10pct | DLPIF | 0.0232 | 0.0253 | 0.0311 |
| 20pct | mean | 0.1523 | 0.3106 | 0.2546 |
| 20pct | linear | 0.0680 | 0.0402 | 0.0446 |
| 20pct | knn | 0.0910 | 0.2033 | 0.2535 |
| 20pct | mice | 0.0805 | 0.2091 | 0.2547 |
| 20pct | WGAN-GP_raw | 0.1220 | 0.2223 | 0.2546 |
| 20pct | SAITS | 0.0952 | 0.0812 | 0.0703 |
| 20pct | Precip2Stage | 0.1298 | 0.0285 | 0.0515 |
| 20pct | DLPIF | 0.0531 | 0.0250 | 0.0277 |
| block7d | mean | 0.0233 | 0.0399 | 0.0407 |
| block7d | linear | 0.0214 | 0.0520 | 0.0627 |
| block7d | knn | 0.0504 | 0.0290 | 0.0394 |
| block7d | mice | 0.0241 | 0.0342 | 0.0401 |
| block7d | WGAN-GP_raw | 0.0316 | 0.0327 | 0.0407 |
| block7d | SAITS | 0.0239 | 0.0103 | 0.0124 |
| block7d | Precip2Stage | 0.0337 | 0.0183 | 0.0352 |
| block7d | DLPIF | 0.0288 | 0.0319 | 0.0352 |
| block30d | mean | 0.0337 | 0.0899 | 0.0836 |
| block30d | linear | 0.0561 | 0.0590 | 0.0687 |
| block30d | knn | 0.0139 | 0.0641 | 0.0782 |
| block30d | mice | 0.0204 | 0.0752 | 0.0809 |
| block30d | WGAN-GP_raw | 0.0115 | 0.0710 | 0.0836 |
| block30d | SAITS | 0.0165 | 0.0279 | 0.0223 |
| block30d | Precip2Stage | 0.0165 | 0.0296 | 0.0394 |
| block30d | DLPIF | 0.0549 | 0.0390 | 0.0396 |
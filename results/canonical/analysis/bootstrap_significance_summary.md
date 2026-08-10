# Bootstrap Significance (recomputed, seed 42 primary realization)

Block-bootstrap (10000 resamples), contiguous same-station masked-run blocks as the resampling unit.

| Comparison | Metric | Scenario | Diff (B - A) | 95% CI | p-value | Significant (p<0.05) |
|---|---|---|---|---|---|---|
| AmountRF_DLPIF vs Linear | f1 | 10pct | +0.1338 | [+0.0718, +0.1981] | 0.0000 | True |
| AmountRF_DLPIF vs Linear | f1 | block30d | +0.3894 | [+0.2884, +0.5167] | 0.0000 | True |
| AmountRF_DLPIF vs SAITS | rmse_wet | 10pct | -1.8503 | [-2.7110, -1.0009] | 0.0000 | True |
| AmountRF_DLPIF vs SAITS | rmse_wet | block30d | -2.2623 | [-2.7812, -1.6430] | 0.0000 | True |
| AmountRF_DLPIF vs SAITS | rmse_p95 | 10pct | -8.1900 | [-12.5207, -5.7100] | 0.0030 | True |
| AmountRF_DLPIF vs SingleStageRF | f1 | 10pct | +0.1232 | [+0.0719, +0.1747] | 0.0000 | True |
| AmountRF_DLPIF vs SingleStageRF | f1 | block30d | +0.1684 | [+0.1139, +0.2294] | 0.0000 | True |
| MissingnessIndicatorRF vs DirectTwoStageRF | f1 | 10pct | -0.0013 | [-0.0147, +0.0107] | 0.9294 | False |
| MissingnessIndicatorRF vs DirectTwoStageRF | f1 | 20pct | -0.0045 | [-0.0189, +0.0098] | 0.5324 | False |
| MissingnessIndicatorRF vs DirectTwoStageRF | f1 | block7d | -0.0054 | [-0.0215, +0.0088] | 0.4950 | False |
| MissingnessIndicatorRF vs DirectTwoStageRF | f1 | block30d | -0.0014 | [-0.0097, +0.0076] | 0.7826 | False |
| Linear vs DirectTwoStageRF | f1 | netblock30d | +0.5202 | [+0.3680, +0.6129] | 0.0000 | True |
| SAITS vs DirectTwoStageRF | f1 | netblock30d | +0.4017 | [+0.3027, +0.4828] | 0.0000 | True |

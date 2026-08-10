# False-Wet-Rate Analysis (Table 4 source)

Computed from results/canonical/predictions/*.csv, mean across seeds. "Mean pred (true-dry)" is UNCONDITIONAL over all true-dry positions (diluted by correctly-predicted-zero positions); "Mean pred | false-wet" is CONDITIONAL on the prediction actually exceeding the 0.1 mm threshold -- the true residual amount on the misclassified subset.

| Scenario | Method | False Wet Rate | Mean pred (true-dry, unconditional) | Mean pred \| false-wet | p95 pred (true-dry) | share[0,0.1) | share[>5mm] |
|---|---|---|---|---|---|---|---|
| 10pct | Mean | 100.00% | 1.5873 | 1.5873 | 1.9319 | 0.0% | 0.0% |
| 10pct | Linear | 32.36% | 0.6853 | 2.1146 | 4.0800 | 67.6% | 3.3% |
| 10pct | KNN | 39.27% | 0.8712 | 2.2006 | 4.1560 | 60.7% | 4.0% |
| 10pct | MICE | 62.18% | 1.1219 | 1.7891 | 4.9568 | 37.8% | 5.1% |
| 10pct | SAITS | 21.58% | 0.1185 | 0.4682 | 0.6043 | 78.4% | 0.0% |
| 10pct | AmountRF_DLPIF | 13.33% | 0.3862 | 2.8925 | 3.0675 | 86.7% | 1.8% |
| 10pct | DirectTwoStageRF | 13.33% | 0.3862 | 2.8925 | 3.0675 | 86.7% | 1.8% |
| 10pct | MissingnessIndicatorRF | 13.70% | 0.4007 | 2.9291 | 3.1082 | 86.3% | 1.9% |
| 10pct | SingleStageRF | 45.21% | 0.4367 | 0.9280 | 2.1965 | 54.8% | 0.6% |
| 10pct | WGANGP_PrecipFix | 24.73% | 0.0858 | 0.3441 | 0.2731 | 75.3% | 0.0% |
| 10pct | WGANGP_raw | 99.88% | 1.0988 | 1.0996 | 2.0424 | 0.1% | 0.1% |
| 20pct | Mean | 100.00% | 1.5786 | 1.5786 | 1.9319 | 0.0% | 0.0% |
| 20pct | Linear | 33.40% | 0.7425 | 2.2205 | 4.4700 | 66.6% | 4.5% |
| 20pct | KNN | 33.02% | 0.7505 | 2.2483 | 4.5480 | 67.0% | 4.4% |
| 20pct | MICE | 61.67% | 0.8975 | 1.4393 | 4.4820 | 38.3% | 4.2% |
| 20pct | SAITS | 16.70% | 0.0925 | 0.4321 | 0.4817 | 83.3% | 0.0% |
| 20pct | AmountRF_DLPIF | 11.39% | 0.4102 | 3.6113 | 3.0541 | 88.6% | 2.5% |
| 20pct | DirectTwoStageRF | 11.39% | 0.4102 | 3.6113 | 3.0541 | 88.6% | 2.5% |
| 20pct | MissingnessIndicatorRF | 12.02% | 0.4236 | 3.5493 | 3.0318 | 88.0% | 2.5% |
| 20pct | SingleStageRF | 51.11% | 0.4912 | 0.9287 | 2.4149 | 48.9% | 1.3% |
| 20pct | WGANGP_PrecipFix | 22.14% | 0.1075 | 0.4886 | 0.5800 | 77.9% | 0.1% |
| 20pct | WGANGP_raw | 99.81% | 1.0823 | 1.0833 | 1.9993 | 0.2% | 0.2% |
| block7d | Mean | 100.00% | 1.5871 | 1.5871 | 1.9319 | 0.0% | 0.0% |
| block7d | Linear | 29.51% | 1.0932 | 3.7009 | 7.8000 | 70.5% | 7.6% |
| block7d | KNN | 36.04% | 0.7892 | 2.1717 | 4.6700 | 64.0% | 4.4% |
| block7d | MICE | 59.54% | 1.0366 | 1.7235 | 4.7940 | 40.5% | 4.2% |
| block7d | SAITS | 18.43% | 0.1042 | 0.4538 | 0.5912 | 81.6% | 0.0% |
| block7d | AmountRF_DLPIF | 10.31% | 0.3489 | 3.3950 | 2.8981 | 89.7% | 1.9% |
| block7d | DirectTwoStageRF | 10.31% | 0.3489 | 3.3950 | 2.8981 | 89.7% | 1.9% |
| block7d | MissingnessIndicatorRF | 10.60% | 0.3563 | 3.3681 | 2.9883 | 89.4% | 1.9% |
| block7d | SingleStageRF | 47.53% | 0.4286 | 0.8655 | 2.3157 | 52.5% | 1.1% |
| block7d | WGANGP_PrecipFix | 21.85% | 0.0927 | 0.4178 | 0.5333 | 78.1% | 0.0% |
| block7d | WGANGP_raw | 99.71% | 1.0533 | 1.0547 | 2.0145 | 0.3% | 0.0% |
| block30d | Mean | 100.00% | 1.5853 | 1.5853 | 1.9319 | 0.0% | 0.0% |
| block30d | Linear | 40.11% | 1.6260 | 4.0384 | 10.4919 | 59.9% | 14.5% |
| block30d | KNN | 33.75% | 0.6287 | 1.8441 | 3.8300 | 66.2% | 3.0% |
| block30d | MICE | 66.25% | 0.9455 | 1.4135 | 3.9990 | 33.8% | 3.2% |
| block30d | SAITS | 15.43% | 0.0878 | 0.4393 | 0.4578 | 84.6% | 0.0% |
| block30d | AmountRF_DLPIF | 12.96% | 0.5227 | 4.0444 | 4.0676 | 87.0% | 3.5% |
| block30d | DirectTwoStageRF | 12.96% | 0.5227 | 4.0444 | 4.0676 | 87.0% | 3.5% |
| block30d | MissingnessIndicatorRF | 13.25% | 0.5068 | 3.8404 | 4.0418 | 86.8% | 3.4% |
| block30d | SingleStageRF | 53.65% | 0.5828 | 1.0566 | 3.0119 | 46.4% | 2.0% |
| block30d | WGANGP_PrecipFix | 20.91% | 0.0886 | 0.4269 | 0.5305 | 79.1% | 0.0% |
| block30d | WGANGP_raw | 97.35% | 1.0721 | 1.0853 | 2.0158 | 2.6% | 0.1% |

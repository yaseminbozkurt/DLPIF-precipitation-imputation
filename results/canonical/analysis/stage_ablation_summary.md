# Occurrence-only vs. DLPIF Stage Ablation (Tables 3/5 source)

Both variants share the identical Stage-1 occurrence-RF wet/dry decision. "Occurrence-only" fills wet-predicted cells with that station's training-set wet-day climatological mean; "DLPIF" fills them with the Stage-2 amount-RF regression. Mean +/- std across seeds 42, 123, 456.

| Scenario | Occ-only RMSE_wet | DLPIF RMSE_wet | Occ-only Extreme RMSE | DLPIF Extreme RMSE |
|---|---|---|---|---|
| 10pct | 5.91 ± 0.019 | 5.40 ± 0.033 | 21.42 ± 0.000 | 17.51 ± 0.163 |
| 20pct | 5.30 ± 0.030 | 4.84 ± 0.022 | 19.46 ± 0.000 | 14.86 ± 0.135 |
| block7d | 5.94 ± 0.027 | 4.89 ± 0.018 | 22.30 ± 0.000 | 15.52 ± 0.204 |
| block30d | 6.79 ± 0.002 | 6.08 ± 0.018 | 24.20 ± 0.000 | 20.36 ± 0.089 |

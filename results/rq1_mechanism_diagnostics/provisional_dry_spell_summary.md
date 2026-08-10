# PROVISIONAL RQ1 Dry-Spell / CDD Diagnostic

**NOT part of the canonical, audited pipeline** (results/canonical/) --
WGAN-GP has no GPU-generated output for the new mechanism scenarios, so
WGANGP_raw/WGANGP_PrecipFix are excluded here. See module docstring.

Averages across 4 stations and all available seeds. A dry day is PRECIP <= 0.1mm.

| Scenario | Method | Obs CDD | Pred CDD | CDD Error | Obs Mean | Pred Mean | Mean Error | Obs p95 | Pred p95 | p95 Error |
|---|---|---|---|---|---|---|---|---|---|---|
| 10pct | Mean | 45.2 | 31.0 | 14.25 | 5.66 | 3.99 | 1.67 | 18.1 | 11.4 | 6.72 |
| 10pct | Linear | 45.2 | 45.0 | 0.75 | 5.66 | 5.98 | 0.32 | 18.1 | 19.9 | 1.79 |
| 10pct | KNN | 45.2 | 44.8 | 1.00 | 5.66 | 5.15 | 0.51 | 18.1 | 17.0 | 1.09 |
| 10pct | MICE | 45.2 | 38.8 | 6.50 | 5.66 | 4.70 | 0.96 | 18.1 | 15.8 | 2.26 |
| 10pct | SAITS | 45.2 | 44.5 | 0.75 | 5.66 | 5.43 | 0.29 | 18.1 | 17.5 | 0.60 |
| 10pct | SingleStageRF | 45.2 | 41.5 | 3.75 | 5.66 | 5.02 | 0.64 | 18.1 | 16.8 | 1.31 |
| 10pct | DirectTwoStageRF | 45.2 | 44.5 | 0.75 | 5.66 | 5.65 | 0.10 | 18.1 | 18.6 | 0.57 |
| 20pct | Mean | 45.2 | 19.0 | 26.25 | 5.66 | 3.05 | 2.61 | 18.1 | 8.1 | 9.97 |
| 20pct | Linear | 45.2 | 49.8 | 5.00 | 5.66 | 6.29 | 0.63 | 18.1 | 24.7 | 6.64 |
| 20pct | KNN | 45.2 | 38.0 | 7.75 | 5.66 | 4.80 | 0.86 | 18.1 | 17.3 | 3.59 |
| 20pct | MICE | 45.2 | 28.2 | 17.00 | 5.66 | 4.01 | 1.65 | 18.1 | 12.3 | 5.77 |
| 20pct | SAITS | 45.2 | 49.1 | 6.17 | 5.66 | 5.32 | 0.48 | 18.1 | 19.5 | 3.04 |
| 20pct | SingleStageRF | 45.2 | 40.1 | 5.17 | 5.66 | 4.42 | 1.24 | 18.1 | 13.9 | 4.18 |
| 20pct | DirectTwoStageRF | 45.2 | 45.5 | 0.25 | 5.66 | 5.73 | 0.10 | 18.1 | 21.5 | 3.44 |
| block7d | Mean | 45.2 | 30.8 | 14.50 | 5.66 | 4.73 | 0.93 | 18.1 | 15.2 | 2.86 |
| block7d | Linear | 45.2 | 59.2 | 14.00 | 5.66 | 6.82 | 1.16 | 18.1 | 26.3 | 8.21 |
| block7d | KNN | 45.2 | 37.8 | 7.50 | 5.66 | 5.05 | 0.61 | 18.1 | 17.1 | 0.97 |
| block7d | MICE | 45.2 | 34.0 | 11.25 | 5.66 | 4.76 | 0.90 | 18.1 | 16.0 | 2.09 |
| block7d | SAITS | 45.2 | 48.2 | 9.17 | 5.66 | 5.67 | 0.43 | 18.1 | 18.5 | 1.92 |
| block7d | SingleStageRF | 45.2 | 36.2 | 9.08 | 5.66 | 4.82 | 0.84 | 18.1 | 16.3 | 1.78 |
| block7d | DirectTwoStageRF | 45.2 | 50.0 | 4.75 | 5.66 | 5.75 | 0.25 | 18.1 | 18.6 | 0.83 |
| block30d | Mean | 45.2 | 44.2 | 1.00 | 5.66 | 5.51 | 0.39 | 18.1 | 18.7 | 0.61 |
| block30d | Linear | 45.2 | 57.8 | 12.50 | 5.66 | 6.93 | 1.27 | 18.1 | 30.2 | 12.11 |
| block30d | KNN | 45.2 | 44.2 | 1.00 | 5.66 | 5.11 | 0.55 | 18.1 | 17.6 | 0.46 |
| block30d | MICE | 45.2 | 44.2 | 1.00 | 5.66 | 5.09 | 0.57 | 18.1 | 18.0 | 0.10 |
| block30d | SAITS | 45.2 | 45.8 | 1.92 | 5.66 | 5.94 | 0.49 | 18.1 | 20.6 | 2.63 |
| block30d | SingleStageRF | 45.2 | 44.2 | 1.00 | 5.66 | 5.14 | 0.53 | 18.1 | 18.0 | 0.10 |
| block30d | DirectTwoStageRF | 45.2 | 45.8 | 0.50 | 5.66 | 5.69 | 0.21 | 18.1 | 18.7 | 0.67 |
| mar_meteo | Mean | 45.2 | 25.5 | 19.75 | 5.66 | 3.76 | 1.90 | 18.1 | 10.9 | 7.20 |
| mar_meteo | Linear | 45.2 | 45.2 | 0.00 | 5.66 | 5.91 | 0.25 | 18.1 | 19.4 | 1.29 |
| mar_meteo | KNN | 45.2 | 38.8 | 6.50 | 5.66 | 5.00 | 0.66 | 18.1 | 17.3 | 1.15 |
| mar_meteo | MICE | 45.2 | 33.2 | 12.00 | 5.66 | 4.52 | 1.14 | 18.1 | 15.2 | 2.90 |
| mar_meteo | SAITS | 45.2 | 42.9 | 2.33 | 5.66 | 5.29 | 0.41 | 18.1 | 17.4 | 1.16 |
| mar_meteo | SingleStageRF | 45.2 | 40.8 | 4.50 | 5.66 | 4.96 | 0.70 | 18.1 | 16.4 | 1.71 |
| mar_meteo | DirectTwoStageRF | 45.2 | 45.2 | 0.00 | 5.66 | 5.54 | 0.12 | 18.1 | 18.1 | 0.50 |
| mnar_wet | Mean | 45.2 | 29.2 | 16.00 | 5.66 | 4.27 | 1.39 | 18.1 | 13.0 | 5.04 |
| mnar_wet | Linear | 45.2 | 45.2 | 0.00 | 5.66 | 6.04 | 0.38 | 18.1 | 19.2 | 1.10 |
| mnar_wet | KNN | 45.2 | 42.5 | 2.75 | 5.66 | 5.16 | 0.50 | 18.1 | 17.6 | 0.65 |
| mnar_wet | MICE | 45.2 | 39.0 | 6.25 | 5.66 | 4.91 | 0.75 | 18.1 | 16.3 | 1.80 |
| mnar_wet | SAITS | 45.2 | 45.0 | 0.25 | 5.66 | 5.54 | 0.29 | 18.1 | 18.0 | 0.62 |
| mnar_wet | SingleStageRF | 45.2 | 42.0 | 3.25 | 5.66 | 5.29 | 0.37 | 18.1 | 18.0 | 0.45 |
| mnar_wet | DirectTwoStageRF | 45.2 | 45.2 | 0.00 | 5.66 | 5.66 | 0.11 | 18.1 | 18.2 | 0.33 |
| mnar_intensity_moderate | Mean | 45.2 | 31.5 | 13.75 | 5.66 | 4.36 | 1.30 | 18.1 | 12.7 | 5.38 |
| mnar_intensity_moderate | Linear | 45.2 | 48.5 | 3.25 | 5.66 | 6.05 | 0.39 | 18.1 | 20.4 | 2.34 |
| mnar_intensity_moderate | KNN | 45.2 | 41.2 | 4.00 | 5.66 | 5.20 | 0.46 | 18.1 | 17.4 | 0.71 |
| mnar_intensity_moderate | MICE | 45.2 | 41.0 | 4.25 | 5.66 | 4.93 | 0.73 | 18.1 | 16.6 | 1.42 |
| mnar_intensity_moderate | SAITS | 45.2 | 44.8 | 0.50 | 5.66 | 5.45 | 0.26 | 18.1 | 18.3 | 1.00 |
| mnar_intensity_moderate | SingleStageRF | 45.2 | 41.8 | 3.42 | 5.66 | 5.10 | 0.56 | 18.1 | 17.1 | 1.01 |
| mnar_intensity_moderate | DirectTwoStageRF | 45.2 | 44.5 | 0.75 | 5.66 | 5.57 | 0.10 | 18.1 | 18.1 | 0.52 |
| mnar_intensity_severe | Mean | 45.2 | 36.2 | 9.00 | 5.66 | 4.96 | 0.70 | 18.1 | 16.8 | 1.31 |
| mnar_intensity_severe | Linear | 45.2 | 45.5 | 0.25 | 5.66 | 6.32 | 0.66 | 18.1 | 19.8 | 1.74 |
| mnar_intensity_severe | KNN | 45.2 | 42.0 | 3.75 | 5.66 | 5.53 | 0.14 | 18.1 | 17.7 | 0.77 |
| mnar_intensity_severe | MICE | 45.2 | 41.2 | 4.50 | 5.66 | 5.30 | 0.36 | 18.1 | 17.6 | 1.06 |
| mnar_intensity_severe | SAITS | 45.2 | 45.3 | 0.58 | 5.66 | 5.74 | 0.18 | 18.1 | 18.9 | 1.42 |
| mnar_intensity_severe | SingleStageRF | 45.2 | 41.8 | 3.50 | 5.66 | 5.46 | 0.20 | 18.1 | 17.9 | 0.49 |
| mnar_intensity_severe | DirectTwoStageRF | 45.2 | 45.5 | 0.25 | 5.66 | 5.67 | 0.04 | 18.1 | 18.5 | 0.47 |

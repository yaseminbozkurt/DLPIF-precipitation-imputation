# Audit — Linear Interpolation Baseline (Cross-Station Leakage Fix)

**Date:** 2026-07-20
**Script affected:** `src/03_baseline_imputation.py` (`linear_interpolation()`)
**Status:** FIXED — station-wise interpolation implemented and re-run for all 4 test scenarios.

## 1. Finding

`preprocessed_{train,val,test}.npz` stores rows sorted **DATE-major, STATION_ID-minor**
(see `01_data_preprocessing.py::load_data()` → `sort_values(['DATE','STATION_ID'])`).
For a fixed date, consecutive rows belong to different stations
(`17155 → 17704 → 17748 → 17750`), then the array moves to the next date.

The original `linear_interpolation()` called `pandas.DataFrame.interpolate()` directly
on this flat array:

```python
df  = pd.DataFrame(test_corrupted, columns=meteo_vars)
out = df.interpolate(method='linear', limit_direction='both').values
```

Because of the row ordering, this does **not** interpolate a station's own
value across time. It interpolates across *different stations on the same
(or adjacent) date*. A missing value at (date *t*, station GEDIZ) was filled
using the nearest non-missing rows in flat order — typically SIMAV or a
neighbouring station on date *t*, or KÜTAHYA on date *t*±1 — rather than
GEDIZ's own value on date *t*-1 / *t*+1. The baseline labelled "Linear
Interpolation" was therefore effectively a **same-day cross-station
proxy**, not a temporal interpolator.

## 2. Root Cause

Row-order assumption mismatch: `linear_interpolation()` assumed
STATION-major/DATE-minor ordering (or single-station data), but the NPZ
layout produced by `01_data_preprocessing.py` is DATE-major/STATION-minor.
No other baseline (`mean_imputation`, `knn_imputation`, `mice_imputation`)
is order-sensitive in the same way, since they treat each row independently
of its neighbours — only `interpolate()` depends on row adjacency.

## 3. Fix

`linear_interpolation()` now accepts `station_ids` and interpolates each
station's subsequence independently, preserving that subsequence's original
(chronological) row order before writing results back to their original
positions:

```python
def linear_interpolation(train_data, test_corrupted, meteo_vars, station_ids):
    station_ids = np.asarray(station_ids)
    out = test_corrupted.copy()
    for sid in pd.unique(station_ids):
        idx  = np.where(station_ids == sid)[0]
        df_s = pd.DataFrame(test_corrupted[idx], columns=meteo_vars)
        out[idx] = df_s.interpolate(method='linear', limit_direction='both').values
    return np.clip(out, 0, 1).astype(np.float32)
```

Verified on synthetic data with two interleaved stations (values `0.0–0.4`
vs `0.9–0.5`, one gap per station): the fixed function reconstructs each
station's series exactly via its own linear trend, with zero cross-station
contamination.

`main()` now loads `station_ids = te_npz['station_ids']` once and passes it
to both call sites (10% primary block and the all-scenario loop).

## 4. Re-run

`python src/03_baseline_imputation.py` executed against
`preprocessed_{train,val,test}.npz` (from the `soneklemee` reproduction
snapshot). Outputs regenerated for **all 4 test scenarios**
(`10pct`, `20pct`, `block7d`, `block30d`):

- `src/baseline_results.pkl` (all methods × all scenarios)
- `src/baseline_rmse.csv`

Macro-average (7-variable) RMSE, 10% scenario: Linear 0.0454 (previously
best of the four baselines; still best after the fix — the bug only
affected the PRECIP-specific wet/dry structure, not the multivariate RMSE
materially).

## 5. Impact on PRECIP Wet-Day Metrics

Wet-day classification / RMSE metrics recomputed with the same
methodology as `generate_clean_tables.py::precip_cls()` /
`extreme_metrics()` (threshold 0.1 mm, p95 = 16.74 mm). Full numbers in
`results/linear_stationwise_fix_comparison.csv`.

| Scenario | Bias (before → after) | F1 (before → after) | RMSE$_{wet}$ mm (before → after) | RMSE$_{p95}$ mm (before → after) |
|---|---|---|---|---|
| Random 10% | 0.0430 → **0.1013** | 0.7311 → **0.6522** | 5.75 → **7.28** | 19.61 → **24.42** |
| Random 20% | 0.0480 → **0.1058** | 0.7359 → **0.6791** | 5.50 → **6.65** | 15.60 → **21.94** |
| Block 7d | 0.0976 → **0.1097** | 0.6579 → **0.5835** | 6.48 → **6.87** | 21.52 → **22.19** |
| Block 30d | 0.0485 → **0.1074** | 0.4749 → **0.4772** | 7.71 → **7.82** | 23.03 → **23.50** |

**Direction of the effect:** the corrected (temporally-honest) Linear
baseline is *worse* on every occurrence and magnitude metric than the
buggy cross-station version, in every scenario except Block 30d (where it
is essentially unchanged). This is the expected behaviour for a
purely-temporal linear interpolator applied to an intermittent,
bursty variable: it smears nonzero values into dry days sitting between
two wet days, inflating wet-day frequency bias and wet-day RMSE. The old
version's apparent strength came from borrowing same-day values from
spatially correlated neighbouring stations, not from genuine temporal
skill.

**Manuscript implication (for later edit, not applied yet):** the
Discussion/Conclusion claim that "linear interpolation remains a
competitive baseline under sparse random gaps (10%)" was based on the
buggy cross-station numbers (F1 = 0.731). With the fix, Linear's 10%
F1 drops to 0.652, well below DLPIF (0.742); the previously
borderline-significant DLPIF-vs-Linear F1 comparison at 10% (ΔF1 = +0.019,
p = 0.063) should be re-run with the corrected baseline and will very
likely become clearly significant. Block 30d numbers (the scenario used
in the Abstract's headline comparison, F1 0.759 vs 0.475) are unaffected.

## 6. Outputs of this audit

- `results/audit_linear_interpolation.md` (this file)
- `results/linear_stationwise_fix_comparison.csv` (machine-readable before/after)

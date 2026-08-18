# External Validation — Ohio (USA) GHCN-Daily Network

A first, deliberately scoped-down external check of RQ1 (mechanism-shift
stress) and RQ4a (graded local/neighbour context-loss ablation) on four
real, freely-available GHCN-Daily stations in central Ohio, USA — a
network never seen during model development and geographically/
climatically unrelated to the primary Kütahya (Türkiye) study. See
manuscript Sections 5.7, 6.7, 7.1, and 7.4 for full methodology and
results, and the docstring at the top of `run_ohio_external_validation.py`
for a complete, disclosed list of scope reductions relative to the
primary study (reduced 4-variable feature set, RQ1/RQ4a only, adapted
MAR-Meteo mechanism, reused MNAR-Intensity beta doses).

## Data

Raw daily station data (`USW*.csv`) downloaded 2026-08-18 from NOAA's
public GHCN-Daily distribution (`noaa-ghcn-pds` on AWS Open Data,
`https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{station_id}.csv`)
— public domain, no institutional permission required. Citation:
Menne, M. J., Durre, I., Vose, R. S., Gleason, B. E., & Houston, T. G.
(2012). An overview of the Global Historical Climatology Network-Daily
database. *Journal of Atmospheric and Oceanic Technology*, 29(7), 897–910.
https://doi.org/10.1175/JTECH-D-11-00103.1

| Station | Name | Lat | Lon | Elev (m) | Joint TMAX+TMIN+PRECIP completeness (2005–2023) |
|---|---|---|---|---|---|
| USW00004804 | Columbus OSU AP | 40.0783N | 83.0783W | 274.9 | 99.7% |
| USW00004855 | Marion Muni AP | 40.6158N | 83.0672W | 300.2 | 96.5% |
| USW00004858 | Newark Heath AP | 40.0264N | 82.4633W | 267.6 | 98.9% |
| USW00053844 | Lancaster Fairfield Co AP | 39.7572N | 82.6633W | 258.8 | 99.0% |

An initial candidate network — four Turkish Black Sea coastal GHCN
stations, chosen to test a different climate regime within Türkiye — was
independently verified and rejected before this one was selected: joint
TMAX/TMIN/PRECIP completeness there was only 20–61% over 2005–2023, far
below what a fair replication requires. That verification is not
reproduced in this directory (no station data for the rejected candidates
is included), only reported in the manuscript and this README for
provenance.

## Reproduction

```bash
python run_ohio_external_validation.py
```

Reads the four `USW*.csv` files in this directory, builds the station-day
panel, adjacency graph, and missingness scenarios, trains DLPIF's Stage 1/
Stage 2 (identical hyperparameters to the primary study), and writes:

```
results/ohio_data_quality.csv              per-station, per-variable completeness
results/ohio_rq1_mechanism_results.csv     F1/bias/RMSE_wet per scenario x seed
results/ohio_rq4a_graded_context_loss.csv  F1 per family x level x context-seed x model-seed
```

## Headline results (see manuscript Sections 6.7, 7.1, 7.4 for full discussion)

- **RQ1**: wet-day RMSE divergence under increasing MNAR-Intensity severity
  replicates monotonically (6.92 → 10.83 → 16.26 mm); occurrence F1 does
  not rise monotonically as it does on the primary network (0.897 → 0.906
  → 0.856), coinciding with sharply more negative bias at the most severe
  dose (-0.077 → -0.196).
- **RQ4a**: local-context loss remains nearly inert (replicates); joint
  local+neighbour loss causes complete collapse, F1 = 0.000 (replicates
  exactly); neighbour-loss *alone* is considerably more damaging here
  (F1 → 0.074 at zero neighbours) than on the primary network (F1 → 0.574)
  — plausibly a consequence of this network's reduced local feature set
  (no relative humidity) rather than a genuine climate effect.

This is reported in the manuscript as a **partial** replication throughout
— confirming the qualitative reliability warnings RQ1 and RQ4a raise, not
claiming an exact quantitative match, and explicit about which
discrepancies are plausibly artefacts of this check's own scope reductions
rather than genuine cross-network findings.

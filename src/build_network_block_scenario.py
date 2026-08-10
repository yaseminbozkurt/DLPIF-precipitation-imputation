"""
build_network_block_scenario.py
=================================
Adds a fifth test scenario, "netblock30d": network-wide, simultaneous,
multivariate 30-day block missingness. Every existing scenario (10pct,
20pct, block7d, block30d) masks each (station, variable) pair
INDEPENDENTLY -- a given date can be missing for one station's PRECIP
while every other station and every other variable stays fully observed
on that same date (Section 4.2.1: "Variable Independence... Date
Independence"). netblock30d instead selects contiguous 30-day calendar
blocks where ALL 4 stations AND ALL 7 meteorological variables are
masked simultaneously -- a total-blackout scenario (e.g. a month-long
regional data-relay outage) that eliminates BOTH the local temporal
context (station-own lagged values, already unavailable under
block30d) AND the cross-station neighbour context (neighbor_avg /
neighbor_mask, still available under every other scenario) at the same
time. This is the most severe stress test of DLPIF's reliance on
concurrent neighbouring-station signals.

Method
------
1. Load the existing preprocessed_test.npz (already containing 10pct,
   20pct, block7d, block30d) and adjacency.pkl -- read-only; nothing
   about the existing scenarios or splits is touched.
2. Reshape real_mask to (n_dates, S, V) (DATE-major, STATION-minor row
   order, matching every other array in this pipeline) and flag dates
   where ALL stations and ALL variables are naturally observed --
   exactly the "eligible for artificial masking" rule of Section 4.2.1,
   extended from a single (station, variable) cell to the full network
   x variable block for a given date.
3. Candidate 30-day windows are calendar-contiguous runs of such fully-
   eligible dates. Candidates are drawn without replacement (RNG seed
   42 + 30 + 1000, distinct from the existing block30d seed 42+30) and
   reserved with a 1-day buffer on each side, mirroring
   01_data_preprocessing.py::block_missingness()'s selection procedure,
   until a ~20% missing-day target (BLOCK_RATE, matching block7d/
   block30d's convention) is reached.
4. All 4 stations x all 7 variables are masked (art_mask=1,
   corrupted=NaN) at every selected block's rows.
5. neighbor_avg_netblock30d / neighbor_mask_netblock30d are recomputed
   from this new corrupted array via 01_data_preprocessing.compute_
   neighbor_avg() -- at every masked date, no station has ANY neighbour
   data available for ANY variable, so neighbor_mask is exactly 0 there
   by construction (not hand-set -- a direct consequence of the shared
   corrupted array every other scenario's neighbour features are also
   derived from).
6. The four new keys are merged into preprocessed_test.npz (train/val
   untouched -- Stage 1/Stage 2 are trained once on 10pct and applied
   without retraining to every test scenario, exactly as for block7d/
   block30d).

Output
------
  preprocessed_test.npz   -- updated in place (original backed up first
                              to preprocessed_test_pre_netblock30d.npz.bak)
  network_block_diagnostics.csv
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prep01", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_data_preprocessing.py"))
prep01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep01)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK_LEN = 30
BLOCK_RATE = 0.20
SEED = 42 + BLOCK_LEN + 1000
SCEN_KEY = 'netblock30d'


def select_network_blocks(real_mask, dates, station_ids, block_len, rate, seed):
    stations = sorted(np.unique(station_ids).tolist())
    S = len(stations)
    n_rows = real_mask.shape[0]
    n_dates = n_rows // S
    V = real_mask.shape[1]

    date_series = pd.Series(pd.to_datetime(dates)).dt.normalize()
    # Rows are DATE-major / STATION-minor (see compute_neighbor_avg docstring);
    # verify that assumption holds before reshaping.
    RM = real_mask.reshape(n_dates, S, V)
    ordered_dates = date_series.iloc[np.arange(0, n_rows, S)].reset_index(drop=True)
    assert len(ordered_dates) == n_dates

    full_obs = RM.all(axis=(1, 2))  # (n_dates,) bool

    # Build calendar-day index -> row-block index map (require actual
    # day-to-day calendar contiguity, not just row adjacency).
    day_to_idx = {pd.Timestamp(d): i for i, d in enumerate(ordered_dates)}

    candidates = []
    for i, day in enumerate(ordered_dates):
        day = pd.Timestamp(day)
        window_days = [day + pd.Timedelta(days=k) for k in range(block_len)]
        window_idx = [day_to_idx.get(d) for d in window_days]
        if any(idx is None for idx in window_idx):
            continue
        if not all(full_obs[idx] for idx in window_idx):
            continue
        candidates.append((day, window_idx, window_days))

    target_missing_dates = int(round(full_obs.sum() * rate))
    target_blocks = max(1, int(round(target_missing_dates / block_len)))

    rng = np.random.default_rng(seed)
    reserved = set()
    selected = []
    for cand_idx in rng.permutation(len(candidates)):
        day, window_idx, window_days = candidates[int(cand_idx)]
        if any(d in reserved for d in window_days):
            continue
        selected.append((day, window_idx, window_days))
        reserved.update(window_days)
        reserved.add(window_days[0] - pd.Timedelta(days=1))
        reserved.add(window_days[-1] + pd.Timedelta(days=1))
        if len(selected) >= target_blocks:
            break

    return selected, n_dates, S, V, stations


def main():
    print('=' * 70)
    print(f'  BUILD NETWORK-WIDE SIMULTANEOUS MULTIVARIATE BLOCK SCENARIO')
    print(f'  ({SCEN_KEY}: {BLOCK_LEN}-day blocks, target rate {BLOCK_RATE:.0%},'
          f' ALL stations x ALL variables masked simultaneously)')
    print('=' * 70)

    test_path = os.path.join(SRC_DIR, 'preprocessed_test.npz')
    backup_path = os.path.join(SRC_DIR, 'preprocessed_test_pre_netblock30d.npz.bak')
    if not os.path.exists(backup_path):
        shutil.copy2(test_path, backup_path)
        print(f'  Backed up original test npz -> {backup_path}')
    else:
        print(f'  Backup already exists, not overwriting: {backup_path}')

    te = dict(np.load(test_path, allow_pickle=True))
    real_mask = te['real_mask'].astype(np.float32)
    data = te['data'].astype(np.float32)
    dates = te['dates']
    station_ids = te['station_ids']
    meteo_vars = list(te['meteo_vars'])

    with open(os.path.join(SRC_DIR, 'adjacency.pkl'), 'rb') as f:
        import pickle
        adj = pickle.load(f)
    A_knn = adj['A_knn']
    adj_stations = adj['stations']

    selected, n_dates, S, V, stations = select_network_blocks(
        real_mask, dates, station_ids, BLOCK_LEN, BLOCK_RATE, SEED)
    assert stations == sorted(adj_stations), (stations, adj_stations)

    print(f'  n_dates={n_dates}  n_stations={S}  n_vars={V}')
    print(f'  Selected {len(selected)} network-wide {BLOCK_LEN}-day blocks:')
    for day, window_idx, window_days in selected:
        print(f'    {window_days[0].date()} -> {window_days[-1].date()}')

    # Apply mask: every selected block masks ALL stations x ALL variables.
    art_mask = np.zeros_like(real_mask, dtype=np.float32)
    corrupted = data.copy()
    masked_row_idx = []
    for day, window_idx, window_days in selected:
        for d_idx in window_idx:
            rows = np.arange(d_idx * S, d_idx * S + S)  # this date's S station-rows
            masked_row_idx.extend(rows.tolist())
            art_mask[rows, :] = 1.0
            corrupted[rows, :] = np.nan

    masked_row_idx = np.array(sorted(set(masked_row_idx)))
    n_hidden_cells = int(art_mask.sum())
    print(f'  Masked {len(masked_row_idx)} (date,station) rows x {V} variables '
          f'= {n_hidden_cells} cells total')
    print(f'  PRECIP cells masked: {int(art_mask[:, meteo_vars.index("PRECIP")].sum())}')

    # ── Diagnostics / assertions ────────────────────────────────────────────
    bad_natural = int(((art_mask > 0.5) & (real_mask < 0.5)).sum())
    assert bad_natural == 0, f'artificial mask covers {bad_natural} naturally-missing cells'

    # Simultaneity check: every masked date has ALL S stations x V vars masked.
    AM = art_mask.reshape(n_dates, S, V)
    masked_dates = np.where(AM.any(axis=(1, 2)))[0]
    for d_idx in masked_dates:
        assert AM[d_idx].all(), f'date index {d_idx} is only partially masked (not network-wide)'
    print(f'  [OK] Simultaneity verified: {len(masked_dates)} masked dates, '
          f'each with ALL {S} stations x {V} variables masked.')

    # Block-length check.
    date_series = pd.Series(pd.to_datetime(dates)).dt.normalize()
    ordered_dates = date_series.iloc[np.arange(0, len(dates), S)].reset_index(drop=True)
    run_lengths, run_len = [], 0
    prev_masked, prev_day = False, None
    for i in range(n_dates):
        is_masked = bool(AM[i].any())
        day = pd.Timestamp(ordered_dates.iloc[i])
        is_next_day = prev_day is not None and day == prev_day + pd.Timedelta(days=1)
        if is_masked and run_len and is_next_day:
            run_len += 1
        elif is_masked:
            if run_len:
                run_lengths.append(run_len)
            run_len = 1
        else:
            if run_len:
                run_lengths.append(run_len)
            run_len = 0
        prev_day = day
    if run_len:
        run_lengths.append(run_len)
    assert run_lengths and min(run_lengths) == max(run_lengths) == BLOCK_LEN, \
        f'unexpected block lengths: {sorted(set(run_lengths))}'
    print(f'  [OK] All {len(run_lengths)} blocks are exactly {BLOCK_LEN} days long.')

    # ── Recompute neighbour features from the new corrupted array ─────────
    print('  Recomputing neighbor_avg / neighbor_mask for the new scenario ...')
    nbr_avg, nbr_mask = prep01.compute_neighbor_avg(corrupted, station_ids, A_knn, stations)
    masked_nbr_mask = nbr_mask.reshape(n_dates, S, V)[masked_dates]
    assert masked_nbr_mask.max() == 0.0, \
        'neighbor_mask is nonzero at a network-wide masked date -- neighbour leakage'
    print('  [OK] neighbor_mask is exactly 0 at every network-wide masked date '
          '(no cross-station leakage during the blackout).')

    # ── Save ─────────────────────────────────────────────────────────────
    te[f'corrupted_{SCEN_KEY}'] = corrupted.astype(np.float32)
    te[f'art_mask_{SCEN_KEY}'] = art_mask.astype(np.float32)
    te[f'neighbor_avg_{SCEN_KEY}'] = nbr_avg.astype(np.float32)
    te[f'neighbor_mask_{SCEN_KEY}'] = nbr_mask.astype(np.float32)

    np.savez_compressed(test_path, **te)
    print(f'\n  -> Updated {test_path} with corrupted_{SCEN_KEY}, art_mask_{SCEN_KEY}, '
          f'neighbor_avg_{SCEN_KEY}, neighbor_mask_{SCEN_KEY}')

    diag_rows = []
    for station in stations:
        s_rows = np.where(station_ids == station)[0]
        for v_idx, v in enumerate(meteo_vars):
            observed_count = int(real_mask[s_rows, v_idx].sum())
            actual_missing = int(art_mask[s_rows, v_idx].sum())
            diag_rows.append(dict(
                scenario=SCEN_KEY, variable=v, station=station,
                observed_count=observed_count, actual_missing_count=actual_missing,
                missing_rate=(actual_missing / observed_count) if observed_count else 0.0,
                number_of_blocks=len(selected), block_length=BLOCK_LEN,
            ))
    diag_df = pd.DataFrame(diag_rows)
    diag_path = os.path.join(SRC_DIR, '..', 'results', 'network_block_diagnostics.csv')
    diag_df.to_csv(diag_path, index=False)
    print(f'  -> {diag_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()

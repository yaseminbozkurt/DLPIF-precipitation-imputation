"""
01_data_preprocessing.py
========================
Kutahya Region – 4 Stations – Q1 Full Preprocessing Pipeline
Stations: KUTAHYA, TAVSANLI, SIMAV, GEDIZ  (1973-2023, daily)

Outputs:
    preprocessed_train.npz
    preprocessed_val.npz
    preprocessed_test.npz
    scaler.pkl
    adjacency.pkl
    missingness_report.csv
"""
import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
import pickle
import os
import math
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(OUTPUT_DIR, 'dataset_birlestirilmis.csv')

# Variables to impute (7 variables including PRECIP)
METEO_VARS   = ['TMIN', 'TMEAN', 'TMAX', 'RH_MEAN', 'P_MEAN', 'WIND_MEAN', 'PRECIP']
MISS_RATES   = [0.10, 0.20]               # random missingness rates (MVP: 10, 20%)
BLOCK_LENS   = [7, 30]                    # block lengths (days) — test only (MVP)
BLOCK_RATE   = 0.20                        # fraction to remove in block scenario
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

TEMPORAL_FEATURES = ['DOY_SIN', 'DOY_COS', 'MON_SIN', 'MON_COS', 'SEASON']


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Parse
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("  KUTAHYA REGION – 4 STATIONS – Q1 PREPROCESSING PIPELINE")
    print("=" * 60)
    print("\n1. Loading data ...")
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig', low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DATE'])
    df = df.sort_values(['DATE', 'STATION_ID']).reset_index(drop=True)
    print(f"   Rows  : {len(df):,}")
    print(f"   Dates : {df['DATE'].min().date()} → {df['DATE'].max().date()}")
    print(f"   Stations: {sorted(df['STATION_ID'].unique().tolist())}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_data(df: pd.DataFrame):
    print("\n2. Data Validation ...")

    # 2.1 Duplicate check
    dup = df.duplicated(subset=['DATE', 'STATION_ID']).sum()
    print(f"   Duplicates (DATE+STATION_ID): {dup}")

    # 2.2 Time continuity per station
    print("\n   Time continuity per station:")
    for sid, grp in df.groupby('STATION_ID'):
        grp = grp.sort_values('DATE')
        full_idx = pd.date_range(grp['DATE'].min(), grp['DATE'].max(), freq='D')
        missing_days = len(full_idx) - len(grp)
        print(f"     {str(sid):12s}: {len(grp):6,} records | {missing_days:5,} missing days "
              f"({grp['DATE'].min().date()} → {grp['DATE'].max().date()})")

    # 2.3 Physical bounds check
    print("\n   Physical bounds violations:")
    violations = {}
    if all(v in df.columns for v in ['TMIN','TMEAN','TMAX']):
        vio = ((df['TMIN'] > df['TMEAN']) | (df['TMEAN'] > df['TMAX'])).sum()
        violations['TMIN<=TMEAN<=TMAX'] = vio
        print(f"     TMIN<=TMEAN<=TMAX violations : {vio}")
    if 'RH_MEAN' in df.columns:
        vio = ((df['RH_MEAN'] < 0) | (df['RH_MEAN'] > 100)).sum()
        violations['RH_MEAN [0,100]'] = vio
        print(f"     RH_MEAN out of [0,100]        : {vio}")
    if 'PRECIP' in df.columns:
        vio = (df['PRECIP'] < 0).sum()
        violations['PRECIP>=0'] = vio
        print(f"     PRECIP < 0                    : {vio}")
    if 'P_MEAN' in df.columns:
        vio = ((df['P_MEAN'] < 850) | (df['P_MEAN'] > 1050)).sum()
        violations['P_MEAN [850,1050]'] = vio
        print(f"     P_MEAN out of [850,1050]       : {vio}")

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missingness Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_missingness(df: pd.DataFrame) -> pd.DataFrame:
    print("\n3. Missingness Analysis ...")
    records = []
    avail_vars = [v for v in METEO_VARS if v in df.columns]

    for var in avail_vars:
        total  = len(df)
        n_miss = df[var].isna().sum()
        pct    = 100 * n_miss / total

        # Consecutive missing run analysis
        is_missing = df[var].isna().astype(int).values
        runs = []
        cur  = 0
        for v in is_missing:
            if v:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)

        max_run  = max(runs) if runs else 0
        mean_run = round(np.mean(runs), 2) if runs else 0
        n_runs   = len(runs)

        records.append({'Variable': var, 'Total': total, 'Missing': n_miss,
                        'Missing_%': round(pct, 4),
                        'N_gaps': n_runs, 'Max_consecutive': max_run,
                        'Mean_consecutive': mean_run})
        print(f"   {var:12s}: {n_miss:6,}/{total:6,} ({pct:.2f}%) | "
              f"gaps={n_runs}, max_consec={max_run}")

    # Per-station breakdown
    print("\n   Per-station missingness:")
    for sid, grp in df.groupby('STATION_ID'):
        miss_pcts = []
        for var in avail_vars:
            if var in grp.columns:
                p = 100 * grp[var].isna().sum() / len(grp)
                miss_pcts.append(f"{var}={p:.1f}%")
        print(f"     {str(sid):12s}: " + " | ".join(miss_pcts))

    report = pd.DataFrame(records)
    path   = os.path.join(OUTPUT_DIR, 'missingness_report.csv')
    report.to_csv(path, index=False)
    print(f"\n   Report saved → {path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 4. Adjacency Matrix (Haversine + Elevation)
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def build_adjacency(df: pd.DataFrame, k: int = 2):
    """Build adjacency matrix from station coordinates."""
    print("\n4. Building adjacency matrix ...")
    # Get unique station coordinates
    sta_coords = (df.groupby('STATION_ID')[['LAT','LON','ELEV']]
                    .mean()
                    .reset_index()
                    .sort_values('STATION_ID')
                    .reset_index(drop=True))
    print(f"   Station coordinates:")
    for _, row in sta_coords.iterrows():
        print(f"     {str(row['STATION_ID']):12s}: LAT={row['LAT']:.4f}  LON={row['LON']:.4f}  ELEV={row['ELEV']:.0f}m")

    stations = sta_coords['STATION_ID'].tolist()
    n = len(stations)

    # Compute geo distance matrix (km)
    geo_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                geo_dist[i, j] = haversine_km(
                    sta_coords.loc[i, 'LAT'], sta_coords.loc[i, 'LON'],
                    sta_coords.loc[j, 'LAT'], sta_coords.loc[j, 'LON']
                )

    # Elevation difference matrix (m)
    elev_diff = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                elev_diff[i, j] = abs(sta_coords.loc[i, 'ELEV'] - sta_coords.loc[j, 'ELEV'])

    # Normalise both matrices to [0,1]
    geo_norm  = geo_dist  / (geo_dist.max()  + 1e-8)
    elev_norm = elev_diff / (elev_diff.max() + 1e-8)

    # Combined distance: 0.7 * geo + 0.3 * elev_diff
    combined = 0.7 * geo_norm + 0.3 * elev_norm
    np.fill_diagonal(combined, 0)

    # Gaussian kernel adjacency: Aij = exp(-dist)
    sigma = combined[combined > 0].std() + 1e-8
    A_gauss = np.exp(-combined / sigma)
    np.fill_diagonal(A_gauss, 0)

    # kNN mask (k=2 nearest per station)
    knn_mask = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        row_dist = combined[i].copy()
        row_dist[i] = np.inf
        nn_idx = np.argsort(row_dist)[:k]
        knn_mask[i, nn_idx] = 1.0

    # Final adjacency = Gaussian * kNN mask (keep only k nearest)
    A_knn = A_gauss * knn_mask

    print(f"\n   Geo distance matrix (km):")
    for i, s in enumerate(stations):
        row_str = "  ".join(f"{geo_dist[i,j]:6.1f}" for j in range(n))
        print(f"     {str(s):10s}: {row_str}")

    print(f"\n   kNN-Gaussian adjacency (k={k}):")
    for i, s in enumerate(stations):
        row_str = "  ".join(f"{A_knn[i,j]:.4f}" for j in range(n))
        print(f"     {str(s):10s}: {row_str}")

    adj_data = {
        'stations'   : stations,
        'geo_dist'   : geo_dist,
        'elev_diff'  : elev_diff,
        'combined'   : combined,
        'A_gaussian' : A_gauss,
        'A_knn'      : A_knn,
        'k'          : k,
        'sta_coords' : sta_coords,
    }
    adj_path = os.path.join(OUTPUT_DIR, 'adjacency.pkl')
    with open(adj_path, 'wb') as f:
        pickle.dump(adj_data, f)
    print(f"\n   Adjacency saved → {adj_path}")
    return adj_data


# ─────────────────────────────────────────────────────────────────────────────
# 5. Temporal & Spatial Features
# ─────────────────────────────────────────────────────────────────────────────
def add_temporal_spatial_features(df: pd.DataFrame) -> tuple:
    print("\n5. Adding temporal & spatial features ...")
    df = df.copy()

    # Temporal
    df['DOY']     = df['DATE'].dt.dayofyear
    df['MONTH']   = df['DATE'].dt.month
    df['YEAR']    = df['DATE'].dt.year
    df['DOY_SIN'] = np.sin(2 * np.pi * df['DOY'] / 365.0)
    df['DOY_COS'] = np.cos(2 * np.pi * df['DOY'] / 365.0)
    df['MON_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12.0)
    df['MON_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12.0)
    season_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
    df['SEASON']  = df['MONTH'].map(season_map)

    # Spatial: LAT/LON/ELEV already in dataset; ensure float32
    for col in ['LAT', 'LON', 'ELEV']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # Station one-hot encoding
    stations = sorted(df['STATION_ID'].unique().tolist())
    for sid in stations:
        df[f'STA_{sid}'] = (df['STATION_ID'] == sid).astype(np.float32)

    print(f"   Temporal feats: DOY_SIN/COS, MON_SIN/COS, SEASON")
    print(f"   Spatial cols  : LAT, LON, ELEV")
    print(f"   Station one-hot: {[f'STA_{s}' for s in stations]}")
    return df, stations


# ─────────────────────────────────────────────────────────────────────────────
# 6. Temporal Split (by unique DATE – no shuffle)
# ─────────────────────────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    print("\n6. Temporal split (70 / 15 / 15) ...")
    unique_dates = np.sort(df['DATE'].unique())
    n            = len(unique_dates)
    n_train      = int(n * train_frac)
    n_val        = int(n * val_frac)

    tr_cut  = unique_dates[n_train - 1]
    val_cut = unique_dates[n_train + n_val - 1]

    train_df = df[df['DATE'] <= tr_cut].copy()
    val_df   = df[(df['DATE'] > tr_cut) & (df['DATE'] <= val_cut)].copy()
    test_df  = df[df['DATE'] > val_cut].copy()

    assert train_df['DATE'].max() < val_df['DATE'].min(),  "Leakage: train/val overlap!"
    assert val_df['DATE'].max()   < test_df['DATE'].min(), "Leakage: val/test overlap!"

    print(f"   Unique dates: {n:,}  (train={n_train} | val={n_val} | test={n-n_train-n_val})")
    print(f"   Train: {len(train_df):6,} rows  {train_df['DATE'].min().date()} → {train_df['DATE'].max().date()}")
    print(f"   Val  : {len(val_df):6,} rows  {val_df['DATE'].min().date()} → {val_df['DATE'].max().date()}")
    print(f"   Test : {len(test_df):6,} rows  {test_df['DATE'].min().date()} → {test_df['DATE'].max().date()}")
    print("   [OK] No date overlap — leakage check passed")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Normalisation (fit on TRAIN only)
# ─────────────────────────────────────────────────────────────────────────────
def normalize(train_df, val_df, test_df, meteo_vars):
    print("\n7. Normalisation (MinMaxScaler, fit=train) ...")
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_meteo  = train_df[meteo_vars].copy()
    train_medians = train_meteo.median()

    train_filled = train_meteo.fillna(train_medians)
    scaler.fit(train_filled)

    def transform_df(df_in):
        df_out = df_in.copy()
        raw    = df_in[meteo_vars].values.astype(float)
        filled = np.where(np.isnan(raw), train_medians.values[np.newaxis, :], raw)
        scaled = scaler.transform(filled)
        scaled[np.isnan(raw)] = np.nan
        for i, v in enumerate(meteo_vars):
            df_out[v + '_NORM'] = scaled[:, i]
        return df_out

    train_df = transform_df(train_df)
    val_df   = transform_df(val_df)
    test_df  = transform_df(test_df)

    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'meteo_vars': meteo_vars,
                     'medians': train_medians.to_dict()}, f)
    print(f"   Scaler saved → {scaler_path}")
    print("   NORM ranges (train non-NaN):")
    for v in meteo_vars:
        col = v + '_NORM'
        mn, mx = train_df[col].min(), train_df[col].max()
        print(f"     {v:12s}: [{mn:.4f}, {mx:.4f}]")
    return train_df, val_df, test_df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 8. Artificial Missingness
# ─────────────────────────────────────────────────────────────────────────────
def random_missingness(data: np.ndarray, real_mask: np.ndarray,
                       miss_rate: float, seed: int = 42):
    """Randomly hide observed values. Returns (corrupted, art_mask)."""
    rng = np.random.default_rng(seed)
    art_mask  = np.zeros_like(real_mask, dtype=np.float32)
    corrupted = data.copy()
    for col in range(data.shape[1]):
        obs_idx  = np.where(real_mask[:, col] == 1)[0]
        n_remove = int(len(obs_idx) * miss_rate)
        if n_remove == 0:
            continue
        remove = rng.choice(obs_idx, size=n_remove, replace=False)
        art_mask[remove, col]  = 1
        corrupted[remove, col] = np.nan
    return corrupted, art_mask


def block_missingness(data: np.ndarray, real_mask: np.ndarray,
                      block_len: int, miss_rate: float, seed: int = 42):
    """Introduce consecutive block gaps."""
    rng      = np.random.default_rng(seed)
    art_mask = np.zeros_like(real_mask, dtype=np.float32)
    corrupted= data.copy()
    N        = data.shape[0]

    for col in range(data.shape[1]):
        obs_idx  = np.where(real_mask[:, col] == 1)[0]
        n_total  = len(obs_idx)
        n_remove = int(n_total * miss_rate)
        if n_remove == 0:
            continue
        n_blocks = max(1, n_remove // block_len)
        max_st   = n_total - block_len
        if max_st <= 0:
            continue
        n_blocks = min(n_blocks, max_st // max(block_len, 1))
        if n_blocks == 0:
            continue
        starts = rng.choice(max_st, size=n_blocks, replace=False)
        for s in starts:
            blk = obs_idx[s: s + block_len]
            art_mask[blk, col]  = 1
            corrupted[blk, col] = np.nan
    return corrupted, art_mask


# ─────────────────────────────────────────────────────────────────────────────
# Array builder
# ─────────────────────────────────────────────────────────────────────────────
SPATIAL_RAW_COLS = ['LAT', 'LON', 'ELEV']

def prepare_arrays(df, meteo_vars, temporal_feats, station_cols):
    norm_cols   = [v + '_NORM' for v in meteo_vars]
    data        = df[norm_cols].values.astype(np.float32)
    real_mask   = (~np.isnan(data)).astype(np.float32)
    temporal    = df[temporal_feats].values.astype(np.float32)
    # Spatial: one-hot + LAT/LON/ELEV
    spatial_cols_all = station_cols + [c for c in SPATIAL_RAW_COLS if c in df.columns]
    spatial     = df[spatial_cols_all].values.astype(np.float32) if spatial_cols_all else np.zeros((len(df), 0), dtype=np.float32)
    dates       = df['DATE'].values
    station_ids = df['STATION_ID'].values
    return data, real_mask, temporal, spatial, dates, station_ids


# ───────────────────────────────────────────────────────────────────────────────
# Neighbor average (Mode B support)
# ───────────────────────────────────────────────────────────────────────────────
def compute_neighbor_avg(data_norm, station_ids_arr, A_knn, station_list):
    """
    Compute kNN-weighted neighbor average per row (Mode B feature).

    Parameters
    ----------
    data_norm      : (N, V)  normalised data with NaN for real missing
    station_ids_arr: (N,)    station ID per row  (sorted DATE, STATION_ID)
    A_knn          : (S, S)  adjacency matrix (kNN-Gaussian weights)
    station_list   : list[S] station IDs in the same order as A_knn rows/cols

    Returns
    -------
    neighbor_avg  : (N, V)  float32  (NaN where no neighbour available)
    neighbor_mask : (N, V)  float32  1 = neighbour data available, 0 = not
    """
    N, V    = data_norm.shape
    S       = len(station_list)
    n_dates = N // S          # assumes equal number of rows per station

    # Reshape to (n_dates, S, V) — order is DATE → STATION_ID ascending
    D = data_norm.reshape(n_dates, S, V)   # NaN preserved

    nav = np.zeros((n_dates, S, V), dtype=np.float64)  # weighted sum
    nw  = np.zeros((n_dates, S, V), dtype=np.float64)  # weight sum

    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            w = float(A_knn[i, j])
            if w <= 0.0:
                continue
            dj    = D[:, j, :]                       # (n_dates, V)
            valid = (~np.isnan(dj)).astype(np.float64)
            nav[:, i, :] += w * valid * np.nan_to_num(dj, nan=0.0)
            nw[:, i, :]  += w * valid

    mask         = nw > 0.0
    neighbor_avg  = np.where(mask, nav / np.where(mask, nw, 1.0), np.nan)
    neighbor_mask = mask.astype(np.float32)

    print(f"     neighbor_avg coverage: {100.0 * mask.mean():.1f}% of (date,station,var) cells have ≥1 neighbour")
    return neighbor_avg.reshape(N, V).astype(np.float32), \
           neighbor_mask.reshape(N, V).astype(np.float32)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Validate
    validate_data(df)

    # 3. Missingness analysis (on raw data)
    avail_vars = [v for v in METEO_VARS if v in df.columns]
    analyze_missingness(df)

    # 4. Adjacency matrix (requires LAT, LON, ELEV in data)
    if all(c in df.columns for c in ['LAT', 'LON', 'ELEV']):
        adj_data = build_adjacency(df, k=2)
    else:
        print("\n4. [SKIP] LAT/LON/ELEV not found in dataset — no adjacency matrix")
        adj_data = None

    # 5. Temporal & Spatial features
    df, stations = add_temporal_spatial_features(df)
    station_cols = [f'STA_{s}' for s in stations]

    # 6. Split
    train_df, val_df, test_df = temporal_split(df)

    # 7. Normalize
    meteo_vars = [v for v in METEO_VARS if v in df.columns]
    train_df, val_df, test_df, scaler = normalize(train_df, val_df, test_df, meteo_vars)

    # 8. Arrays & Missingness scenarios
    print("\n8. Building NumPy arrays, missingness scenarios & neighbour avg ...")
    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    # Station list (in sort order, matching adjacency matrix)
    station_list = sorted(df['STATION_ID'].unique().tolist())
    A_knn        = adj_data['A_knn'] if adj_data is not None else None

    for split_name, split_df in splits.items():
        data, real_mask, temporal, spatial, dates, station_ids = prepare_arrays(
            split_df, meteo_vars, TEMPORAL_FEATURES, station_cols
        )

        save_dict = {
            'data'        : data,
            'real_mask'   : real_mask,
            'temporal'    : temporal,
            'spatial'     : spatial,
            'dates'       : dates,
            'station_ids' : station_ids,
            'meteo_vars'  : np.array(meteo_vars),
        }

        # Neighbour average (Mode B) — uses real (uncorrupted) normalised data
        if A_knn is not None:
            print(f"   {split_name:5s} | computing neighbour_avg ...")
            nbr_avg, nbr_mask = compute_neighbor_avg(
                data, station_ids, A_knn, station_list
            )
            save_dict['neighbor_avg']  = nbr_avg
            save_dict['neighbor_mask'] = nbr_mask

        # Random missingness scenarios: 10%, 20%
        for rate in MISS_RATES:
            seed_r = RANDOM_SEED + int(rate * 1000)
            c, am  = random_missingness(data, real_mask, rate, seed=seed_r)
            key    = f'{int(rate*100):02d}pct'
            save_dict[f'corrupted_{key}'] = c
            save_dict[f'art_mask_{key}']  = am
            n_hidden = int(am.sum())
            print(f"   {split_name:5s} | random {rate*100:4.0f}% → {n_hidden:,} values hidden")

        # Block missingness — test split only (7d & 30d)
        if split_name == 'test':
            for blen in BLOCK_LENS:
                c, am = block_missingness(data, real_mask, blen, BLOCK_RATE,
                                          seed=RANDOM_SEED + blen)
                save_dict[f'corrupted_block{blen}d'] = c
                save_dict[f'art_mask_block{blen}d']  = am
                n_hidden = int(am.sum())
                print(f"   {split_name:5s} | block {blen:3d}d      → {n_hidden:,} values hidden")

        out_path = os.path.join(OUTPUT_DIR, f'preprocessed_{split_name}.npz')
        np.savez_compressed(out_path, **save_dict)
        print(f"   → Saved: {out_path}  (data={data.shape}, temporal={temporal.shape})")

    # 9. Summary stats (original scale)
    print("\n9. Train statistics (original scale):")
    for v in meteo_vars:
        ser = train_df[v].dropna()
        if len(ser) > 0:
            print(f"   {v:12s}: mean={ser.mean():.2f}  std={ser.std():.2f}  "
                  f"min={ser.min():.2f}  max={ser.max():.2f}")

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

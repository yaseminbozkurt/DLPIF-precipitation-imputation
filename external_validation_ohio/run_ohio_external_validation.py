# -*- coding: utf-8 -*-
"""
run_ohio_external_validation.py
==================================
Independent external-validation replication of RQ1 (mechanism-shifted
missingness) and RQ4a (graded local/neighbour context-loss ablation) on a
network of 4 real, freely-available GHCN-Daily (NOAA) stations in central
Ohio, USA -- a network never seen during, and geographically/climatically
unrelated to, the primary Kutahya (Turkiye) study. Downloaded and verified
2026-08-18; no synthetic or fabricated values anywhere in this script.

WHY THIS EXISTS
---------------
The primary manuscript's own Limitations (Section 7.6, "Regional scope")
state the reliability findings are characterised within a single,
sparse-adjacency regional network and explicitly do not claim transfer to
a different network. This script is a first, disclosed, deliberately
scoped-down attempt at exactly that external check, using data that
requires no institutional permission (unlike the primary MGM-sourced
dataset).

STATIONS (verified real-data completeness before selection; a first
candidate network of Turkish Black Sea coastal GHCN stations was rejected
for this purpose after verification showed only 20-61% joint
TMAX+TMIN+PRCP completeness in 2005-2023 -- far below what a fair
replication requires; see conversation record):

  USW00004804  Columbus OSU AP        40.0783N  83.0783W  274.9m
  USW00004855  Marion Muni AP         40.6158N  83.0672W  300.2m
  USW00004858  Newark Heath AP        40.0264N  82.4633W  267.6m
  USW00053844  Lancaster Fairfield AP 39.7572N  82.6633W  258.8m

All four are automated ASOS airport stations with continuous reporting
since the late 1990s and 96.5-99.7% joint TMAX+TMIN+PRCP completeness over
2005-2023 -- comparable to the primary dataset's ~97-99% post-2005
completeness, and the same 19-year window (2005-01-01 to 2023-12-31),
70/15/15 chronological split.

DELIBERATE SCOPE REDUCTIONS (disclosed, not silent)
----------------------------------------------------
1. Variables: TMIN, TMAX, TMEAN, PRECIP only (4, vs. the primary study's
   7). GHCN-Daily does not reliably report RH_MEAN/P_MEAN/WIND_MEAN for
   this or almost any non-US network at daily resolution; humidity and
   pressure are absent from this station set entirely, and wind (AWND) is
   present but excluded to keep the variable set identical to what a
   typical non-US GHCN network would offer -- the point of this check is
   transfer to realistically-available open data, not cherry-picking the
   best-instrumented network available.
2. Research questions: RQ1 (mechanism-shift stress: MCAR-10, MNAR-Wet,
   MNAR-Intensity-Moderate/Severe, and an ADAPTED MAR-Meteo, see below)
   and RQ4a (graded local/neighbour context-loss ablation) only. RQ2
   (calibration), RQ3 (conformal), RQ4b (gate selection/application), the
   WGAN-GP/SAITS backbones, and block-missingness scenarios are NOT
   replicated here -- each would require substantially more engineering
   (a second base imputer, a second transformer training run, a second
   validation-only gate-selection procedure) beyond what a first
   external check should attempt in one pass.
3. MAR-Meteo mechanism ADAPTED: the primary study's formula weights by
   RH/WIND/PRESS, none of which exist here. This script substitutes the
   one meteorological covariate that IS available -- TMAX -- keeping the
   same functional form (logistic weighting, beta=2.0) and the same
   qualitative meaning (missingness depends on an OBSERVED COVARIATE,
   not on PRECIP itself), but this is a genuine adaptation, not a literal
   replication, and is reported as such.
4. MNAR-Intensity beta doses (Moderate=0.5, Severe=1.3) are REUSED from
   the primary study's validation-selected values rather than re-derived
   via a fresh grid search on this network's own validation partition
   (which the primary study's Section 5.1.3 procedure would require) --
   a disclosed time-boxing simplification. Realised test-set masking
   rates are reported descriptively, exactly as the primary study already
   does for its own frozen doses.
5. p95 extreme threshold IS freshly computed from this network's own
   train+val wet-day distribution (Section 4.7's convention) -- reusing
   the primary study's 19.20mm would be meaningless for a different
   climate.
6. Stage 1/Stage 2 Random Forest hyperparameters, wet-day threshold
   (0.1mm), chronological split fractions, MinMax scaling protocol,
   adjacency construction (kNN-Gaussian, k=2, 0.7*geo+0.3*elev, Haversine),
   and temporal features (DOY/MONTH sin-cos + season) are all IDENTICAL
   to the primary study (01_data_preprocessing.py, direct_two_stage_rf.py)
   -- only the inputs differ, not the method.

Outputs -> external_validation_ohio/results/
  ohio_data_quality.csv
  ohio_rq1_mechanism_results.csv
  ohio_rq4a_graded_context_loss.csv
  ohio_external_validation_summary.md
"""
import os
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

STATIONS = {
    'USW00004804': dict(name='Columbus OSU AP',        lat=40.0783, lon=-83.0783, elev=274.9),
    'USW00004855': dict(name='Marion Muni AP',          lat=40.6158, lon=-83.0672, elev=300.2),
    'USW00004858': dict(name='Newark Heath AP',         lat=40.0264, lon=-82.4633, elev=267.6),
    'USW00053844': dict(name='Lancaster Fairfield AP',  lat=39.7572, lon=-82.6633, elev=258.8),
}
STUDY_START, STUDY_END = '2005-01-01', '2023-12-31'
WET_THRESH = 0.1
SEEDS = [42, 123, 456]
CONTEXT_SEEDS = [101, 202, 303]
VARS = ['TMIN', 'TMAX', 'TMEAN', 'PRECIP']  # PRECIP always last -> pidx fixed below
PIDX = VARS.index('PRECIP')


# ─────────────────────────────────────────────────────────────────────────
# 1. Load + build the station-day panel
# ─────────────────────────────────────────────────────────────────────────

def load_station(sid):
    df = pd.read_csv(os.path.join(HERE, f'{sid}.csv'), dtype={'DATE': str}, low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    piv = df.pivot_table(index='DATE', columns='ELEMENT', values='DATA_VALUE', aggfunc='first')
    full_idx = pd.date_range(STUDY_START, STUDY_END, freq='D')
    piv = piv.reindex(full_idx)
    out = pd.DataFrame(index=full_idx)
    out['TMAX'] = piv.get('TMAX', np.nan) / 10.0       # tenths C -> C
    out['TMIN'] = piv.get('TMIN', np.nan) / 10.0
    tavg = piv.get('TAVG', pd.Series(np.nan, index=full_idx)) / 10.0
    computed = (out['TMAX'] + out['TMIN']) / 2.0
    out['TMEAN'] = tavg.where(tavg.notna(), computed)
    out['PRECIP'] = (piv.get('PRCP', np.nan) / 10.0).clip(lower=0)  # tenths mm -> mm
    out['STATION_ID'] = sid
    out['DATE'] = out.index
    return out.reset_index(drop=True)


def build_panel():
    frames = [load_station(sid) for sid in STATIONS]
    df = pd.concat(frames, ignore_index=True).sort_values(['DATE', 'STATION_ID']).reset_index(drop=True)
    for sid, meta in STATIONS.items():
        m = df['STATION_ID'] == sid
        df.loc[m, 'LAT'] = meta['lat']; df.loc[m, 'LON'] = meta['lon']; df.loc[m, 'ELEV'] = meta['elev']
    return df


# ─────────────────────────────────────────────────────────────────────────
# 2. Adjacency (identical construction to 01_data_preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_adjacency(station_ids, k=2):
    n = len(station_ids)
    coords = {sid: STATIONS[sid] for sid in station_ids}
    geo = np.zeros((n, n)); elev = np.zeros((n, n))
    for i, si in enumerate(station_ids):
        for j, sj in enumerate(station_ids):
            if i == j:
                continue
            geo[i, j] = haversine_km(coords[si]['lat'], coords[si]['lon'], coords[sj]['lat'], coords[sj]['lon'])
            elev[i, j] = abs(coords[si]['elev'] - coords[sj]['elev'])
    geo_n = geo / (geo.max() + 1e-8); elev_n = elev / (elev.max() + 1e-8)
    combined = 0.7 * geo_n + 0.3 * elev_n
    np.fill_diagonal(combined, 0)
    sigma = combined[combined > 0].std() + 1e-8
    A_gauss = np.exp(-combined / sigma); np.fill_diagonal(A_gauss, 0)
    knn_mask = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        row = combined[i].copy(); row[i] = np.inf
        nn_idx = np.argsort(row)[:k]
        knn_mask[i, nn_idx] = 1.0
    A_knn = A_gauss * knn_mask
    print('  Geo distance matrix (km):')
    for i, s in enumerate(station_ids):
        print('    ' + f'{s:14s}' + '  '.join(f'{geo[i, j]:6.1f}' for j in range(n)))
    print('  kNN-Gaussian adjacency (k=%d):' % k)
    for i, s in enumerate(station_ids):
        print('    ' + f'{s:14s}' + '  '.join(f'{A_knn[i, j]:.4f}' for j in range(n)))
    return A_knn


def neighbor_features(df_wide, A_knn, station_ids, var_cols):
    """df_wide: rows=DATE, cols=(station, var). Returns neighbor_avg, neighbor_mask
    arrays aligned to the long-format (station,date) row order used elsewhere."""
    n_sta = len(station_ids)
    dates = df_wide.index
    navg = {}
    nmask = {}
    n_var = len(var_cols)
    n_rows = len(df_wide.index)
    for i, sid in enumerate(station_ids):
        neighbors = [j for j in range(n_sta) if A_knn[i, j] > 0]
        if not neighbors:
            navg[sid] = np.zeros((n_rows, n_var), dtype=np.float32)
            nmask[sid] = np.zeros((n_rows, n_var), dtype=np.float32)
            continue
        w = np.array([A_knn[i, j] for j in neighbors])
        w = w / w.sum() if w.sum() > 0 else w
        avg_cols = []
        mask_cols = []
        for v in var_cols:
            vals = np.stack([df_wide[(v, station_ids[j])].to_numpy() for j in neighbors], axis=1)
            obs = ~np.isnan(vals)
            vals_filled = np.nan_to_num(vals, nan=0.0)
            denom = (obs * w).sum(axis=1)
            num = (vals_filled * obs * w).sum(axis=1)
            avg = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
            avg_cols.append(avg)
            mask_cols.append((obs.sum(axis=1) > 0).astype(np.float32))
        navg[sid] = np.stack(avg_cols, axis=1)
        nmask[sid] = np.stack(mask_cols, axis=1)
    return navg, nmask


# ─────────────────────────────────────────────────────────────────────────
# 3. Missingness scenarios
# ─────────────────────────────────────────────────────────────────────────

def random_missingness(data, real_mask, miss_rate, seed):
    rng = np.random.default_rng(seed)
    art_mask = np.zeros_like(real_mask, dtype=np.float32)
    corrupted = data.copy()
    for col in range(data.shape[1]):
        obs_idx = np.where(real_mask[:, col] == 1)[0]
        n_remove = int(len(obs_idx) * miss_rate)
        if n_remove == 0:
            continue
        remove = rng.choice(obs_idx, size=n_remove, replace=False)
        art_mask[remove, col] = 1
        corrupted[remove, col] = np.nan
    return corrupted, art_mask


def weighted_precip_mask(precip_gt, real_mask_precip, base_art_mask_precip, weights, seed):
    """Draw the SAME cell budget as base_art_mask_precip, from the same
    naturally-observed pool, but with probability proportional to `weights`
    instead of uniformly -- matches Section 5.1's mechanism-shift design."""
    rng = np.random.default_rng(seed)
    n_target = int(base_art_mask_precip.sum())
    obs_idx = np.where(real_mask_precip == 1)[0]
    w = weights[obs_idx].astype(np.float64)
    w = w / w.sum()
    chosen = rng.choice(obs_idx, size=n_target, replace=False, p=w)
    art_mask = np.zeros_like(base_art_mask_precip)
    art_mask[chosen] = 1
    return art_mask


# ─────────────────────────────────────────────────────────────────────────
# 4. Feature construction (mirrors direct_two_stage_rf.py exactly)
# ─────────────────────────────────────────────────────────────────────────

def build_occ_X(cor, tmp, na, nm, pidx):
    cols = [i for i in range(cor.shape[1]) if i != pidx]
    parts = [np.nan_to_num(cor[:, cols], nan=0.0).astype(np.float32),
             np.nan_to_num(tmp, nan=0.0).astype(np.float32),
             np.nan_to_num(na, nan=0.0).astype(np.float32),
             np.nan_to_num(nm, nan=0.0).astype(np.float32)]
    return np.concatenate(parts, axis=1)


def build_amt_X(cor_mm, tmp, na_mm, nm, pidx):
    cor = cor_mm.copy(); cor[:, pidx] = 0.0
    return np.concatenate([np.nan_to_num(cor, nan=0.0).astype(np.float32),
                           tmp.astype(np.float32),
                           np.nan_to_num(na_mm, nan=0.0).astype(np.float32),
                           nm.astype(np.float32)], axis=1)


def train_occ(X_tr, y_tr, X_va, y_va, seed):
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    va_p = rf.predict_proba(X_va)[:, 1]
    best_f1, best_cut = -1.0, 0.5
    for cut in np.arange(0.20, 0.82, 0.02):
        f = f1_score(y_va, (va_p >= cut).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_cut = f, float(cut)
    cut = round(best_cut, 3)
    return rf, cut


def eval_f1_rmse(gt, precip_pred_wet, occ_pred_wet, mask_idx):
    """gt, predictions already restricted to masked positions."""
    pw = occ_pred_wet; gw = gt > WET_THRESH
    f1 = f1_score(gw, pw, zero_division=0)
    prec = precision_score(gw, pw, zero_division=0)
    rec = recall_score(gw, pw, zero_division=0)
    bias = float(pw.mean() - gw.mean())
    ws = gw
    rmse_wet = float(np.sqrt(np.mean((precip_pred_wet[ws] - gt[ws]) ** 2))) if ws.sum() else np.nan
    return dict(f1=round(float(f1), 4), precision=round(float(prec), 4), recall=round(float(rec), 4),
               bias=round(bias, 4), rmse_wet=round(rmse_wet, 4) if rmse_wet == rmse_wet else np.nan,
               n=int(mask_idx.sum()))


def main():
    print('=' * 70)
    print('  EXTERNAL VALIDATION -- OHIO (USA) GHCN-DAILY NETWORK, 4 STATIONS')
    print('  RQ1 (mechanism-shift) + RQ4a (graded context-loss)')
    print('=' * 70)

    df = build_panel()
    station_ids = sorted(STATIONS.keys())

    # -- Data quality report --
    qual_rows = []
    for sid in station_ids:
        sub = df[df.STATION_ID == sid]
        n = len(sub)
        for v in VARS:
            n_obs = sub[v].notna().sum()
            qual_rows.append(dict(station=sid, name=STATIONS[sid]['name'], variable=v,
                                  n_days=n, n_observed=int(n_obs), pct_observed=round(100 * n_obs / n, 2)))
        all3 = sub[['TMAX', 'TMIN', 'PRECIP']].notna().all(axis=1).sum()
        qual_rows.append(dict(station=sid, name=STATIONS[sid]['name'], variable='ALL3_JOINT',
                              n_days=n, n_observed=int(all3), pct_observed=round(100 * all3 / n, 2)))
    qual_df = pd.DataFrame(qual_rows)
    qual_df.to_csv(os.path.join(RESULTS_DIR, 'ohio_data_quality.csv'), index=False)
    print('\n-- Data quality (2005-2023) --')
    print(qual_df[qual_df.variable == 'ALL3_JOINT'][['name', 'pct_observed']].to_string(index=False))

    A_knn = build_adjacency(station_ids, k=2)

    # -- Long format, station-major within date (matches original row convention) --
    long_df = df.sort_values(['DATE', 'STATION_ID']).reset_index(drop=True)
    dates_u = np.sort(long_df['DATE'].unique())
    n_dates = len(dates_u); n_sta = len(station_ids)
    print(f'\n  n_dates={n_dates}  n_stations={n_sta}  n_rows={len(long_df)}')

    # temporal features
    doy = long_df['DATE'].dt.dayofyear.to_numpy()
    month = long_df['DATE'].dt.month.to_numpy()
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season = np.array([season_map[m] for m in month], dtype=np.float32)
    temporal = np.stack([
        np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0),
        np.sin(2 * np.pi * month / 12.0), np.cos(2 * np.pi * month / 12.0),
        season,
    ], axis=1).astype(np.float32)

    data = long_df[VARS].to_numpy(dtype=np.float64)
    real_mask = (~np.isnan(data)).astype(np.float32)

    # -- Neighbour raw (mm/degC) features built once from natural data (used
    #    for the "natural" neighbour context; per-scenario corrupted versions
    #    are recomputed after masking, exactly mirroring 01_data_preprocessing.py) --
    wide_natural = long_df.pivot_table(index='DATE', columns='STATION_ID', values=VARS).reindex(dates_u)
    navg_nat, nmask_nat = neighbor_features(wide_natural, A_knn, station_ids, VARS)

    def stack_neighbor(dct):
        # station-major within date, matching long_df row order
        n_var = len(VARS)
        out = np.zeros((n_dates * n_sta, n_var), dtype=np.float32)
        for di in range(n_dates):
            for si, sid in enumerate(station_ids):
                out[di * n_sta + si] = dct[sid][di]
        return out

    # -- Chronological split (70/15/15 by unique date) --
    n_train = int(n_dates * 0.70); n_val = int(n_dates * 0.15)
    tr_cut = dates_u[n_train - 1]; val_cut = dates_u[n_train + n_val - 1]
    tr_idx = long_df['DATE'].to_numpy() <= tr_cut
    va_idx = (long_df['DATE'].to_numpy() > tr_cut) & (long_df['DATE'].to_numpy() <= val_cut)
    te_idx = long_df['DATE'].to_numpy() > val_cut
    print(f'  Train: {tr_idx.sum()} rows ({dates_u[0].astype("M8[D]")} to {tr_cut.astype("M8[D]")})')
    print(f'  Val  : {va_idx.sum()} rows')
    print(f'  Test : {te_idx.sum()} rows ({(dates_u[n_train+n_val]).astype("M8[D]")} to {dates_u[-1].astype("M8[D]")})')

    # -- Scale (fit on train only; simple MinMax on natural, non-NaN values) --
    scaler = MinMaxScaler()
    scaler.fit(np.nan_to_num(data[tr_idx], nan=np.nanmedian(data[tr_idx], axis=0)))

    # -- Artificial 10% MCAR masks, independently per split (mirrors 01_data_preprocessing.py) --
    tr_cor, tr_art = random_missingness(data[tr_idx], real_mask[tr_idx], 0.10, seed=42)
    va_cor, va_art = random_missingness(data[va_idx], real_mask[va_idx], 0.10, seed=42)
    te_cor_10, te_art_10 = random_missingness(data[te_idx], real_mask[te_idx], 0.10, seed=42)

    # -- p95 extreme threshold from train+val NATURAL wet-day distribution --
    trval_precip = np.concatenate([data[tr_idx, PIDX], data[va_idx, PIDX]])
    trval_precip = trval_precip[~np.isnan(trval_precip)]
    wet_trval = trval_precip[trval_precip > WET_THRESH]
    P95 = float(np.percentile(wet_trval, 95))
    print(f'\n  p95 extreme threshold (train+val wet days, n={len(wet_trval)}): {P95:.2f} mm')

    # -- MAR-Meteo (adapted, TMAX-only) + MNAR-Wet + MNAR-Intensity on TEST --
    te_gt_full = data[te_idx]
    te_real = real_mask[te_idx]
    tmax_col = VARS.index('TMAX')
    tmax_obs = te_gt_full[:, tmax_col]
    tmax_norm = (tmax_obs - np.nanmin(tmax_obs)) / (np.nanmax(tmax_obs) - np.nanmin(tmax_obs) + 1e-8)
    tmax_norm = np.nan_to_num(tmax_norm, nan=np.nanmean(tmax_norm))
    w_mar = 1 / (1 + np.exp(-2.0 * tmax_norm))

    precip_gt_te = te_gt_full[:, PIDX]
    w_mnar_wet = np.where(precip_gt_te > WET_THRESH, 3.0, 1.0)
    w_mnar_mod = (1 + np.nan_to_num(precip_gt_te, nan=0.0)) ** 0.5
    w_mnar_sev = (1 + np.nan_to_num(precip_gt_te, nan=0.0)) ** 1.3

    base_art_precip = te_art_10[:, PIDX]
    real_precip_te = te_real[:, PIDX]

    scenarios_te = {'mcar_10pct': (te_cor_10, te_art_10)}
    for name, w in [('mar_meteo', w_mar), ('mnar_wet', w_mnar_wet),
                     ('mnar_intensity_moderate', w_mnar_mod), ('mnar_intensity_severe', w_mnar_sev)]:
        art_precip = weighted_precip_mask(precip_gt_te, real_precip_te, base_art_precip, w, seed=42)
        cor = te_cor_10.copy()  # non-PRECIP columns identical to mcar_10pct
        art = te_art_10.copy()
        cor[:, PIDX] = np.where(art_precip == 1, np.nan, te_gt_full[:, PIDX])
        art[:, PIDX] = art_precip
        scenarios_te[name] = (cor, art)
        realised_rate = 100 * art_precip.sum() / real_precip_te.sum()
        print(f'  [{name:26s}] PRECIP masked cells={int(art_precip.sum())} '
             f'({realised_rate:.1f}% of naturally-observed PRECIP)')

    # -- Neighbour features per scenario (recomputed from each scenario's corrupted array) --
    def neighbor_for_scenario(cor_split_full, split_mask):
        """Rebuild wide panel with this scenario's corruption applied only
        within the split, natural values elsewhere, then recompute neighbour avg/mask."""
        full_data = data.copy()
        full_data[split_mask] = cor_split_full
        tmp_long = long_df.copy()
        for i, v in enumerate(VARS):
            tmp_long[v] = full_data[:, i]
        wide_scn = tmp_long.pivot_table(index='DATE', columns='STATION_ID', values=VARS).reindex(dates_u)
        navg, nmask = neighbor_features(wide_scn, A_knn, station_ids, VARS)
        return stack_neighbor(navg), stack_neighbor(nmask)

    tr_na, tr_nm = neighbor_for_scenario(tr_cor, tr_idx)
    tr_na, tr_nm = tr_na[tr_idx], tr_nm[tr_idx]
    va_na, va_nm = neighbor_for_scenario(va_cor, va_idx)
    va_na, va_nm = va_na[va_idx], va_nm[va_idx]

    te_scenario_features = {}
    for name, (cor, art) in scenarios_te.items():
        na, nm = neighbor_for_scenario(cor, te_idx)
        te_scenario_features[name] = dict(cor=cor, art=art, na=na[te_idx], nm=nm[te_idx])

    # -- Build X matrices --
    X_tr_occ = build_occ_X(tr_cor, temporal[tr_idx], tr_na, tr_nm, PIDX)
    X_va_occ = build_occ_X(va_cor, temporal[va_idx], va_na, va_nm, PIDX)
    X_tr_amt = build_amt_X(np.nan_to_num(tr_cor, nan=0.0), temporal[tr_idx], tr_na, tr_nm, PIDX)

    tr_gt = data[tr_idx][:, PIDX]; va_gt = data[va_idx][:, PIDX]
    tr_obs = real_mask[tr_idx][:, PIDX].astype(bool)
    va_obs = real_mask[va_idx][:, PIDX].astype(bool)
    tr_wet = (tr_gt > WET_THRESH) & tr_obs

    X_tr_occ_obs = X_tr_occ[tr_obs]; y_tr = (tr_gt[tr_obs] > WET_THRESH).astype(int)
    X_va_occ_obs = X_va_occ[va_obs]; y_va = (va_gt[va_obs] > WET_THRESH).astype(int)

    print(f'\n  Feature dim: occurrence={X_tr_occ.shape[1]}  amount={X_tr_amt.shape[1]+0}')
    print(f'  Train obs (occ)={len(X_tr_occ_obs)}  Val obs (occ)={len(X_va_occ_obs)}  Train wet={tr_wet.sum()}')

    # ── RQ1: train + evaluate across mechanism-shift scenarios, 3 seeds ──
    rq1_rows = []
    for seed in SEEDS:
        occ_rf, cut = train_occ(X_tr_occ_obs, y_tr, X_va_occ_obs, y_va, seed)
        amt_rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        amt_rf.fit(X_tr_amt[tr_wet], tr_gt[tr_wet])

        for name, feats in te_scenario_features.items():
            cor, art, na, nm = feats['cor'], feats['art'], feats['na'], feats['nm']
            m = art[:, PIDX] > 0.5
            gt = data[te_idx][:, PIDX][m]
            X_occ_s = build_occ_X(cor, temporal[te_idx], na, nm, PIDX)[m]
            X_amt_s = build_amt_X(np.nan_to_num(cor, nan=0.0), temporal[te_idx], na, nm, PIDX)[m]
            p_occ = occ_rf.predict_proba(X_occ_s)[:, 1]
            pred_wet = p_occ >= cut
            amt_pred = np.zeros(len(gt))
            if pred_wet.sum() > 0:
                amt_pred[pred_wet] = np.maximum(0.0, amt_rf.predict(X_amt_s[pred_wet]))
            row = eval_f1_rmse(gt, amt_pred, pred_wet, m)
            row.update(seed=seed, scenario=name, cutoff=cut)
            rq1_rows.append(row)
            print(f'    seed={seed} [{name:26s}] F1={row["f1"]:.4f} bias={row["bias"]:+.4f} RMSE_wet={row["rmse_wet"]}')

    rq1_df = pd.DataFrame(rq1_rows)
    rq1_df.to_csv(os.path.join(RESULTS_DIR, 'ohio_rq1_mechanism_results.csv'), index=False)

    # ── RQ4a: graded local/neighbour context-loss ablation on mcar_10pct mask ──
    print('\n' + '=' * 70)
    print('  RQ4a: graded local/neighbour context-loss ablation')
    print('=' * 70)
    base_cor, base_art = scenarios_te['mcar_10pct']
    base_m = base_art[:, PIDX] > 0.5
    base_gt = data[te_idx][:, PIDX][base_m]
    n_local = 3  # TMIN, TMAX, TMEAN
    local_cols = [i for i in range(len(VARS)) if i != PIDX]

    # Train each seed's occurrence classifier ONCE and reuse across every
    # ablation level (the classifier itself never changes -- only the
    # inference-time input features do), matching the frozen-model
    # protocol used throughout the primary study's RQ4a.
    trained = {seed: train_occ(X_tr_occ_obs, y_tr, X_va_occ_obs, y_va, seed) for seed in SEEDS}

    def rebuild_neighbor_features(local_drop_by_row=None, neighbor_keep=2):
        """Recompute neighbour avg/mask from the (possibly locally-corrupted)
        raw records under a MODIFIED adjacency (dropping the farthest
        neighbour(s) for neighbor_keep<2) -- mirrors the primary study's
        graded_context_loss.py: 'zeroing the corresponding entries of a copy
        of the adjacency weight matrix and recomputing neighbour-averaged
        features from the unmodified raw records'."""
        cor_mod = base_cor.copy()
        if local_drop_by_row is not None:
            for row, drop_cols in local_drop_by_row.items():
                cor_mod[row, drop_cols] = np.nan
        full_data = data.copy()
        full_data[te_idx] = cor_mod
        tmp_long = long_df.copy()
        for i, v in enumerate(VARS):
            tmp_long[v] = full_data[:, i]
        wide_scn = tmp_long.pivot_table(index='DATE', columns='STATION_ID', values=VARS).reindex(dates_u)
        if neighbor_keep >= 2:
            A_level = A_knn
        elif neighbor_keep == 1:
            # keep only the single nearest neighbour per station (drop the
            # farther of the two k=2 neighbours)
            A_level = A_knn.copy()
            for i in range(len(station_ids)):
                nz = np.where(A_level[i] > 0)[0]
                if len(nz) > 1:
                    farthest = nz[np.argmin(A_level[i, nz])]
                    A_level[i, farthest] = 0.0
        else:
            A_level = np.zeros_like(A_knn)
        navg, nmask = neighbor_features(wide_scn, A_level, station_ids, VARS)
        return cor_mod, stack_neighbor(navg)[te_idx], stack_neighbor(nmask)[te_idx]

    rq4a_rows = []

    def run_family(family, level_desc, local_keep, neighbor_keep):
        for cseed in CONTEXT_SEEDS:
            rng = np.random.default_rng(cseed)
            local_drop_by_row = None
            if local_keep < n_local:
                local_drop_by_row = {}
                for row in np.where(base_m)[0]:
                    avail = list(local_cols)
                    rng.shuffle(avail)
                    local_drop_by_row[row] = avail[:n_local - local_keep]
            cor_mod, na_mod, nm_mod = rebuild_neighbor_features(local_drop_by_row, neighbor_keep)
            for seed in SEEDS:
                occ_rf, cut = trained[seed]
                X_occ_s = build_occ_X(cor_mod, temporal[te_idx], na_mod, nm_mod, PIDX)[base_m]
                p_occ = occ_rf.predict_proba(X_occ_s)[:, 1]
                pred_wet = p_occ >= cut
                gw = base_gt > WET_THRESH
                f1 = f1_score(gw, pred_wet, zero_division=0)
                rq4a_rows.append(dict(family=family, level=level_desc, context_seed=cseed,
                                      model_seed=seed, f1=round(float(f1), 4), n=int(base_m.sum())))
            print(f'    [{family:14s} {level_desc:6s}] cseed={cseed} done')

    run_family('local-loss', '3/3', 3, 2)
    run_family('local-loss', '2/3', 2, 2)
    run_family('local-loss', '1/3', 1, 2)
    run_family('local-loss', '0/3', 0, 2)
    run_family('neighbour-loss', '2/2', 3, 2)
    run_family('neighbour-loss', '1/2', 3, 1)
    run_family('neighbour-loss', '0/2', 3, 0)
    run_family('joint-loss', '3L+2N', 3, 2)
    run_family('joint-loss', '1L+1N', 1, 1)
    run_family('joint-loss', '0L+0N', 0, 0)

    rq4a_df = pd.DataFrame(rq4a_rows)
    rq4a_df.to_csv(os.path.join(RESULTS_DIR, 'ohio_rq4a_graded_context_loss.csv'), index=False)

    agg = rq4a_df.groupby(['family', 'level'])['f1'].agg(['mean', 'std']).reset_index()
    print(agg.to_string(index=False))

    print('\nDONE. Results in', RESULTS_DIR)


if __name__ == '__main__':
    main()

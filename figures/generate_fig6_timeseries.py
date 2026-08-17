# -*- coding: utf-8 -*-
"""
generate_fig6_timeseries.py
=============================
Regenerates Figure 6 (illustrative reconstruction window): ground truth,
SAITS, WGAN-GP (calibrated), and DLPIF (AmountRF_DLPIF) reconstructed full
series from build_canonical_outputs.build_filled_series(), seed 42, 10%
random missingness scenario (the paper's stated "primary realization"), on
the post-2005-restricted, leakage-fixed canonical pipeline.

Window selection follows the same programmatic heuristic described in the
existing Figure 6 caption ("at least two distinct multi-day wet spells
separated by true dry periods"), applied deterministically (first
qualifying window for the first station in station-ID order) so the
selection remains non-manual, matching the caption's existing claim.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, SRC_DIR)

import build_canonical_outputs as bco
import direct_two_stage_rf as d2s

WET_THRESH = 0.1
SCEN = '10pct'
SEED = 42
WINDOW_LEN = 45
MIN_WET_SPELLS = 2
MIN_SPELL_LEN = 2   # a "spell" is >=2 consecutive wet days


def find_illustrative_window(gt, station_mask):
    """First WINDOW_LEN-day window (within one station's rows, in date
    order) containing >= MIN_WET_SPELLS distinct wet spells (each
    >= MIN_SPELL_LEN consecutive wet days) separated by dry gaps."""
    idx = np.where(station_mask)[0]
    gt_st = gt[idx]
    n = len(gt_st)
    for start in range(0, n - WINDOW_LEN):
        w = gt_st[start:start + WINDOW_LEN]
        wet = w > WET_THRESH
        # count distinct spells of >= MIN_SPELL_LEN consecutive wet days
        spells = 0
        run = 0
        for v in wet:
            if v:
                run += 1
            else:
                if run >= MIN_SPELL_LEN:
                    spells += 1
                run = 0
        if run >= MIN_SPELL_LEN:
            spells += 1
        if spells >= MIN_WET_SPELLS:
            return idx[start:start + WINDOW_LEN]
    raise RuntimeError('no qualifying window found')


def main():
    sc, mv = d2s.load_scaler()
    pidx = mv.index('PRECIP')
    te = d2s.load_npz('test')
    dates = pd.to_datetime(te['dates'])
    sids = te['station_ids']
    y_true_full = sc.inverse_transform(
        np.nan_to_num(te['data'].astype(np.float64), nan=0.0))[:, pidx]

    saits_series = bco.build_filled_series('SAITS', SEED, SCEN, sc, pidx, te, dates, sids,
                                           y_true_full)
    wgangp_series = bco.build_filled_series('WGANGP_PrecipFix', SEED, SCEN, sc, pidx, te, dates, sids,
                                            y_true_full)
    dlpif_series = bco.build_filled_series('AmountRF_DLPIF', SEED, SCEN, sc, pidx, te, dates, sids,
                                           y_true_full)
    mask_key = dict((s, mk) for s, _ck, mk in bco.SCENARIOS)[SCEN]
    art_mask_precip = te[mask_key].astype(np.float32)[:, pidx] > 0.5

    station = int(sorted(np.unique(sids))[0])
    station_mask = sids == station
    window_idx = find_illustrative_window(y_true_full, station_mask)

    gt_w = y_true_full[window_idx]
    saits_w = saits_series['y_filled'].to_numpy()[window_idx]
    wgangp_w = wgangp_series['y_filled'].to_numpy()[window_idx]
    dlpif_w = dlpif_series['y_filled'].to_numpy()[window_idx]
    masked_w = art_mask_precip[window_idx]
    t_ax = np.arange(len(window_idx))

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.plot(t_ax, gt_w, color='black', lw=1.6, label='GT Observed', zorder=5)
    ax.scatter(t_ax[masked_w], gt_w[masked_w], color='#888888', s=45,
              label='Masked GT', zorder=6)
    ax.plot(t_ax, saits_w, color='#2ca07f', lw=1.4, ls='-.', label='SAITS', zorder=4)
    ax.plot(t_ax, wgangp_w, color='#8c564b', lw=1.4, ls=':', label='WGAN-GP', zorder=3)
    ax.plot(t_ax, dlpif_w, color='#1f77b4', lw=1.8, label='DLPIF (Ours)', zorder=4)

    ax.set_xlabel('Time Steps (Days)', fontsize=11)
    ax.set_ylabel('Precipitation (mm/day)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right', ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(out_dir, 'figures')
    png_path = os.path.join(figs_dir, 'Figure_6_TimeSeries.png')
    pdf_path = os.path.join(figs_dir, 'Figure_6_TimeSeries.pdf')
    svg_path = os.path.join(figs_dir, 'Figure_6_TimeSeries.svg')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'station={station}  window rows={window_idx[0]}..{window_idx[-1]}  '
         f'dates={dates[window_idx[0]].date()}..{dates[window_idx[-1]].date()}')
    print(f'Saved:\n  {png_path}\n  {pdf_path}\n  {svg_path}')


if __name__ == '__main__':
    main()

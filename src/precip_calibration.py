"""
precip_calibration.py  — Precipitation post-processing / calibration module
=============================================================================
Fixes the "always wet" GAN behavior by:
  1. Tuning a wet-day threshold on VALIDATION data only (no test leakage).
  2. Applying quantile-mapping on positive precipitation values (optional).
  3. Clipping negatives to zero.

Public API
----------
  calibrator = PrecipCalibrator(precip_idx, scaler)
  calibrator.fit_threshold(val_imp_norm, val_gt_norm, val_art_mask)
  cal_arr = calibrator.apply(imp_norm)
  calibrator.save(path)
  calibrator = PrecipCalibrator.load(path)
"""

import json
import os
import warnings

import numpy as np

WET_THRESH_DEFAULT = 0.1   # mm — standard WMO wet-day threshold
THRESH_GRID        = list(np.round(np.arange(0.0, 2.05, 0.05), 3))


def _to_orig(scaler, arr_norm, idx):
    """Return 1-D original-unit array for column `idx` from a normalised 2-D array."""
    arr_norm = np.clip(arr_norm, 0.0, 1.0)
    orig = scaler.inverse_transform(arr_norm)
    return orig[:, idx]

def _wetday_metrics(pred_mm, gt_mm, thresh):
    """Compute wet-day classification metrics at a given threshold (mm)."""
    pred_wet = pred_mm > thresh
    gt_wet   = gt_mm   > WET_THRESH_DEFAULT   # GT always evaluated at 0.1 mm

    tp = int(( pred_wet &  gt_wet).sum())
    fp = int(( pred_wet & ~gt_wet).sum())
    fn = int((~pred_wet &  gt_wet).sum())
    tn = int((~pred_wet & ~gt_wet).sum())
    n  = len(pred_mm)

    freq_gt   = float(gt_wet.mean())  if n > 0 else np.nan
    freq_pred = float(pred_wet.mean()) if n > 0 else np.nan
    bias      = freq_pred - freq_gt

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    dry_acc = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    csi  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        'thresh'    : thresh,
        'freq_gt'   : round(freq_gt,   4),
        'freq_pred' : round(freq_pred,  4),
        'bias'      : round(bias,       4),
        'precision' : round(prec,       4),
        'recall'    : round(rec,        4),
        'f1'        : round(f1,         4),
        'dry_acc'   : round(dry_acc,    4) if not np.isnan(dry_acc) else np.nan,
        'csi'       : round(csi,        4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }

# Quantile mapping (optional)

def _build_qmap(ref_values, n_quantiles=200):
    """Build a CDF mapping from uniform quantiles → reference distribution."""
    ref_pos = ref_values[ref_values > 0.0]
    if len(ref_pos) < 10:
        return None
    q_levels = np.linspace(0.0, 1.0, n_quantiles + 1)
    q_values = np.percentile(ref_pos, q_levels * 100.0)
    return {'q_levels': q_levels.tolist(), 'q_values': q_values.tolist()}

def _apply_qmap(pred_pos, qmap):
    """Map positive predicted precipitation values to the reference distribution."""
    if qmap is None or len(pred_pos) == 0:
        return pred_pos
    q_levels = np.array(qmap['q_levels'])
    q_values = np.array(qmap['q_values'])
    # Get empirical CDF rank of each prediction, then map to reference
    pred_cdf = np.interp(pred_pos,
                         np.percentile(pred_pos, q_levels * 100.0),
                         q_levels)
    return np.interp(pred_cdf, q_levels, q_values).astype(np.float32)

# Main class

class PrecipCalibrator:
    """
    Fits a wet-day threshold on validation data and applies post-processing
    to GAN-imputed PRECIP values only.

    Parameters
    ----------
    precip_idx : int
        Column index of PRECIP in the (n_rows, n_meteo) normalised arrays.
    scaler : sklearn MinMaxScaler
        Fitted scaler from scaler.pkl (used to convert to original mm units).
    use_quantile_mapping : bool
        Whether to apply quantile mapping on positive predicted values.
    """

    def __init__(self, precip_idx, scaler, use_quantile_mapping=True):
        self.precip_idx           = precip_idx
        self.scaler               = scaler
        self.use_quantile_mapping = use_quantile_mapping

        # Fitted parameters (set by fit_threshold)
        self.threshold_mm          = WET_THRESH_DEFAULT
        self.val_metrics_before    = {}
        self.val_metrics_after     = {}
        self.qmap                  = None
        self._fit_done             = False

    # Fit

    def fit_threshold(self, val_imp_norm, val_gt_norm, val_art_mask,
                      verbose=True):
        """
        Tune wet-day threshold using VALIDATION data only.

        Parameters
        ----------
        val_imp_norm  : np.ndarray  (n_val, n_meteo) — normalised GAN output
        val_gt_norm   : np.ndarray  (n_val, n_meteo) — normalised ground truth
        val_art_mask  : np.ndarray  (n_val, n_meteo) — 1 = artificially hidden
        verbose       : bool
        """
        idx = self.precip_idx
        mask = val_art_mask[:, idx].astype(bool)

        if mask.sum() == 0:
            warnings.warn("[PrecipCalibrator] No masked PRECIP cells in val set; "
                          "keeping default threshold.")
            return

        pred_mm = _to_orig(self.scaler, val_imp_norm, idx)[mask]
        gt_mm   = _to_orig(self.scaler, val_gt_norm,  idx)[mask]

        # Guard: all-zero or constant predictions
        if pred_mm.max() <= 0:
            warnings.warn("[PrecipCalibrator] All predicted PRECIP are ≤0; "
                          "skipping threshold search, using 0.0.")
            self.threshold_mm = 0.0
            self._compute_before_after(pred_mm, gt_mm, verbose)
            return

        # Search threshold grid
        results = [_wetday_metrics(pred_mm, gt_mm, t) for t in THRESH_GRID]

        # Primary: minimise |bias|; secondary: maximise F1
        results.sort(key=lambda x: (abs(x['bias']), -x['f1']))
        best = results[0]
        self.threshold_mm = best['thresh']

        # Metrics BEFORE fix (at default 0.1 mm threshold)
        self.val_metrics_before = _wetday_metrics(pred_mm, gt_mm, WET_THRESH_DEFAULT)

        # Metrics AFTER fix (at chosen threshold)
        pred_cal = self._calibrate_precip_1d(pred_mm)
        self.val_metrics_after = _wetday_metrics(pred_cal, gt_mm, WET_THRESH_DEFAULT)

        if verbose:
            self._print_fit_summary()

        # Build quantile map from POSITIVE validation ground-truth values
        if self.use_quantile_mapping:
            gt_pos = gt_mm[gt_mm > WET_THRESH_DEFAULT]
            self.qmap = _build_qmap(gt_pos)

        self._fit_done = True

    def _compute_before_after(self, pred_mm, gt_mm, verbose):
        self.val_metrics_before = _wetday_metrics(pred_mm, gt_mm, WET_THRESH_DEFAULT)
        pred_cal = self._calibrate_precip_1d(pred_mm)
        self.val_metrics_after  = _wetday_metrics(pred_cal, gt_mm, WET_THRESH_DEFAULT)
        if verbose:
            self._print_fit_summary()
        self._fit_done = True

    def _print_fit_summary(self):
        b = self.val_metrics_before
        a = self.val_metrics_after
        print(f"\n  [PrecipCalibrator] Chosen wet-day threshold : {self.threshold_mm:.2f} mm")
        print(f"  Val wet-day freq  BEFORE fix : gt={b['freq_gt']:.4f}  "
              f"pred={b['freq_pred']:.4f}  bias={b['bias']:+.4f}  F1={b['f1']:.4f}")
        print(f"  Val wet-day freq  AFTER  fix : gt={a['freq_gt']:.4f}  "
              f"pred={a['freq_pred']:.4f}  bias={a['bias']:+.4f}  F1={a['f1']:.4f}")

    # Apply

    def _calibrate_precip_1d(self, pred_mm):
        """Apply threshold and (optionally) quantile mapping to a 1-D mm array."""
        cal = pred_mm.copy().astype(np.float32)
        dry_mask = cal <= self.threshold_mm
        cal[dry_mask] = 0.0

        if self.use_quantile_mapping and self.qmap is not None:
            wet_mask = ~dry_mask
            if wet_mask.sum() > 0:
                cal[wet_mask] = _apply_qmap(cal[wet_mask], self.qmap)

        # Hard safety: no negatives
        cal = np.clip(cal, 0.0, None)
        return cal

    def apply(self, imp_norm, art_mask=None):
        """
        Post-process a full normalised imputation array.

        Only the PRECIP column and only on imputed cells (where art_mask == 1)
        are modified.  If art_mask is None, ALL cells are post-processed (useful
        for whole-array calibration).

        Parameters
        ----------
        imp_norm : np.ndarray  (n_rows, n_meteo)   normalised GAN output
        art_mask : np.ndarray  (n_rows, n_meteo)   1 = imputed position  [optional]

        Returns
        -------
        cal_norm : np.ndarray  (n_rows, n_meteo)   calibrated normalised array
                   (only PRECIP column is changed)
        """
        if not self._fit_done:
            warnings.warn("[PrecipCalibrator] fit_threshold() was not called; "
                          "applying default threshold only.")

        cal = imp_norm.copy().astype(np.float32)
        idx = self.precip_idx

        # Convert to original units
        imp_orig = _to_orig(self.scaler, cal, idx)

        # Build the row-level mask for imputed positions
        if art_mask is not None:
            row_mask = art_mask[:, idx].astype(bool)
        else:
            row_mask = np.ones(len(cal), dtype=bool)

        if row_mask.sum() == 0:
            return cal

        # Calibrate only imputed rows
        imp_orig[row_mask] = self._calibrate_precip_1d(imp_orig[row_mask])

        # Re-normalise ONLY the PRECIP column back to [0,1]
        scaler_min = self.scaler.data_min_[idx]
        scaler_rng = self.scaler.data_range_[idx]
        if scaler_rng > 0:
            cal[:, idx] = np.clip(
                (imp_orig - scaler_min) / scaler_rng, 0.0, 1.0
            )
        else:
            cal[:, idx] = 0.0

        return cal

    # Serialise / deserialise

    def to_dict(self):
        return {
            'precip_idx'          : int(self.precip_idx),
            'threshold_mm'        : float(self.threshold_mm),
            'use_quantile_mapping': bool(self.use_quantile_mapping),
            'qmap'                : self.qmap,
            'val_metrics_before'  : self.val_metrics_before,
            'val_metrics_after'   : self.val_metrics_after,
            'fit_done'            : bool(self._fit_done),
        }

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, allow_nan=False,
                      default=lambda x: None)
        print(f"  Calibration params → {path}")

    @classmethod
    def from_dict(cls, d, scaler):
        obj = cls(
            precip_idx=d['precip_idx'],
            scaler=scaler,
            use_quantile_mapping=d.get('use_quantile_mapping', True),
        )
        obj.threshold_mm       = d.get('threshold_mm', WET_THRESH_DEFAULT)
        obj.qmap               = d.get('qmap', None)
        obj.val_metrics_before = d.get('val_metrics_before', {})
        obj.val_metrics_after  = d.get('val_metrics_after',  {})
        obj._fit_done          = d.get('fit_done', False)
        return obj

    @classmethod
    def load(cls, path, scaler):
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return cls.from_dict(d, scaler)

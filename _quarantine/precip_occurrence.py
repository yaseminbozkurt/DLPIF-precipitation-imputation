"""
precip_occurrence.py  —  Wet-day occurrence classifier for precipitation
=========================================================================
Stage A of the two-stage precipitation imputation approach.

Design
------
  Classifier  : RandomForestClassifier(n_estimators=300, class_weight='balanced')
  Train data  : observed rows in train split (real_mask[:,precip_idx]==1) ONLY
  Cutoff      : tuned on validation observed rows by maximising F1
  Leakage     : test GT is NEVER used for fitting or cutoff tuning

Public API (module-level)
--------------------------
  WET_THRESH_MM           float constant (0.1 mm)
  _build_features(...)    feature concatenation helper
  _wet_labels(...)        mm array → binary label array

  PrecipOccurrenceModel
    .fit(tr_corr, tr_temporal, tr_nbr, tr_gt_norm, tr_real_mask,
         va_corr, va_temporal, va_nbr, va_gt_norm, va_real_mask,
         precip_idx, scaler, gan_tr_precip=None, gan_va_precip=None,
         verbose=True)
    .predict_proba(corr, temporal, nbr, gan_precip=None) -> (n,) float32
    .predict(corr, temporal, nbr, gan_precip=None) -> (n,) int32
    .save(path_prefix)        -> writes .pkl + .json
    .load(path_prefix)        classmethod
"""

import json
import pickle
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# ─── Public constant ─────────────────────────────────────────────────────────
WET_THRESH_MM = 0.1   # WMO standard wet-day threshold


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _build_features(corr, temporal, nbr, gan_precip=None):
    """
    Concatenate per-row feature groups into a 2-D design matrix.

    Parameters
    ----------
    corr       : (n, n_meteo)    normalised corrupted array (known=real, masked=0)
    temporal   : (n, n_temporal) temporal features (sin/cos DOY, month …)
    nbr        : (n, n_meteo)    neighbour average (zeros for Mode A)
    gan_precip : (n,) or (n,1) or None   GAN raw PRECIP column [optional]

    Returns
    -------
    X : (n, n_features)  float32
    """
    parts = [
        np.nan_to_num(corr,     nan=0.0).astype(np.float32),
        np.nan_to_num(temporal, nan=0.0).astype(np.float32),
        np.nan_to_num(nbr,      nan=0.0).astype(np.float32),
    ]
    if gan_precip is not None:
        gp = np.nan_to_num(gan_precip, nan=0.0).astype(np.float32)
        if gp.ndim == 1:
            gp = gp[:, None]
        parts.append(gp)
    return np.concatenate(parts, axis=1)


def _wet_labels(precip_mm):
    """Convert mm array to binary wet (1) / dry (0) labels."""
    return (precip_mm > WET_THRESH_MM).astype(np.int32)


# ─── Classifier ──────────────────────────────────────────────────────────────

class PrecipOccurrenceModel:
    """
    Random Forest wet-day occurrence classifier.

    Parameters
    ----------
    seed             : int  random seed (match GAN seed for reproducibility)
    n_estimators     : int  number of trees (default 300)
    min_samples_leaf : int  minimum leaf size (default 5, prevents overfit)
    """

    def __init__(self, seed=42, n_estimators=300, min_samples_leaf=5):
        self.seed             = int(seed)
        self.n_estimators     = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.precip_idx       = None
        self.scaler           = None
        self.cutoff           = 0.5
        self.train_metrics    = {}
        self.val_metrics      = {}
        self._rf              = None
        self._fit_done        = False

    # ------------------------------------------------------------------
    def fit(self,
            tr_corr, tr_temporal, tr_nbr, tr_gt_norm, tr_real_mask,
            va_corr, va_temporal, va_nbr, va_gt_norm, va_real_mask,
            precip_idx, scaler,
            gan_tr_precip=None, gan_va_precip=None,
            verbose=True):
        """
        Fit RF on observed TRAIN rows; tune probability cutoff on observed VAL rows.

        Only rows where real_mask[:,precip_idx]==1 are used for training.
        Cutoff is chosen to maximise val-set F1 (grid search 0.20→0.80).
        Test ground truth is NEVER touched here.
        """
        self.precip_idx = int(precip_idx)
        self.scaler     = scaler

        sc_min = scaler.data_min_[precip_idx]
        sc_rng = scaler.data_range_[precip_idx]

        def to_mm(norm_col):
            return np.clip(norm_col, 0, 1) * sc_rng + sc_min

        # Select observed rows only
        tr_obs = tr_real_mask[:, precip_idx].astype(bool)
        va_obs = va_real_mask[:, precip_idx].astype(bool)

        if tr_obs.sum() == 0:
            warnings.warn("[PrecipOccurrenceModel] No observed PRECIP rows in train!")
            return

        X_tr = _build_features(tr_corr, tr_temporal, tr_nbr, gan_tr_precip)[tr_obs]
        y_tr = _wet_labels(to_mm(tr_gt_norm[tr_obs, precip_idx]))

        X_va = _build_features(va_corr, va_temporal, va_nbr, gan_va_precip)[va_obs]
        y_va = _wet_labels(to_mm(va_gt_norm[va_obs, precip_idx]))

        if verbose:
            print(f"\n  [PrecipOccurrenceModel] Training  (seed={self.seed})")
            print(f"  Train observed rows : {len(X_tr):,}  wet={y_tr.mean():.3f}")
            print(f"  Val   observed rows : {len(X_va):,}  wet={y_va.mean():.3f}")
            print(f"  Features per row    : {X_tr.shape[1]}")

        self._rf = RandomForestClassifier(
            n_estimators     = self.n_estimators,
            min_samples_leaf = self.min_samples_leaf,
            class_weight     = 'balanced',
            random_state     = self.seed,
            n_jobs           = -1,
        )
        self._rf.fit(X_tr, y_tr)

        # Train metrics
        y_tr_pred = self._rf.predict(X_tr)
        self.train_metrics = {
            'n'        : int(len(y_tr)),
            'wet_frac' : float(round(y_tr.mean(), 4)),
            'f1'       : float(round(f1_score(y_tr, y_tr_pred, zero_division=0), 4)),
            'precision': float(round(precision_score(y_tr, y_tr_pred, zero_division=0), 4)),
            'recall'   : float(round(recall_score(y_tr, y_tr_pred, zero_division=0), 4)),
        }

        # Cutoff grid search on val (maximise F1)
        va_proba = self._rf.predict_proba(X_va)[:, 1]
        best_f1, best_cut = -1.0, 0.5
        for cut in np.arange(0.20, 0.82, 0.02):
            y_pred = (va_proba >= cut).astype(int)
            f = f1_score(y_va, y_pred, zero_division=0)
            if f > best_f1:
                best_f1, best_cut = f, float(cut)
        self.cutoff = round(best_cut, 3)

        y_va_pred = (va_proba >= self.cutoff).astype(int)
        self.val_metrics = {
            'n'           : int(len(y_va)),
            'wet_frac_gt' : float(round(float(y_va.mean()), 4)),
            'wet_frac_pred': float(round(float(y_va_pred.mean()), 4)),
            'bias'        : float(round(float(y_va_pred.mean() - y_va.mean()), 4)),
            'f1'          : float(round(f1_score(y_va, y_va_pred, zero_division=0), 4)),
            'precision'   : float(round(precision_score(y_va, y_va_pred, zero_division=0), 4)),
            'recall'      : float(round(recall_score(y_va, y_va_pred, zero_division=0), 4)),
        }
        self._fit_done = True

        if verbose:
            vm = self.val_metrics
            print(f"  --- Train (in-sample) ---")
            tm = self.train_metrics
            print(f"  F1={tm['f1']:.4f}  P={tm['precision']:.4f}  R={tm['recall']:.4f}")
            print(f"  --- Val (cutoff tuning) ---")
            print(f"  Cutoff     : {self.cutoff:.3f}")
            print(f"  Wet freq   : gt={vm['wet_frac_gt']:.4f}  pred={vm['wet_frac_pred']:.4f}  "
                  f"bias={vm['bias']:+.4f}")
            print(f"  F1={vm['f1']:.4f}  P={vm['precision']:.4f}  R={vm['recall']:.4f}")

    # ------------------------------------------------------------------
    def predict_proba(self, corr, temporal, nbr, gan_precip=None):
        """Return P(wet) per row. Shape: (n,) float32"""
        if not self._fit_done:
            raise RuntimeError("Call fit() first.")
        X = _build_features(corr, temporal, nbr, gan_precip)
        return self._rf.predict_proba(X)[:, 1].astype(np.float32)

    def predict(self, corr, temporal, nbr, gan_precip=None, cutoff=None):
        """Return binary wet-day labels (0/1). Shape: (n,) int32"""
        proba = self.predict_proba(corr, temporal, nbr, gan_precip)
        cut   = cutoff if cutoff is not None else self.cutoff
        return (proba >= cut).astype(np.int32)

    # ------------------------------------------------------------------
    def save(self, path_prefix):
        """Save to {path_prefix}.pkl  and  {path_prefix}.json"""
        with open(path_prefix + '.pkl', 'wb') as f:
            pickle.dump({'rf': self._rf, 'scaler': self.scaler}, f)
        meta = {
            'seed'            : self.seed,
            'n_estimators'    : self.n_estimators,
            'min_samples_leaf': self.min_samples_leaf,
            'precip_idx'      : self.precip_idx,
            'cutoff'          : self.cutoff,
            'train_metrics'   : self.train_metrics,
            'val_metrics'     : self.val_metrics,
            'fit_done'        : self._fit_done,
        }
        with open(path_prefix + '.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f"  Occurrence model   → {path_prefix}.pkl  +  .json")

    @classmethod
    def load(cls, path_prefix):
        """Load from saved files."""
        with open(path_prefix + '.json', 'r', encoding='utf-8') as f:
            meta = json.load(f)
        with open(path_prefix + '.pkl', 'rb') as f:
            objs = pickle.load(f)
        obj = cls(seed=meta['seed'],
                  n_estimators=meta['n_estimators'],
                  min_samples_leaf=meta.get('min_samples_leaf', 5))
        obj.precip_idx    = meta['precip_idx']
        obj.cutoff        = meta['cutoff']
        obj.train_metrics = meta.get('train_metrics', {})
        obj.val_metrics   = meta.get('val_metrics', {})
        obj._fit_done     = meta.get('fit_done', False)
        obj._rf           = objs['rf']
        obj.scaler        = objs['scaler']
        return obj

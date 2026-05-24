"""
baselines_dl/train_saits_v2.py
================================
SAITS v2 — production training script with both precipitation fixes:

  Fix 1: log1p PRECIP normalization
    log1p(mm) / log1p(165) -> threshold 0.1mm = 0.0186 in norm space (31x vs 0.000606)

  Fix 2: Last-epoch checkpoint for inference
    model_saving_strategy="all" -> saves SAITS_epoch{N}.pypots each epoch
    After fit(), explicitly load the last epoch checkpoint before imputation.
    (pypots always reloads best-val checkpoint in memory after fit(); this overrides it.)

  Fix 3: epochs=80, patience=None (no early stopping)

Output (isolated, nothing else modified):
  experiments_dl/saits_v2_seed{N}/
  results_dl/saits_v2/
  logs_dl/saits_v2_seed{N}.log

Usage:
  python baselines_dl/train_saits_v2.py --seed 42
"""

import os, sys, glob, re, json, time, random, logging, pickle, argparse
import numpy as np
import torch

PROJECT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, "experiments_dl")
RESULTS_DIR     = os.path.join(PROJECT_DIR, "results_dl")
LOGS_DIR        = os.path.join(PROJECT_DIR, "logs_dl")
V2_RESULTS_DIR  = os.path.join(RESULTS_DIR, "saits_v2")
sys.path.insert(0, PROJECT_DIR)

N_STEPS    = 30
N_FEATURES = 7
EPOCHS     = 80
PATIENCE   = None   # disabled — train all epochs

SAITS_CFG = dict(
    n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16,
    d_ffn=128, dropout=0.1, attn_dropout=0.0,
    diagonal_attention_mask=True, ORT_weight=1, MIT_weight=1,
    batch_size=32,
)
LOG_MAX = float(np.log1p(165.0))   # = 5.1120


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Logger ────────────────────────────────────────────────────────────────────
def build_logger(path, name):
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    log.addHandler(logging.FileHandler(path, mode="w", encoding="utf-8"))
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    log.addHandler(ch)
    return log


# ── PRECIP transforms ────────────────────────────────────────────────────────
def to_log1p(flat, pidx, sc):
    """MinMax [0,1] PRECIP -> log1p-norm [0,1]. NaN preserved."""
    out = flat.copy()
    mm  = sc.data_min_[pidx] + flat[:, pidx].astype(np.float64) * sc.data_range_[pidx]
    out[:, pidx] = np.where(np.isnan(mm), np.nan,
                            np.log1p(np.maximum(mm, 0)) / LOG_MAX).astype(np.float32)
    return out

def log1p_to_mm(flat, pidx):
    """log1p-norm PRECIP -> mm."""
    out = flat.copy()
    out[:, pidx] = np.expm1(np.clip(flat[:, pidx], 0, 1).astype(np.float64) * LOG_MAX).astype(np.float32)
    return out

def mm_to_norm(flat, pidx, sc):
    """mm PRECIP -> MinMax [0,1]  (compatible with evaluate_saits.py)."""
    out = flat.copy()
    out[:, pidx] = np.clip(flat[:, pidx] / sc.data_range_[pidx], 0, 1).astype(np.float32)
    return out


# ── Windowing helpers ─────────────────────────────────────────────────────────
def build_3d(arr, ids, n_steps=N_STEPS):
    uniq = list(dict.fromkeys(ids.tolist()))
    segs = []
    for sid in uniq:
        sta = arr[ids == sid].astype(np.float32)
        n_w = len(sta) // n_steps
        segs.append(sta[:n_w * n_steps].reshape(n_w, n_steps, -1))
    return np.concatenate(segs, axis=0)

def reconstruct_flat(imp3d, ids, n_steps=N_STEPS):
    uniq  = list(dict.fromkeys(ids.tolist()))
    n_sta = len(uniq); F = imp3d.shape[2]
    wins  = imp3d.shape[0] // n_sta; T = wins * n_steps
    arrs  = [imp3d[i*wins:(i+1)*wins].reshape(T, F) for i in range(n_sta)]
    flat  = np.empty((T * n_sta, F), dtype=np.float32)
    for t in range(T):
        for s in range(n_sta): flat[t * n_sta + s] = arrs[s][t]
    return flat


# ── Quick precipitation metrics ───────────────────────────────────────────────
def precip_metrics(pred_mm, gt_mm, mask_1d, pidx, thresh=0.1):
    sel     = mask_1d.astype(bool)
    p, g    = pred_mm[sel, pidx], gt_mm[sel, pidx]
    pw, gw  = p >= thresh, g >= thresh
    tp = int(( pw &  gw).sum()); fp = int(( pw & ~gw).sum()); fn = int((~pw &  gw).sum())
    prec = tp/(tp+fp) if tp+fp else 0.0; rec = tp/(tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    csi  = tp/(tp+fp+fn) if tp+fp+fn else 0.0
    rmse_wet = float(np.sqrt(np.mean((p[gw]-g[gw])**2))) if gw.sum() > 0 else float("nan")
    return {
        "freq_gt":    round(float(gw.mean()), 4),
        "freq_pred":  round(float(pw.mean()), 4),
        "bias":       round(float(pw.mean()-gw.mean()), 4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "F1":         round(f1,   4),
        "CSI":        round(csi,  4),
        "RMSE_wet":   round(rmse_wet, 4),
        "frac_dry":   round(float((p < thresh).mean()), 4),
    }


# ── Epoch-N checkpoint loader ─────────────────────────────────────────────────
def load_last_epoch_checkpoint(saits, exp_dir, log):
    """After fit(), reload the last-epoch checkpoint to override best-val (epoch-1)."""
    pattern = os.path.join(exp_dir, "**", "SAITS_epoch*.pypots")
    ckpts   = glob.glob(pattern, recursive=True)
    if not ckpts:
        log.warning("  No per-epoch checkpoints found; using current in-memory model.")
        return None

    def epoch_num(p):
        m = re.search(r"epoch(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    last_ckpt   = max(ckpts, key=epoch_num)
    last_epoch  = epoch_num(last_ckpt)
    log.info(f"  Loading last-epoch checkpoint: epoch {last_epoch} -> {os.path.basename(last_ckpt)}")
    saits.load(last_ckpt)
    log.info(f"  Checkpoint loaded (overrides epoch-1 best-val model).")
    return last_epoch


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    SEED = args.seed

    run_name = f"saits_v2_seed{SEED}"
    exp_dir  = os.path.join(EXPERIMENTS_DIR, run_name)
    log_path = os.path.join(LOGS_DIR, f"{run_name}.log")

    os.makedirs(exp_dir,        exist_ok=True)
    os.makedirs(V2_RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,       exist_ok=True)

    log = build_logger(log_path, run_name)
    log.info("=" * 60)
    log.info(f"  SAITS v2  |  {run_name}")
    log.info(f"  Fix1: log1p PRECIP  LOG_MAX={LOG_MAX:.4f}")
    log.info(f"  Fix2: model_saving_strategy=all + load last epoch")
    log.info(f"  Fix3: epochs={EPOCHS}  patience={PATIENCE}")
    log.info("=" * 60)

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Seed={SEED}  Device={device}")

    # ── Data ──────────────────────────────────────────────────────────────
    with open(os.path.join(PROJECT_DIR, "scaler.pkl"), "rb") as f:
        sc_data = pickle.load(f)
    sc    = sc_data["scaler"]
    mvars = list(sc_data["meteo_vars"])
    pidx  = mvars.index("PRECIP")
    log.info(f"Variables: {mvars}  PRECIP_idx={pidx}")

    npz_tr = np.load(os.path.join(PROJECT_DIR, "preprocessed_train.npz"), allow_pickle=True)
    npz_va = np.load(os.path.join(PROJECT_DIR, "preprocessed_val.npz"),   allow_pickle=True)
    npz_te = np.load(os.path.join(PROJECT_DIR, "preprocessed_test.npz"),  allow_pickle=True)
    ids_tr = npz_tr["station_ids"]; ids_va = npz_va["station_ids"]; ids_te = npz_te["station_ids"]

    # Apply log1p PRECIP transform and build 3-D arrays
    X_tr  = build_3d(to_log1p(npz_tr["data"].astype(np.float32), pidx, sc), ids_tr)
    X_va  = build_3d(to_log1p(npz_va["data"].astype(np.float32), pidx, sc), ids_va)
    thresh_new = np.log1p(0.1) / LOG_MAX
    log.info(f"train X: {X_tr.shape}  val X: {X_va.shape}")
    log.info(f"PRECIP threshold: 0.1mm = {thresh_new:.6f} in log1p-norm  (was 0.000606)")

    # Save config
    json.dump({
        "run_name": run_name, "seed": SEED, "epochs": EPOCHS,
        "patience": None, "model_saving_strategy": "all",
        "precip_transform": "log1p", "log_max": LOG_MAX,
        "thresh_norm_before": 0.000606, "thresh_norm_after": round(thresh_new, 6),
        **SAITS_CFG,
    }, open(os.path.join(exp_dir, "config.json"), "w"), indent=2)

    # ── Train ─────────────────────────────────────────────────────────────
    # Skip training if epoch-80 checkpoint already exists
    pattern_existing = os.path.join(exp_dir, "**", "SAITS_epoch*.pypots")
    existing_ckpts   = glob.glob(pattern_existing, recursive=True)
    def epoch_num_fn(p):
        m = re.search(r"epoch(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    max_existing = max((epoch_num_fn(c) for c in existing_ckpts), default=0)

    from pypots.imputation import SAITS

    if max_existing >= EPOCHS:
        log.info(f"Skip training: epoch-{max_existing} checkpoint already exists.")
        saits = SAITS(
            n_steps=N_STEPS, n_features=N_FEATURES, **SAITS_CFG,
            epochs=1, patience=None, device=device,
            saving_path=None, model_saving_strategy=None, verbose=False,
        )
    else:
        log.info("Training ...")
        saits = SAITS(
            n_steps=N_STEPS, n_features=N_FEATURES, **SAITS_CFG,
            epochs=EPOCHS, patience=PATIENCE, device=device,
            saving_path=exp_dir,
            model_saving_strategy="all",     # saves epoch-N.pypots each epoch
            verbose=True,
        )
        t0 = time.time()
        saits.fit(
            train_set={"X": X_tr, "X_ori": X_tr.copy()},
            val_set={"X": X_va, "X_ori": X_va.copy()},
        )
        log.info(f"Training done: {(time.time()-t0)/60:.1f} min")

    # ── Load last-epoch checkpoint (overrides epoch-1 best-val) ───────────
    last_epoch = load_last_epoch_checkpoint(saits, exp_dir, log)

    # ── Ground truth in mm ────────────────────────────────────────────────
    N = 11160
    gt_norm = npz_te["data"].astype(np.float32)[:N]
    gt_mm   = np.empty_like(gt_norm)
    for i in range(N_FEATURES):
        gt_mm[:, i] = (sc.data_min_[i] + np.clip(gt_norm[:,i],0,1)*sc.data_range_[i]).astype(np.float32)

    # ── Impute all scenarios ───────────────────────────────────────────────
    SCENARIOS = {
        "10pct":    ("corrupted_10pct",    "art_mask_10pct"),
        "20pct":    ("corrupted_20pct",    "art_mask_20pct"),
        "block7d":  ("corrupted_block7d",  "art_mask_block7d"),
        "block30d": ("corrupted_block30d", "art_mask_block30d"),
    }

    all_rows = []
    log.info("Generating test predictions (epoch-80 model) ...")

    for scen, (ck, ak) in SCENARIOS.items():
        if ck not in npz_te.files: continue

        corr_log  = to_log1p(npz_te[ck].astype(np.float32), pidx, sc)
        X_te      = build_3d(corr_log, ids_te)
        imp3d     = saits.impute({"X": X_te}).astype(np.float32)

        flat_log  = reconstruct_flat(imp3d, ids_te)
        flat_mm   = log1p_to_mm(flat_log, pidx)
        flat_norm = mm_to_norm(flat_mm, pidx, sc)   # back to [0,1] for eval compatibility

        # Save flat npy
        npy_path = os.path.join(V2_RESULTS_DIR,
                   f"saits_imputed_test_{scen}_v2_seed{SEED}_flat.npy")
        np.save(npy_path, flat_norm)

        # Compute mm for non-PRECIP columns
        pred_mm = np.empty_like(flat_mm)
        for i in range(N_FEATURES):
            if i == pidx:
                pred_mm[:, i] = flat_mm[:, i]
            else:
                pred_mm[:, i] = (sc.data_min_[i] + np.clip(flat_log[:,i],0,1)*sc.data_range_[i]).astype(np.float32)

        mask_1d = npz_te[ak].astype(np.float32)[:N, pidx]
        m = precip_metrics(pred_mm, gt_mm, mask_1d, pidx)
        m.update({"method": f"SAITS_v2_seed{SEED}", "scenario": scen, "epoch": last_epoch})
        all_rows.append(m)

        log.info(f"  [{scen}] freq_pred={m['freq_pred']:.4f}  bias={m['bias']:+.4f}  "
                 f"F1={m['F1']:.4f}  frac_dry={m['frac_dry']:.4f}")

    # ── Save per-seed CSV ─────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(V2_RESULTS_DIR, f"evaluation_v2_seed{SEED}.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(all_rows)
        log.info(f"Per-seed CSV saved: {csv_path}")

    log.info("=" * 60)
    log.info(f"  DONE  |  seed={SEED}  last_epoch={last_epoch}")
    log.info(f"  Checkpoint   : {exp_dir}")
    log.info(f"  Predictions  : {V2_RESULTS_DIR}")
    log.info(f"  Per-seed CSV : {csv_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

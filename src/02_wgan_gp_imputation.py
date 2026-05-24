"""
02_wgan_gp_imputation.py  — Mode A (temporal only) + SEQ_LEN ablation
======================================================================
WGAN-GP with Bidirectional LSTM — Q1 Minimal Pipeline.

Mode A: Input = [corrupted | comb_mask | temporal]
Ablation: SEQ_LEN in [30, 90], SEED=42

Ablation outputs (one set per seq_len):
  gan_model_seq30_seed42.pt      gan_model_seq90_seed42.pt
  gan_imputed_test_seq30_seed42.npy  gan_imputed_test_seq90_seed42.npy
  training_history_seq30.csv     training_history_seq90.csv

NOTE: Legacy defaults (gan_model_seed42.pt, training_history.csv, etc.)
      are NOT written during the ablation run.
"""
import sys, io, os, time, random, pickle
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from precip_calibration import PrecipCalibrator

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ────────────────────────────────────────────────────────────
MODE        = 'B'    # 'A' = temporal only | 'B' = dual-branch spatio-temporal
# SEQ_LEN is set per-ablation loop (30 or 90); no global default needed.
MISS_RATE   = 0.10   # training scenario
BATCH_SIZE  = 128
N_EPOCHS    = 60
N_CRITIC    = 3
LAMBDA_GP   = 10
LAMBDA_RECON = 10
LR          = 1e-4
HIDDEN_SIZE = 64
N_LAYERS    = 2
DROPOUT     = 0.2
PATIENCE    = 10
SEED        = 42
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class MeteoDataset(Dataset):
    def __init__(self, data, real_mask, corrupted, art_mask, temporal,
                 neighbor_avg, neighbor_mask, seq_len, mode='A'):
        self.data      = torch.tensor(np.nan_to_num(data,         nan=0.0), dtype=torch.float32)
        self.gt_mask   = torch.tensor(real_mask,                             dtype=torch.float32)
        self.corrupt   = torch.tensor(np.nan_to_num(corrupted,    nan=0.0), dtype=torch.float32)
        self.art_mask  = torch.tensor(art_mask,                              dtype=torch.float32)
        self.temporal  = torch.tensor(temporal,                              dtype=torch.float32)
        self.nbr_avg   = torch.tensor(np.nan_to_num(neighbor_avg, nan=0.0), dtype=torch.float32)
        self.nbr_mask  = torch.tensor(neighbor_mask,                         dtype=torch.float32)
        self.seq_len   = seq_len
        self.mode      = mode
        self.starts    = list(range(0, len(data) - seq_len, seq_len))   # non-overlapping

    def __len__(self): return len(self.starts)

    def __getitem__(self, i):
        s  = self.starts[i]
        sl = slice(s, s + self.seq_len)
        return {
            'data'    : self.data[sl],
            'gt_mask' : self.gt_mask[sl],
            'corrupt' : self.corrupt[sl],
            'art_mask': self.art_mask[sl],
            'temporal': self.temporal[sl],
            'nbr_avg' : self.nbr_avg[sl],
            'nbr_mask': self.nbr_mask[sl],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """Mode A: single-branch Bidirectional LSTM generator.
    Input  = [corrupted | comb_mask | temporal]
    Output = imputed_meteo (Sigmoid — MinMax normalised space)
    """
    def __init__(self, in_dim, n_meteo, hidden=HIDDEN_SIZE,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0,
                            bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_meteo), nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)


class GeneratorB(nn.Module):
    """Mode B: dual-branch hybrid spatio-temporal generator.

    Temporal branch : BiLSTM on [corrupted | comb_mask | temporal]
                      → hidden representation per time-step (hidden*2 dims)

    Spatial branch  : small feedforward encoder on [nbr_avg | nbr_mask]
                      → spatial embedding per time-step (hidden//2 dims)

    Fusion          : concatenate both → Linear → Sigmoid → imputed_meteo
    """
    def __init__(self, n_meteo, n_temporal, hidden=HIDDEN_SIZE,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        # ----- Temporal branch -----
        temp_in = n_meteo * 2 + n_temporal          # corrupted + comb_mask + temporal
        self.temporal_lstm = nn.LSTM(
            temp_in, hidden, n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True
        )
        # ----- Spatial branch -----
        spat_in  = n_meteo * 2                       # nbr_avg + nbr_mask
        spat_hid = hidden // 2
        self.spatial_enc = nn.Sequential(
            nn.Linear(spat_in, spat_hid), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spat_hid, spat_hid), nn.ReLU(),
        )
        # ----- Fusion -----
        fuse_in = hidden * 2 + spat_hid              # biLSTM out + spatial enc out
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_meteo), nn.Sigmoid()
        )

    def forward(self, x_temp, x_spat):
        """x_temp: (B, T, n_meteo*2+n_temporal)  x_spat: (B, T, n_meteo*2)"""
        temp_out, _ = self.temporal_lstm(x_temp)     # (B, T, hidden*2)
        spat_out    = self.spatial_enc(x_spat)        # (B, T, spat_hid)
        fused       = torch.cat([temp_out, spat_out], dim=-1)
        return self.fusion(fused)                     # (B, T, n_meteo)


class Discriminator(nn.Module):
    """Unidirectional LSTM discriminator (Wasserstein — no sigmoid)."""
    def __init__(self, in_dim, hidden=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────
def gradient_penalty(D, real, fake, device):
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device).expand_as(real)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out  = D(interp)
    grad   = torch.autograd.grad(
        d_out, interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    return ((grad.norm(2, dim=(1, 2)) - 1) ** 2).mean()


def build_combined_mask(batch):
    """Combined mask: real missing OR artificially removed."""
    return (1 - batch['gt_mask'] + batch['art_mask']).clamp(0, 1)


def build_gen_input(batch, mode='A'):
    """Generator input tensor(s).
    Mode A → single tensor: [corrupt | comb_mask | temporal]
    Mode B → tuple (x_temp, x_spat) for dual-branch GeneratorB.
    """
    comb = build_combined_mask(batch)
    x_temp = torch.cat([batch['corrupt'], comb, batch['temporal']], dim=-1)
    if mode == 'B':
        x_spat = torch.cat([batch['nbr_avg'], batch['nbr_mask']], dim=-1)
        return x_temp, x_spat
    return x_temp


def build_disc_input(meteo, comb_mask, temporal):
    """Discriminator input: meteo + comb_mask + temporal."""
    return torch.cat([meteo, comb_mask, temporal], dim=-1)


@torch.no_grad()
def val_rmse(G, loader, device, mode='A'):
    G.eval()
    sq, n = 0.0, 0
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        gen_in = build_gen_input(batch, mode)
        pred   = G(*gen_in) if isinstance(gen_in, tuple) else G(gen_in)
        am     = batch['art_mask']
        if am.sum() > 0:
            sq += ((pred - batch['data']) ** 2 * am).sum().item()
            n  += am.sum().item()
    return (sq / n) ** 0.5 if n > 0 else float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_model(G, D, train_loader, val_loader, mode=MODE,
                n_epochs=N_EPOCHS, patience=PATIENCE):
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))

    best_rmse, patience_cnt, best_G, best_D, best_ep = float('inf'), 0, None, None, 0
    history = []
    t0      = time.time()

    for epoch in range(1, n_epochs + 1):
        G.train(); D.train()
        d_losses, g_losses, recon_losses = [], [], []

        for batch in train_loader:
            batch     = {k: v.to(DEVICE) for k, v in batch.items()}
            gen_in    = build_gen_input(batch, mode)
            comb_mask = build_combined_mask(batch)

            # Discriminator
            for _ in range(N_CRITIC):
                opt_D.zero_grad()
                with torch.no_grad():
                    fake = G(*gen_in) if isinstance(gen_in, tuple) else G(gen_in)
                real_d = build_disc_input(batch['data'],  comb_mask, batch['temporal'])
                fake_d = build_disc_input(fake.detach(), comb_mask, batch['temporal'])
                gp     = gradient_penalty(D, real_d, fake_d, DEVICE)
                d_loss = D(fake_d).mean() - D(real_d).mean() + LAMBDA_GP * gp
                d_loss.backward(); opt_D.step()
            d_losses.append(d_loss.item())

            # Generator
            opt_G.zero_grad()
            fake   = G(*gen_in) if isinstance(gen_in, tuple) else G(gen_in)
            fake_d = build_disc_input(fake, comb_mask, batch['temporal'])
            g_adv  = -D(fake_d).mean()
            am     = batch['art_mask']
            recon  = ((fake - batch['data']) ** 2 * am).sum() / am.sum().clamp(min=1)
            g_loss = g_adv + LAMBDA_RECON * recon
            g_loss.backward(); opt_G.step()
            g_losses.append(g_adv.item()); recon_losses.append(recon.item())

        vr = val_rmse(G, val_loader, DEVICE, mode)
        history.append({
            'epoch': epoch, 'val_rmse': vr,
            'd_loss': np.mean(d_losses), 'g_loss': np.mean(g_losses),
            'recon_loss': np.mean(recon_losses)
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{n_epochs} | D={np.mean(d_losses):+.4f} | "
                  f"G={np.mean(g_losses):+.4f} | Recon={np.mean(recon_losses):.4f} | "
                  f"ValRMSE={vr:.4f} | {time.time()-t0:.0f}s")
            sys.stdout.flush()

        if vr < best_rmse - 1e-5:
            best_rmse = vr
            best_G = {k: v.cpu().clone() for k, v in G.state_dict().items()}
            best_D = {k: v.cpu().clone() for k, v in D.state_dict().items()}
            best_ep = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop @ epoch {epoch}. Best ValRMSE={best_rmse:.4f} (ep {best_ep})")
                sys.stdout.flush()
                break

    return best_G, best_D, best_ep, best_rmse, pd.DataFrame(history)


# ─────────────────────────────────────────────────────────────────────────────
# Inference (sliding window)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def impute(G, data_np, real_mask_np, temporal_np,
           nbr_avg_np, nbr_mask_np, n_meteo, seq_len=30, mode=MODE):
    """Sliding-window inference.
    Mode A: calls G(x_temp)           — single concatenated input.
    Mode B: calls G(x_temp, x_spat)   — dual-branch GeneratorB.
    """
    G.eval().to(DEVICE)
    N      = data_np.shape[0]
    output = np.zeros((N, n_meteo), dtype=np.float32)
    counts = np.zeros((N, n_meteo), dtype=np.float32)
    data_z = np.nan_to_num(data_np, nan=0.0).astype(np.float32)
    comb   = (1 - real_mask_np).astype(np.float32)
    nbr_z  = np.nan_to_num(nbr_avg_np, nan=0.0).astype(np.float32)
    step   = max(1, seq_len // 2)

    def _forward(sl):
        x_temp = torch.cat([
            torch.tensor(data_z[sl][None],          dtype=torch.float32),
            torch.tensor(comb[sl][None],             dtype=torch.float32),
            torch.tensor(temporal_np[sl][None],      dtype=torch.float32),
        ], dim=-1).to(DEVICE)
        if mode == 'B':
            x_spat = torch.cat([
                torch.tensor(nbr_z[sl][None],        dtype=torch.float32),
                torch.tensor(nbr_mask_np[sl][None],  dtype=torch.float32),
            ], dim=-1).to(DEVICE)
            return G(x_temp, x_spat).squeeze(0).cpu().numpy()
        return G(x_temp).squeeze(0).cpu().numpy()

    for start in range(0, N - seq_len + 1, step):
        sl = slice(start, start + seq_len)
        output[sl] += _forward(sl)
        counts[sl] += 1

    if N >= seq_len:
        sl = slice(N - seq_len, N)
        output[sl] += _forward(sl)
        counts[sl] += 1

    counts = np.where(counts == 0, 1, counts)
    return output / counts


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def load_npz(name, miss_key, amask_key):
    z = np.load(os.path.join(OUTPUT_DIR, f'preprocessed_{name}.npz'), allow_pickle=True)
    nbr_avg  = z['neighbor_avg'].astype(np.float32)  if 'neighbor_avg'  in z.files else np.zeros_like(z['data'], dtype=np.float32)
    nbr_mask = z['neighbor_mask'].astype(np.float32) if 'neighbor_mask' in z.files else np.zeros_like(z['data'], dtype=np.float32)
    return (z['data'].astype(np.float32),
            z['real_mask'].astype(np.float32),
            z[miss_key].astype(np.float32),
            z[amask_key].astype(np.float32),
            z['temporal'].astype(np.float32),
            nbr_avg, nbr_mask)


def run_one(seq_len, seed):
    """
    Run one WGAN-GP experiment for the given seq_len and seed.
    Returns (best_epoch, best_val_rmse).
    """

    print("=" * 62)
    print(f"  WGAN-GP ABLATION  — Mode {MODE} | SEQ_LEN={seq_len} | Seed={seed}")
    print(f"  Device: {DEVICE}")
    print(f"  BATCH={BATCH_SIZE}  EPOCHS={N_EPOCHS}  N_CRITIC={N_CRITIC}  PATIENCE={PATIENCE}")
    print(f"  Run tag: seed{seed}")
    print("=" * 62)
    sys.stdout.flush()

    set_seed(seed)

    miss_key  = f'corrupted_{int(MISS_RATE*100):02d}pct'
    amask_key = f'art_mask_{int(MISS_RATE*100):02d}pct'

    tr_d, tr_m, tr_c, tr_a, tr_t, tr_na, tr_nm = load_npz('train', miss_key, amask_key)
    va_d, va_m, va_c, va_a, va_t, va_na, va_nm = load_npz('val',   miss_key, amask_key)
    te_d, te_m, te_c, te_a, te_t, te_na, te_nm = load_npz('test',  miss_key, amask_key)

    N_METEO    = tr_d.shape[1]
    N_TEMPORAL = tr_t.shape[1]

    # Input dims
    IN_DIM_G_TEMP = N_METEO * 2 + N_TEMPORAL     # temporal branch: corrupt+comb+temporal
    IN_DIM_G_SPAT = N_METEO * 2                  # spatial branch:  nbr_avg+nbr_mask (B only)
    IN_DIM_D      = N_METEO * 2 + N_TEMPORAL     # Discriminator: always same

    print(f"  Data  Train{tr_d.shape}  Val{va_d.shape}  Test{te_d.shape}")
    print(f"  N_METEO={N_METEO}  N_TEMPORAL={N_TEMPORAL}")
    print(f"  Mode={MODE}  IN_DIM_G_TEMP={IN_DIM_G_TEMP}  IN_DIM_G_SPAT={IN_DIM_G_SPAT}  IN_DIM_D={IN_DIM_D}")
    sys.stdout.flush()

    train_ds = MeteoDataset(tr_d, tr_m, tr_c, tr_a, tr_t, tr_na, tr_nm, seq_len, MODE)
    val_ds   = MeteoDataset(va_d, va_m, va_c, va_a, va_t, va_na, va_nm, seq_len, MODE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train seqs: {len(train_ds):,}  Val seqs: {len(val_ds):,}")
    sys.stdout.flush()

    if MODE == 'B':
        G = GeneratorB(n_meteo=N_METEO, n_temporal=N_TEMPORAL).to(DEVICE)
    else:
        G = Generator(in_dim=IN_DIM_G_TEMP, n_meteo=N_METEO).to(DEVICE)
    D = Discriminator(in_dim=IN_DIM_D).to(DEVICE)

    G_state, D_state, best_ep, best_rmse, history = train_model(
        G, D, train_loader, val_loader, mode=MODE
    )

    # ── Output file prefix (mode-aware to avoid overwriting Mode A) ──────────
    mode_tag  = f'modeB_seed{seed}' if MODE == 'B' else f'seed{seed}'

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt = {
        'generator'    : G_state,
        'discriminator': D_state,
        'epoch'        : best_ep,
        'val_rmse'     : best_rmse,
        'seed'         : seed,
        'mode'         : MODE,
        'in_dim_g_temp': IN_DIM_G_TEMP,
        'in_dim_g_spat': IN_DIM_G_SPAT,
        'in_dim_d'     : IN_DIM_D,
        'n_meteo'      : N_METEO,
        'n_temporal'   : N_TEMPORAL,
        'config'       : {'SEQ_LEN': seq_len, 'HIDDEN_SIZE': HIDDEN_SIZE,
                          'N_LAYERS': N_LAYERS, 'N_METEO': N_METEO, 'MODE': MODE}
    }
    ckpt_path = os.path.join(OUTPUT_DIR, f'gan_model_{mode_tag}.pt')
    torch.save(ckpt, ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}")

    hist_path = os.path.join(OUTPUT_DIR, f'training_history_{mode_tag}.csv')
    history.to_csv(hist_path, index=False)
    print(f"  Training history  → {hist_path}")

    # ── Test imputation ───────────────────────────────────────────────────────
    print("\n  Running test imputation ...")
    G.load_state_dict(G_state)
    imp = impute(G, te_d, te_m, te_t, te_na, te_nm, N_METEO, seq_len, MODE)
    imp_path = os.path.join(OUTPUT_DIR, f'gan_imputed_test_{mode_tag}.npy')
    np.save(imp_path, imp)
    print(f"  Test imputation   → {imp_path}  (shape={imp.shape})")

    # ── Precipitation calibration (fit on val, apply to test) ─────────────────
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        sc_fit  = scaler_data['scaler']
        mvars   = scaler_data['meteo_vars']

        if 'PRECIP' in mvars:
            precip_idx = list(mvars).index('PRECIP')

            # Build val imputation (needed to fit calibrator)
            print("  Running val imputation for calibration fit ...")
            val_imp = impute(G, va_d, va_m, va_t, va_na, va_nm, N_METEO, seq_len, MODE)

            calibrator = PrecipCalibrator(
                precip_idx=precip_idx,
                scaler=sc_fit,
                use_quantile_mapping=True,
            )
            calibrator.fit_threshold(
                val_imp_norm=val_imp,
                val_gt_norm=va_d,
                val_art_mask=va_a,
                verbose=True,
            )

            # Apply calibration to ALL cells of the raw GAN output.
            # art_mask=None is intentional: the GAN's PRECIP output is
            # unrealistically wet everywhere (not just at masked positions),
            # so we calibrate the full array.  Observed (non-masked) positions
            # are untouched by the scaler logic that follows in evaluation.
            imp_cal = calibrator.apply(imp, art_mask=None)
            cal_path = os.path.join(OUTPUT_DIR,
                                    f'gan_imputed_test_{mode_tag}_precipfix.npy')
            np.save(cal_path, imp_cal)
            print(f"  Calibrated output → {cal_path}")

            json_path = os.path.join(OUTPUT_DIR,
                                     f'precip_calibration_{mode_tag}.json')
            calibrator.save(json_path)

    # ── Metrics in original units (via scaler) ────────────────────────────────
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        sc    = scaler_data['scaler']
        mvars = scaler_data['meteo_vars']

        imp_orig  = sc.inverse_transform(np.clip(imp, 0, 1))
        te_d_orig = sc.inverse_transform(np.nan_to_num(te_d, nan=0.0))

        te_npz = np.load(os.path.join(OUTPUT_DIR, 'preprocessed_test.npz'), allow_pickle=True)
        print()
        for scenario_key in ['art_mask_10pct', 'art_mask_20pct',
                             'art_mask_block7d', 'art_mask_block30d']:
            if scenario_key not in te_npz.files:
                continue
            am_np = te_npz[scenario_key].astype(np.float32)
            print(f"  Test metrics (original units) — {scenario_key}  [Mode {MODE}  SEQ={seq_len}]:")
            rmse_list, mae_list = [], []
            for i, v in enumerate(mvars):
                sel = am_np[:, i].astype(bool)
                if sel.sum() == 0:
                    continue
                r = float(np.sqrt(np.mean((imp_orig[sel, i] - te_d_orig[sel, i]) ** 2)))
                m = float(np.mean(np.abs(imp_orig[sel, i] - te_d_orig[sel, i])))
                rmse_list.append(r); mae_list.append(m)
                print(f"    {v:12s}: RMSE={r:.3f}  MAE={m:.3f}")
            if rmse_list:
                print(f"    {'MEAN (macro)':12s}: RMSE={np.mean(rmse_list):.3f}  MAE={np.mean(mae_list):.3f}")
            print()

        # Physical consistency
        try:
            ti, tm, tx = mvars.index('TMIN'), mvars.index('TMEAN'), mvars.index('TMAX')
            viol = int(((imp_orig[:, ti] > imp_orig[:, tm]) |
                        (imp_orig[:, tm] > imp_orig[:, tx])).sum())
            pct  = 100.0 * viol / len(imp_orig)
            print(f"  Physical check (raw GAN output): TMIN<=TMEAN<=TMAX violations = {viol} ({pct:.3f}%)")
            gt_raw = sc.inverse_transform(np.nan_to_num(te_d, nan=0.0))
            gt_viol = int(((gt_raw[:, ti] > gt_raw[:, tm]) |
                           (gt_raw[:, tm] > gt_raw[:, tx])).sum())
            gt_pct  = 100.0 * gt_viol / len(gt_raw)
            print(f"  Physical check (ground truth   ): TMIN<=TMEAN<=TMAX violations = {gt_viol} ({gt_pct:.3f}%)")
        except (ValueError, IndexError):
            pass

    print("\n" + "=" * 62)
    print(f"  DONE  Mode={MODE}  SEQ_LEN={seq_len}  Seed={seed}")
    print(f"  best_epoch={best_ep}  val_rmse={best_rmse:.4f}")
    print("=" * 62)
    sys.stdout.flush()

    return best_ep, best_rmse


def main():
    """Run the appropriate experiment based on MODE.

    MODE = 'A'  → seed robustness: SEQ_LEN=30, seeds=[42, 123, 456]
                  saves: gan_model_seed{s}.pt / training_history_seed{s}.csv
                         gan_imputed_test_seed{s}.npy

    MODE = 'B'  → pilot experiment: SEQ_LEN=30, SEED=42
                  saves: gan_model_modeB_seed42.pt
                         training_history_modeB_seed42.csv
                         gan_imputed_test_modeB_seed42.npy
    """
    if MODE == 'B':
        # ── Mode B robustness run — 3 seeds ──────────────────────────────────
        seq_len = 30
        seeds   = [42, 123, 456]
        results = []
        print("=" * 62)
        print("  MODE B ROBUSTNESS RUN — SEQ_LEN=30  Seeds=[42, 123, 456]")
        print("  Dual-branch spatio-temporal WGAN-GP")
        print("  Scenarios: 10pct | 20pct | block7d | block30d")
        print("=" * 62)
        for s in seeds:
            best_ep, best_rmse = run_one(seq_len, s)
            results.append({'seed': s, 'best_epoch': best_ep, 'best_val_rmse': best_rmse})

        print("\n" + "=" * 62)
        print("  SEED ROBUSTNESS SUMMARY  — Mode B, SEQ_LEN=30")
        print("=" * 62)
        print(f"  {'Seed':>6}  {'Best Epoch':>10}  {'Best Val RMSE':>13}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*13}")
        for r in results:
            print(f"  {r['seed']:>6}  {r['best_epoch']:>10}  {r['best_val_rmse']:>13.4f}")
        print("=" * 62)
        sys.stdout.flush()
    else:
        # ── Mode A seed robustness ────────────────────────────────────────────
        seq_len = 30
        seeds   = [42, 123, 456]
        results = []
        for s in seeds:
            best_ep, best_rmse = run_one(seq_len, s)
            results.append({'seed': s, 'best_epoch': best_ep, 'best_val_rmse': best_rmse})

        print("\n" + "=" * 62)
        print("  SEED ROBUSTNESS SUMMARY  — Mode A, SEQ_LEN=30")
        print("=" * 62)
        print(f"  {'Seed':>6}  {'BestEpoch':>10}  {'ValRMSE':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
        for r in results:
            print(f"  {r['seed']:>6}  {r['best_epoch']:>10}  {r['best_val_rmse']:>10.4f}")
        print("=" * 62)
        sys.stdout.flush()


if __name__ == '__main__':
    main()

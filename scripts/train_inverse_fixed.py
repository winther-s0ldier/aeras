"""
train_inverse_fixed.py
──────────────────────
Fixed SourceNet training. Three bugs from the original inverse run are corrected:

  Bug 1 — Forward network was UNFROZEN.
    The optimizer used Dx/Dy/lambda to absorb unmodelled physics instead of S.
    Fix: freeze all forward model parameters before training SourceNet.

  Bug 2 — L1 sparsity penalty punished S > 0.
    Any non-zero source incurred a penalty, so the optimizer preferred S = 0.
    Fix: remove sparsity loss entirely.

  Bug 3 — SourceNet output only 1 channel.
    Only PM2.5 got a source term. NO2/O3/SO2 were silently set to S = 0.
    Fix: output OUTPUT_DIM channels (one per pollutant).

  Bonus — Positive bias on final linear layer.
    Breaks the zero-plateau: network starts at S ~ 0.05 so gradients have
    direction to push from the very first step.

Run on GPU 0:
  CUDA_VISIBLE_DEVICES=0 python train_inverse_fixed.py
"""

import os, sys, json, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import src.config as cfg
from src.config import DEVICE, CHECKPOINTS_DIR, OUTPUT_DIM
from src.models.pinn import AerasPINN, compute_pde_residual
from src.data.collocation import latin_hypercube

# ── hyper-parameters ──────────────────────────────────────────────────────────
FORWARD_CKPT  = CHECKPOINTS_DIR / "aeras_v11_inverse_final.pt"
OUTPUT_PREFIX = "aeras_v12_fixed_inverse"
EPOCHS        = 15_000
LR            = 5e-4
BATCH_COLLOC  = 8_192
NUM_COLLOC    = 150_000
PRINT_EVERY   = 500
SAVE_EVERY    = 5_000

# Kaggle dataset path (same as notebook cells)
SPLITS_DIR = Path("/kaggle/input/datasets/rudrakumar21/aeras-splits")

INPUT_COLS  = ["x_norm","y_norm","t_norm","u_wind_norm","v_wind_norm","temp_norm","blh_norm"]
TARGET_COLS = ["pm25_norm","no2_norm","o3_norm","so2_norm"]

# Normalised timestamp of Diwali 2019-10-27 00:00 UTC
# t_min=1514764800 (2018-01-01), t_max=1672527600 (2022-12-31)
_T_MIN, _T_MAX = 1514764800.0, 1672527600.0
T_DIWALI_NORM  = (1572134400.0 - _T_MIN) / (_T_MAX - _T_MIN)   # ≈ 0.363
DIWALI_HALF_WIN = 5.0 / ((_T_MAX - _T_MIN) / 86400.0)          # ±5 days in norm units


# ── SourceNet (fixed) ─────────────────────────────────────────────────────────
class FixedSourceNet(nn.Module):
    """
    Maps (x, y, t) → S[OUTPUT_DIM] with S >= 0 guaranteed via Softplus.

    Key changes vs original SourceNet:
    - OUTPUT_DIM outputs (not 1) so all pollutants get a source term.
    - Positive bias (+0.5) on the final linear layer breaks the zero-plateau
      that caused the original SourceNet to collapse.
    - Larger network (128 hidden, 5 layers) to capture spatial structure.
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 5):
        super().__init__()
        layers, in_dim = [], 3
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Linear(hidden_dim, OUTPUT_DIM)
        nn.init.xavier_normal_(self.head.weight, gain=0.1)
        nn.init.constant_(self.head.bias, 0.5)   # positive bias — breaks zero plateau

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(self.trunk(xyt)))

    def get_source_map(
        self, t_value: float, pollutant_idx: int = 0,
        resolution: int = 50, device: str = "cpu"
    ) -> np.ndarray:
        """Return a 2-D (resolution x resolution) emission map for one pollutant."""
        x  = torch.linspace(0, 1, resolution)
        y  = torch.linspace(0, 1, resolution)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        pts = torch.stack([xx.flatten(), yy.flatten(),
                           torch.full((resolution**2,), t_value)], dim=1).to(device)
        with torch.no_grad():
            S = self(pts)
        return S[:, pollutant_idx].reshape(resolution, resolution).cpu().numpy()


# ── helpers ───────────────────────────────────────────────────────────────────
def build_colloc_with_wind(
    train_inputs: torch.Tensor, xyt_np: np.ndarray, device: str
) -> torch.Tensor:
    """Attach nearest-neighbour wind features to (x,y,t) collocation points."""
    xyt    = torch.tensor(xyt_np, dtype=torch.float32, device=device)
    t_col  = xyt[:, 2]
    t_train, order = train_inputs[:, 2].sort()
    pos    = torch.searchsorted(t_train.contiguous(), t_col.contiguous()).clamp(0, len(t_train)-1)
    lpos   = (pos - 1).clamp(0)
    near   = torch.where((t_col - t_train[lpos]).abs() < (t_col - t_train[pos]).abs(), lpos, pos)
    wind   = train_inputs[order[near], 3:]
    return torch.cat([xyt, wind], dim=1)   # [N, 7]


def main():
    device = str(torch.device(DEVICE))
    print(f"[FIXED-INVERSE] Device : {device}")
    print(f"[FIXED-INVERSE] Forward ckpt : {FORWARD_CKPT}")

    # ── load training data (wind interpolation) ────────────────────────────────
    print("[FIXED-INVERSE] Loading training data for wind features...")
    train_df     = pd.read_parquet(SPLITS_DIR / "train.parquet")
    train_inputs = torch.tensor(
        train_df[INPUT_COLS].fillna(0).values, dtype=torch.float32
    ).to(device)

    # ── load and freeze forward model ──────────────────────────────────────────
    if not FORWARD_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {FORWARD_CKPT}")
    state         = torch.load(FORWARD_CKPT, map_location=device, weights_only=False)
    forward_model = AerasPINN(inverse_mode=False).to(device)
    forward_model.load_state_dict(state["model"])
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False   # ← THE KEY FIX: Option B is now impossible
    print("[FIXED-INVERSE] Forward model FROZEN. Dx/Dy/lambda cannot absorb S.")

    # ── fresh SourceNet ────────────────────────────────────────────────────────
    source_net = FixedSourceNet(hidden_dim=128, num_layers=5).to(device)
    n_params   = sum(p.numel() for p in source_net.parameters())
    print(f"[FIXED-INVERSE] SourceNet params: {n_params:,}")

    optimizer = torch.optim.Adam(source_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5000, T_mult=2, eta_min=1e-6
    )

    # ── collocation points: standard LHS + Diwali-dense ───────────────────────
    print(f"[FIXED-INVERSE] Building {NUM_COLLOC:,} collocation points...")
    lhs_std = latin_hypercube(NUM_COLLOC, dim=3, seed=42)
    colloc  = build_colloc_with_wind(train_inputs, lhs_std, device)

    # Extra Diwali-dense points: sample t in [T_DIWALI ± 5 days]
    n_diwali  = NUM_COLLOC // 5
    lhs_d     = latin_hypercube(n_diwali, dim=3, seed=99)
    lhs_d[:, 2] = np.clip(
        T_DIWALI_NORM + (lhs_d[:, 2] - 0.5) * 2 * DIWALI_HALF_WIN, 0.0, 1.0
    )
    colloc_diwali = build_colloc_with_wind(train_inputs, lhs_d, device)
    colloc        = torch.cat([colloc, colloc_diwali], dim=0)
    print(f"[FIXED-INVERSE] Total collocation: {len(colloc):,} "
          f"(incl. {n_diwali:,} Diwali-dense at t≈{T_DIWALI_NORM:.3f})")

    # ── training loop ──────────────────────────────────────────────────────────
    print(f"\n[FIXED-INVERSE] Training SourceNet for {EPOCHS} epochs.")
    print("  Loss = PDE residual only. No sparsity. Forward model frozen.\n")

    best_loss = float("inf")
    history   = []
    t0        = time.time()

    for epoch in range(1, EPOCHS + 1):
        source_net.train()
        optimizer.zero_grad()

        idx           = torch.randint(0, len(colloc), (BATCH_COLLOC,), device=device)
        colloc_batch  = colloc[idx].float()
        xyt_batch     = colloc_batch[:, :3].clone()   # independent tensor — no shared storage

        # SourceNet output: S[B, OUTPUT_DIM], all >= 0 by Softplus
        S = source_net(xyt_batch)

        # PDE residual with frozen forward model.
        # compute_pde_residual sets requires_grad=True on colloc_batch internally
        # so dC/dt, dC/dx etc. are computed; forward weights frozen so they don't
        # accumulate grads. Only S (and through it source_net) gets grads.
        with torch.enable_grad():
            residual = compute_pde_residual(forward_model, colloc_batch, S)

        loss = (residual ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(source_net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % PRINT_EVERY == 0:
            s_mean = S.detach().mean().item()
            s_max  = S.detach().max().item()
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:6d}/{EPOCHS} | "
                f"PDE loss: {loss.item():.4e} | "
                f"S mean: {s_mean:.4e}  max: {s_max:.4e} | "
                f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed/60:.1f} min"
            )
            entry = {"epoch": epoch, "pde_loss": loss.item(),
                     "S_mean": s_mean, "S_max": s_max}
            history.append(entry)

            if loss.item() < best_loss:
                best_loss = loss.item()
                _save_ckpt(forward_model, source_net, optimizer, epoch, history,
                           CHECKPOINTS_DIR / f"{OUTPUT_PREFIX}_best.pt")

        if epoch % SAVE_EVERY == 0:
            _save_ckpt(forward_model, source_net, optimizer, epoch, history,
                       CHECKPOINTS_DIR / f"{OUTPUT_PREFIX}_epoch{epoch}.pt")

    # ── final save ─────────────────────────────────────────────────────────────
    final_path = CHECKPOINTS_DIR / f"{OUTPUT_PREFIX}_final.pt"
    _save_ckpt(forward_model, source_net, optimizer, EPOCHS, history, final_path)

    # ── source maps ────────────────────────────────────────────────────────────
    print("\n[FIXED-INVERSE] Generating source maps (PM2.5 channel)...")
    source_net.eval()
    maps = {}
    for t_norm, label in [
        (0.25,          "t_0.25"),
        (0.50,          "t_0.50"),
        (0.75,          "t_0.75"),
        (T_DIWALI_NORM, "t_diwali"),
    ]:
        m = source_net.get_source_map(t_norm, pollutant_idx=0,
                                      resolution=50, device=device)
        maps[label] = m
        print(f"  {label}: mean={m.mean():.4e}  max={m.max():.4e}  "
              f"(non-zero: {(m > 1e-4).mean()*100:.1f}%)")

    edgar_path = CHECKPOINTS_DIR / "edgar_delhi_pm25.npy"
    if edgar_path.exists():
        maps["edgar"] = np.load(edgar_path)
        print(f"  edgar: shape={maps['edgar'].shape}")
    else:
        print("  [WARN] edgar_delhi_pm25.npy not found — skipping EDGAR baseline")

    # Overwrite source_maps.npz so demo.py picks up the improved maps
    np.savez(CHECKPOINTS_DIR / "source_maps.npz", **maps)
    np.savez(CHECKPOINTS_DIR / "source_maps_fixed.npz", **maps)
    print(f"[FIXED-INVERSE] Maps saved to checkpoints/source_maps.npz")

    # ── evaluation ─────────────────────────────────────────────────────────────
    _evaluate(forward_model, source_net, device)
    _compare_edgar(source_net, maps, device)

    print(f"\n[FIXED-INVERSE] Done.  Best PDE loss: {best_loss:.4e}")
    print(f"[FIXED-INVERSE] Checkpoint: {final_path}")


# ── save helper ───────────────────────────────────────────────────────────────
def _save_ckpt(forward_model, source_net, optimizer, epoch, history, path):
    torch.save({
        "model":        forward_model.state_dict(),
        "source_model": source_net.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "epoch":        epoch,
        "history":      history,
    }, path)
    print(f"  [CKPT] Saved → {path.name}")


# ── forward + inverse evaluation ──────────────────────────────────────────────
def _evaluate(forward_model, source_net, device):
    print("\n[EVAL] Evaluating on Diwali 2019 test set...")
    df      = pd.read_parquet(SPLITS_DIR / "test_diwali.parquet")
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values,  dtype=torch.float32).to(device)
    targets = torch.tensor(df[TARGET_COLS].fillna(0).values, dtype=torch.float32).to(device)
    mask    = torch.tensor(df[TARGET_COLS].notna().values,   dtype=torch.float32).to(device)

    forward_model.eval()
    source_net.eval()
    results = {}

    with torch.no_grad():
        C_pred = forward_model(inputs)
        S_pred = source_net(inputs[:, :3])

    print(f"\n{'Pollutant':<10} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'S_mean':>12}")
    print("-" * 54)
    for i, name in enumerate(["pm25", "no2", "o3", "so2"]):
        valid = mask[:, i].bool()
        if valid.sum() == 0:
            continue
        p, t = C_pred[valid, i], targets[valid, i]
        mae  = (p - t).abs().mean().item()
        rmse = ((p - t) ** 2).mean().sqrt().item()
        ss_r = ((p - t) ** 2).sum().item()
        ss_t = ((t - t.mean()) ** 2).sum().item()
        r2   = 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")
        sm   = S_pred[valid, i].mean().item()
        print(f"{name.upper():<10} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {sm:>12.4e}")
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "source_mean": sm}

    out = CHECKPOINTS_DIR / "inverse_evaluation_fixed.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EVAL] Saved → {out.name}")


def _compare_edgar(source_net, maps, device):
    """Print a quick comparison between PINN source map and EDGAR at Diwali."""
    if "edgar" not in maps:
        return
    pinn_map  = maps.get("t_diwali", maps.get("t_0.50"))
    edgar_map = maps["edgar"]

    # Resize edgar to match pinn_map resolution
    from scipy.ndimage import zoom
    if edgar_map.shape != pinn_map.shape:
        zf = (pinn_map.shape[0] / edgar_map.shape[0],
              pinn_map.shape[1] / edgar_map.shape[1])
        edgar_map = zoom(edgar_map, zf)

    pinn_peak_idx  = np.unravel_index(pinn_map.argmax(),  pinn_map.shape)
    edgar_peak_idx = np.unravel_index(edgar_map.argmax(), edgar_map.shape)
    dist_cells     = np.sqrt((pinn_peak_idx[0] - edgar_peak_idx[0])**2 +
                             (pinn_peak_idx[1] - edgar_peak_idx[1])**2)
    # 50 cells cover ~1.5° × 1.3° ≈ 165km × 144km in Delhi NCR bbox
    km_per_cell = 165.0 / 50.0
    dist_km     = dist_cells * km_per_cell

    print(f"\n[EDGAR COMPARE] PINN Diwali peak  : cell {pinn_peak_idx}  "
          f"val={pinn_map.max():.4e}")
    print(f"[EDGAR COMPARE] EDGAR peak        : cell {edgar_peak_idx}  "
          f"val={edgar_map.max():.4e}")
    print(f"[EDGAR COMPARE] Peak distance     : {dist_km:.1f} km")
    print("[EDGAR COMPARE] Note: EDGAR = static annual avg (factories/roads). "
          "PINN = dynamic anomaly (Diwali fireworks). Mismatch is EXPECTED.")


if __name__ == "__main__":
    main()

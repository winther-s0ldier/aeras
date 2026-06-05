"""
train_v19_inverse.py
────────────────────
SourceNet trained on frozen v19 (DA4) forward model.

v12 inverse used a frozen v11 base (INPUT_DIM=7, Diwali R²=0.053).
v19 inverse uses frozen v19 base (INPUT_DIM=11, Diwali R²=0.681).
Better forward predictions → better source localization.

Key difference from v12: collocation points now carry all 11 features
(x,y,t + wind + 4 DA lags) to match v19's INPUT_DIM.
SourceNet still maps (x,y,t) → S[4] — only 3 inputs, unchanged.

Run on GPU 0:
  CUDA_VISIBLE_DEVICES=0 python train_v19_inverse.py
"""

import os, sys, json, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import src.config as cfg

# ── must set INPUT_DIM before importing AerasPINN ────────────────────────────
cfg.INPUT_DIM = 11

from src.config import DEVICE, CHECKPOINTS_DIR, OUTPUT_DIM
from src.models.pinn import AerasPINN, compute_pde_residual
from src.data.collocation import latin_hypercube

# ── hyper-parameters ──────────────────────────────────────────────────────────
FORWARD_CKPT  = CHECKPOINTS_DIR / "aeras_v19_da4_final.pt"
OUTPUT_PREFIX = "aeras_v19_inverse"
EPOCHS        = 15_000
LR            = 5e-4
BATCH_COLLOC  = 8_192
NUM_COLLOC    = 150_000
PRINT_EVERY   = 500
SAVE_EVERY    = 5_000

SPLITS_DIR = Path("/kaggle/input/datasets/b1042rudrakumar/aeras-data/splits")

# All 11 features v19 expects
INPUT_COLS = [
    "x_norm", "y_norm", "t_norm",
    "u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm",
    "pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm", "so2_lag1h_norm",
]
TARGET_COLS = ["pm25_norm", "no2_norm", "o3_norm", "so2_norm"]

_T_MIN, _T_MAX  = 1514764800.0, 1672527600.0
T_DIWALI_NORM   = (1572134400.0 - _T_MIN) / (_T_MAX - _T_MIN)   # ≈ 0.363
DIWALI_HALF_WIN = 5.0 / ((_T_MAX - _T_MIN) / 86400.0)


# ── SourceNet (same architecture as v12) ─────────────────────────────────────
class FixedSourceNet(nn.Module):
    """
    Maps (x, y, t) → S[OUTPUT_DIM], S >= 0 via Softplus.
    Architecture unchanged from v12 — appropriate for 40-station constraint.
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
        nn.init.constant_(self.head.bias, 0.5)   # breaks zero plateau

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(self.trunk(xyt)))

    def get_source_map(self, t_value: float, pollutant_idx: int = 0,
                       resolution: int = 50, device: str = "cpu") -> np.ndarray:
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        pts = torch.stack([xx.flatten(), yy.flatten(),
                           torch.full((resolution**2,), t_value)], dim=1).to(device)
        with torch.no_grad():
            S = self(pts)
        return S[:, pollutant_idx].reshape(resolution, resolution).cpu().numpy()


# ── helpers ───────────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "timestamp"]).copy()
    g  = df.groupby("station", observed=True)
    df["pm25_lag1h_norm"] = g["pm25_norm"].shift(1).fillna(0.0)
    df["no2_lag1h_norm"]  = g["no2_norm"].shift(1).fillna(0.0)
    df["o3_lag1h_norm"]   = g["o3_norm"].shift(1).fillna(0.0)
    df["so2_lag1h_norm"]  = g["so2_norm"].shift(1).fillna(0.0)
    return df


def build_colloc_full(train_inputs: torch.Tensor, xyt_np: np.ndarray,
                      device: str) -> torch.Tensor:
    """
    Attach all features (index 3 onwards) from nearest training point.
    Returns [N, 11] — matches v19 INPUT_DIM.
    """
    xyt    = torch.tensor(xyt_np, dtype=torch.float32, device=device)
    t_col  = xyt[:, 2]
    t_train, order = train_inputs[:, 2].sort()
    pos  = torch.searchsorted(t_train.contiguous(), t_col.contiguous()).clamp(0, len(t_train)-1)
    lpos = (pos - 1).clamp(0)
    near = torch.where(
        (t_col - t_train[lpos]).abs() < (t_col - t_train[pos]).abs(), lpos, pos
    )
    all_features = train_inputs[order[near], 3:]   # u,v,T,BLH,pm25_lag,no2_lag,o3_lag,so2_lag
    return torch.cat([xyt, all_features], dim=1)   # [N, 11]


def main():
    device = str(torch.device(DEVICE))
    print(f"[V19-INV] Device      : {device}")
    print(f"[V19-INV] INPUT_DIM   : {cfg.INPUT_DIM}")
    print(f"[V19-INV] Forward ckpt: {FORWARD_CKPT.name}")

    # ── architecture assertion ─────────────────────────────────────────────────
    _probe = AerasPINN()
    assert _probe.fourier.B.shape[0] == 11, (
        f"AerasPINN INPUT_DIM={_probe.fourier.B.shape[0]}, expected 11."
    )
    del _probe

    # ── load training data ─────────────────────────────────────────────────────
    print("[V19-INV] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    train_df = add_lag_features(train_df)
    train_inputs = torch.tensor(
        train_df[INPUT_COLS].fillna(0).values, dtype=torch.float32
    ).to(device)
    print(f"[V19-INV] Training rows: {len(train_inputs):,}")

    # ── load and freeze v19 forward model ─────────────────────────────────────
    if not FORWARD_CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {FORWARD_CKPT}")
    state         = torch.load(FORWARD_CKPT, map_location=device, weights_only=False)
    forward_model = AerasPINN(inverse_mode=False).to(device)
    forward_model.load_state_dict(state["model"])
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False
    print("[V19-INV] v19 forward model FROZEN.")

    # ── fresh SourceNet ────────────────────────────────────────────────────────
    source_net = FixedSourceNet(hidden_dim=128, num_layers=5).to(device)
    n_params   = sum(p.numel() for p in source_net.parameters())
    print(f"[V19-INV] SourceNet params: {n_params:,}")

    optimizer = torch.optim.Adam(source_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5000, T_mult=2, eta_min=1e-6
    )

    # ── collocation points: standard LHS + Diwali-dense ───────────────────────
    print(f"[V19-INV] Building {NUM_COLLOC:,} collocation points...")
    lhs_std = latin_hypercube(NUM_COLLOC, dim=3, seed=42)
    colloc  = build_colloc_full(train_inputs, lhs_std, device)

    n_diwali = NUM_COLLOC // 5
    lhs_d    = latin_hypercube(n_diwali, dim=3, seed=99)
    lhs_d[:, 2] = np.clip(
        T_DIWALI_NORM + (lhs_d[:, 2] - 0.5) * 2 * DIWALI_HALF_WIN, 0.0, 1.0
    )
    colloc_diwali = build_colloc_full(train_inputs, lhs_d, device)
    colloc        = torch.cat([colloc, colloc_diwali], dim=0)
    print(f"[V19-INV] Total collocation: {len(colloc):,} "
          f"(incl. {n_diwali:,} Diwali-dense)")

    # ── training loop ──────────────────────────────────────────────────────────
    print(f"\n[V19-INV] Training SourceNet for {EPOCHS} epochs.")
    print("  Loss = PDE residual only. No sparsity. v19 forward frozen.\n")

    best_loss = float("inf")
    history   = []
    t0        = time.time()

    for epoch in range(1, EPOCHS + 1):
        source_net.train()
        optimizer.zero_grad()

        idx          = torch.randint(0, len(colloc), (BATCH_COLLOC,), device=device)
        colloc_batch = colloc[idx].float()
        xyt_batch    = colloc_batch[:, :3].clone()

        S = source_net(xyt_batch)

        with torch.enable_grad():
            residual = compute_pde_residual(forward_model, colloc_batch, S)

        loss = (residual ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(source_net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % PRINT_EVERY == 0:
            s_mean  = S.detach().mean().item()
            s_max   = S.detach().max().item()
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:6d}/{EPOCHS} | "
                f"PDE loss: {loss.item():.4e} | "
                f"S mean: {s_mean:.4e}  max: {s_max:.4e} | "
                f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed/60:.1f} min"
            )
            history.append({"epoch": epoch, "pde_loss": loss.item(),
                             "S_mean": s_mean, "S_max": s_max})
            if loss.item() < best_loss:
                best_loss = loss.item()
                _save_ckpt(forward_model, source_net, optimizer, epoch, history,
                           CHECKPOINTS_DIR / f"{OUTPUT_PREFIX}_best.pt")

        if epoch % SAVE_EVERY == 0:
            _save_ckpt(forward_model, source_net, optimizer, epoch, history,
                       CHECKPOINTS_DIR / f"{OUTPUT_PREFIX}_epoch{epoch}.pt")

    final_path = CHECKPOINTS_DIR / f"{OUTPUT_PREFIX}_final.pt"
    _save_ckpt(forward_model, source_net, optimizer, EPOCHS, history, final_path)

    # ── source maps ───────────────────────────────────────────────────────────
    print("\n[V19-INV] Generating source maps...")
    source_net.eval()
    maps = {}
    for t_norm, label in [
        (0.25, "t_0.25"), (0.50, "t_0.50"),
        (0.75, "t_0.75"), (T_DIWALI_NORM, "t_diwali"),
    ]:
        m = source_net.get_source_map(t_norm, pollutant_idx=0,
                                      resolution=50, device=device)
        maps[label] = m
        print(f"  {label}: mean={m.mean():.4e}  max={m.max():.4e}  "
              f"non-zero: {(m > 1e-4).mean()*100:.1f}%")

    edgar_path = CHECKPOINTS_DIR / "edgar_delhi_pm25.npy"
    if edgar_path.exists():
        maps["edgar"] = np.load(edgar_path)
    np.savez(CHECKPOINTS_DIR / "source_maps_v19.npz", **maps)
    print("[V19-INV] Source maps saved → source_maps_v19.npz")

    _evaluate(forward_model, source_net, device)
    _compare_edgar(source_net, maps, device)

    print(f"\n[V19-INV] Done. Best PDE loss: {best_loss:.4e}")
    print(f"[V19-INV] Checkpoint: {final_path}")


def _save_ckpt(forward_model, source_net, optimizer, epoch, history, path):
    torch.save({
        "model":        forward_model.state_dict(),
        "source_model": source_net.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "epoch":        epoch,
        "history":      history,
    }, path)
    print(f"  [CKPT] Saved -> {path.name}")


def _evaluate(forward_model, source_net, device):
    print("\n[EVAL] Evaluating on Diwali 2019 test set...")
    df      = pd.read_parquet(SPLITS_DIR / "test_diwali.parquet")
    df      = add_lag_features(df)
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values,  dtype=torch.float32).to(device)
    targets = torch.tensor(df[TARGET_COLS].fillna(0).values, dtype=torch.float32).to(device)
    mask    = torch.tensor(df[TARGET_COLS].notna().values,   dtype=torch.float32).to(device)

    forward_model.eval()
    source_net.eval()
    results = {}

    with torch.no_grad():
        C_pred = forward_model(inputs)
        S_pred = source_net(inputs[:, :3])

    print(f"\n{'Pollutant':<10} {'MAE':>8} {'R2':>8} {'S_mean':>12}")
    print("-" * 44)
    for i, name in enumerate(["pm25", "no2", "o3", "so2"]):
        valid = mask[:, i].bool()
        if valid.sum() == 0:
            continue
        p, t = C_pred[valid, i], targets[valid, i]
        mae  = (p - t).abs().mean().item()
        ss_r = ((p - t)**2).sum().item()
        ss_t = ((t - t.mean())**2).sum().item()
        r2   = 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")
        sm   = S_pred[valid, i].mean().item()
        print(f"{name.upper():<10} {mae:>8.4f} {r2:>8.4f} {sm:>12.4e}")
        results[name] = {"mae": mae, "r2": r2, "source_mean": sm}

    out = CHECKPOINTS_DIR / "v19_inverse_evaluation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EVAL] Saved -> {out.name}")


def _compare_edgar(source_net, maps, device):
    if "edgar" not in maps:
        return
    from scipy.ndimage import zoom
    pinn_map  = maps.get("t_diwali", maps.get("t_0.50"))
    edgar_map = maps["edgar"]
    if edgar_map.shape != pinn_map.shape:
        zf = (pinn_map.shape[0]/edgar_map.shape[0],
              pinn_map.shape[1]/edgar_map.shape[1])
        edgar_map = zoom(edgar_map, zf)
    p_idx = np.unravel_index(pinn_map.argmax(),  pinn_map.shape)
    e_idx = np.unravel_index(edgar_map.argmax(), edgar_map.shape)
    dist  = np.sqrt((p_idx[0]-e_idx[0])**2 + (p_idx[1]-e_idx[1])**2) * (165.0/50.0)
    print(f"\n[EDGAR] PINN peak: cell {p_idx}  val={pinn_map.max():.4e}")
    print(f"[EDGAR] EDGAR peak: cell {e_idx}  val={edgar_map.max():.4e}")
    print(f"[EDGAR] Distance: {dist:.1f} km  (mismatch expected: EDGAR=static, PINN=dynamic Diwali)")


if __name__ == "__main__":
    main()

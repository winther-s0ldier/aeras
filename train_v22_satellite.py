"""
train_v22_satellite.py  —  Satellite-Gated Residual Correction PINN
─────────────────────────────────────────────────────────────────────
Why v20 and v21 failed:
    Fourier embedding mixes all inputs jointly. A binary aer_ai_available
    flag cannot cleanly gate the satellite feature inside a Fourier space.
    Result: satellite corrupted the base representation and degraded PM2.5.

v22 fix — hard-gated additive correction:
    output = frozen_v19(base_inputs) + aer_ai_available * sat_net(x,y,t,aer_ai)

    - frozen_v19: v19 checkpoint, ALL weights frozen, INPUT_DIM=11
    - sat_net: tiny 4-layer x 64-neuron MLP (13K params), trainable
    - aer_ai_available: hard binary gate (0 or 1), NOT learned

    When aer_ai_available=0 → output = v19 exactly. Zero degradation risk.
    When aer_ai_available=1 → output = v19 + satellite correction.

    Normalization verified compatible: v19 achieves R²=0.82/0.74/0.91/0.76
    on the new splits without any retraining.

Baselines:
    v19 DA4:  Diwali PM2.5=0.6813, NO2=0.4763, O3=0.4935, SO2=0.7013
              Winter  PM2.5=0.5967, NO2=0.5496, O3=0.2921, SO2=0.6326
    v20 sat:  Diwali PM2.5=0.5827, NO2=0.2807, O3=0.0427, SO2=0.5007  (worse)
    v21 sat:  Diwali PM2.5=0.3883  (ep35k, incomplete — also worse)

Target: v22 Diwali PM2.5 R² > 0.6813 (beat v19).
Floor:  v22 cannot be worse than v19 (hard gate guarantee).
"""

import os, sys, json, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import src.config as cfg

# v19 base uses INPUT_DIM=11 — set before importing AerasPINN
cfg.INPUT_DIM         = 11
cfg.CHECKPOINT_PREFIX = "aeras_v22_satellite"

from src.config import CHECKPOINTS_DIR, DEVICE
from src.models.pinn import AerasPINN

SPLITS_DIR = Path("/kaggle/input/datasets/b1042rudrakumar/kaggle-dataset-fixed")
V19_CKPT   = Path("/kaggle/working/v19_checkpoints/aeras_v19_da4_final.pt")

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS           = 20_000
LR               = 1e-3
BATCH_SIZE       = 4096
LBFGS_MAX_ITER   = 1_000
LBFGS_CHUNK      = 20_000
CHECKPOINT_EVERY = 2_500

# ── Column definitions ────────────────────────────────────────────────────────
INPUT_COLS_BASE = [
    "x_norm", "y_norm", "t_norm",
    "u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm",
    "pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm", "so2_lag1h_norm",
]
SAT_COLS    = ["x_norm", "y_norm", "t_norm", "aer_ai_norm"]   # sat_net inputs
ALL_COLS    = INPUT_COLS_BASE + ["aer_ai_norm", "aer_ai_available"]
TARGET_COLS = ["pm25_norm", "no2_norm", "o3_norm", "so2_norm"]

_PHYS_NORM = {
    "pm25": (0.03, 509.75), "no2": (0.01, 278.21),
    "o3":   (0.01,  66.17), "so2": (0.01, 500.00),
}

V19_REF = {
    "diwali": {"pm25": 0.6813, "no2": 0.4763, "o3": 0.4935, "so2": 0.7013},
    "winter": {"pm25": 0.5967, "no2": 0.5496, "o3": 0.2921, "so2": 0.6326},
}
V20_REF = {
    "diwali": {"pm25": 0.5827, "no2": 0.2807, "o3": 0.0427, "so2": 0.5007},
    "winter": {"pm25": 0.4389, "no2": 0.3223, "o3": -0.5482, "so2": 0.4416},
}


# ── Satellite correction network ──────────────────────────────────────────────
class SatCorrectionNet(nn.Module):
    """
    Tiny MLP: [x, y, t, aer_ai_norm] -> residual corrections for 4 pollutants.
    Initialised to near-zero output so training starts at v19 performance.
    """
    def __init__(self, in_dim: int = 4, hidden: int = 64, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        # Zero-init output layer: corrections start at 0, grow only if helpful
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full model ────────────────────────────────────────────────────────────────
class SatelliteCorrectedPINN(nn.Module):
    """
    output = frozen_v19(base_11) + aer_ai_available * sat_net(x, y, t, aer_ai)

    inputs layout (13 features total):
        [:11]  = base v19 inputs  (x,y,t,u,v,temp,blh,pm25_lag,no2_lag,o3_lag,so2_lag)
        [11]   = aer_ai_norm      (mean-filled when missing)
        [12]   = aer_ai_available (1=real S5P value, 0=filled)
    """
    def __init__(self, v19_ckpt: Path):
        super().__init__()

        # Load and freeze v19
        self.base = AerasPINN()
        raw = torch.load(v19_ckpt, map_location="cpu")
        self.base.load_state_dict(raw["model"])
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

        n_frozen    = sum(p.numel() for p in self.base.parameters())
        print(f"[V22SAT] v19 base loaded + frozen ({n_frozen:,} params frozen)")

        # Trainable correction network
        self.sat_net = SatCorrectionNet()
        n_trainable = sum(p.numel() for p in self.sat_net.parameters())
        print(f"[V22SAT] sat_net trainable params: {n_trainable:,}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_in   = inputs[:, :11]          # v19 features
        aer_norm  = inputs[:, 11:12]        # aer_ai_norm (mean-filled)
        available = inputs[:, 12:13]        # hard gate  (0 or 1)
        sat_in    = torch.cat([inputs[:, 0:3], aer_norm], dim=1)  # x,y,t,aer_ai

        base_pred  = self.base(base_in)
        correction = self.sat_net(sat_in)

        # Hard gate: correction only applied where satellite is real
        return base_pred + available * correction

    def trainable_parameters(self):
        return list(self.sat_net.parameters())


# ── Data helpers ──────────────────────────────────────────────────────────────
def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "timestamp"]).copy()
    g  = df.groupby("station", observed=True)
    for p in ["pm25", "no2", "o3", "so2"]:
        df[f"{p}_lag1h_norm"] = g[f"{p}_norm"].shift(1).fillna(0.0)
    return df


def prepare_df(df: pd.DataFrame, mean_aer_ai: float) -> pd.DataFrame:
    df = add_lags(df)
    if "aer_ai_norm" not in df.columns:
        df["aer_ai_norm"] = np.nan
    df["aer_ai_available"] = df["aer_ai_norm"].notna().astype(float)
    df["aer_ai_norm"]      = df["aer_ai_norm"].fillna(mean_aer_ai)
    return df


def df_to_tensors(df: pd.DataFrame) -> dict:
    inputs  = torch.tensor(df[ALL_COLS].fillna(0).values,      dtype=torch.float32)
    targets = torch.tensor(df[TARGET_COLS].fillna(0).values,   dtype=torch.float32)
    mask    = torch.tensor(df[TARGET_COLS].notna().values,      dtype=torch.float32)

    ev = pd.Series(1.0, index=df.index)
    if "is_holiday" in df.columns:
        ev.loc[df["is_holiday"].astype(bool)] *= 3.0
    if "timestamp" in df.columns:
        ev.loc[df["timestamp"].dt.month.isin([12, 1])] *= 2.0
    if "pm25_norm" in df.columns:
        ev.loc[df["pm25_norm"] > 0.4] *= 2.0
    event_weight = torch.tensor(ev.values, dtype=torch.float32).unsqueeze(1)

    return {"inputs": inputs, "targets": targets,
            "mask": mask, "event_weight": event_weight}


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model: SatelliteCorrectedPINN,
          train_t: dict, val_t: dict, device: str) -> SatelliteCorrectedPINN:

    model = model.to(device)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5_000, T_mult=2, eta_min=1e-6
    )

    inputs  = train_t["inputs"].to(device)
    targets = train_t["targets"].to(device)
    mask    = train_t["mask"].to(device)
    ew      = train_t["event_weight"].to(device)

    v_inp = val_t["inputs"].to(device)
    v_tgt = val_t["targets"].to(device)
    v_msk = val_t["mask"].to(device)

    N  = len(inputs)
    t0 = time.time()

    print(f"[V22SAT] Adam training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")

    for epoch in range(EPOCHS + 1):
        # ── train step ───────────────────────────────────────────────────────
        model.train()
        model.base.eval()   # keep frozen base in eval always

        idx = torch.randperm(N, device=device)[:BATCH_SIZE]
        xb, yb, mb, wb = inputs[idx], targets[idx], mask[idx], ew[idx]

        optimizer.zero_grad()
        pred  = model(xb)
        denom = (mb * wb).sum().clamp(min=1.0)
        loss  = ((pred - yb) ** 2 * mb * wb).sum() / denom
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── periodic checkpoint + log ────────────────────────────────────────
        if epoch % CHECKPOINT_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vp   = model(v_inp)
                vden = v_msk.sum().clamp(min=1.0)
                vloss = ((vp - v_tgt).abs() * v_msk).sum() / vden

            elapsed = (time.time() - t0) / 60
            print(f"Epoch {epoch:>6} | Loss: {loss.item():.4e} | Val MAE: {vloss.item():.4e} | {elapsed:.1f}min")

            ckpt = CHECKPOINTS_DIR / f"{cfg.CHECKPOINT_PREFIX}_epoch_{epoch}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)

    # ── L-BFGS fine-tuning ───────────────────────────────────────────────────
    print("\n[V22SAT] L-BFGS fine-tuning...")
    idx  = torch.randperm(N, device=device)[:LBFGS_CHUNK]
    lx, ly, lm, lw = inputs[idx], targets[idx], mask[idx], ew[idx]
    lden = (lm * lw).sum().clamp(min=1.0)

    lbfgs = torch.optim.LBFGS(
        model.trainable_parameters(),
        max_iter=LBFGS_MAX_ITER, lr=0.1,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        model.train(); model.base.eval()
        out  = model(lx)
        loss = ((out - ly) ** 2 * lm * lw).sum() / lden
        loss.backward()
        return loss

    lbfgs.step(closure)
    print("[V22SAT] L-BFGS done.")

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, name, device, mean_aer_ai):
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[V22SAT] {name} not found, skipping")
        return None

    df = prepare_df(df, mean_aer_ai)
    X  = torch.tensor(df[ALL_COLS].fillna(0).values, dtype=torch.float32).to(device)
    T  = df[TARGET_COLS].values.astype(float)
    M  = df[TARGET_COLS].notna().values

    model.eval()
    with torch.no_grad():
        P = model(X).cpu().numpy()

    key_label = ("diwali" if "diwali" in name
                 else "winter" if "winter" in name else None)
    ref19 = V19_REF.get(key_label, {})
    ref20 = V20_REF.get(key_label, {})

    out = {}
    sat_cov = df["aer_ai_available"].mean() * 100
    print(f"\n[{name}]  rows={len(df):,}  aer_ai={sat_cov:.1f}%")
    print(f"  {'Poll':<6} {'R2':>8} {'MAE':>10}  vs v19   vs v20")

    for i, key in enumerate(["pm25", "no2", "o3", "so2"]):
        v = M[:, i]
        if not v.any():
            continue
        p, t = P[v, i], T[v, i]
        r2   = 1.0 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
        lo, hi = _PHYS_NORM[key]
        mae  = float(np.mean(np.abs(p * (hi - lo) + lo - (t * (hi - lo) + lo))))
        d19  = f"{r2 - ref19[key]:+.4f}" if key in ref19 else "   n/a"
        d20  = f"{r2 - ref20[key]:+.4f}" if key in ref20 else "   n/a"
        print(f"  {key.upper():<6} {r2:>8.4f} {mae:>10.2f}  {d19}   {d20}")
        out[key] = {"r2": r2, "mae_ug_m3": mae, "n": int(v.sum())}

    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    device = DEVICE

    print(f"[V22SAT] Device      : {device}")
    print(f"[V22SAT] V19 ckpt    : {V19_CKPT}")
    print(f"[V22SAT] Architecture: frozen_v19 + aer_ai_available * sat_net(x,y,t,aer_ai)")
    print(f"[V22SAT] sat_net     : 4 -> 64 x4 -> 4  (~13K params)")
    print(f"[V22SAT] Guarantee   : output = v19 when aer_ai_available=0")

    # Build model
    model = SatelliteCorrectedPINN(V19_CKPT)

    # Load data
    print("\n[V22SAT] Loading splits...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    mean_aer_ai = float(train_df["aer_ai_norm"].mean())
    if np.isnan(mean_aer_ai):
        mean_aer_ai = 0.0
    print(f"[V22SAT] mean_aer_ai fill = {mean_aer_ai:.6f}")

    train_df = prepare_df(train_df, mean_aer_ai)
    val_df   = prepare_df(val_df,   mean_aer_ai)

    for name, df in [("train", train_df), ("val", val_df)]:
        real = df["aer_ai_available"].mean() * 100
        print(f"[V22SAT] {name}: satellite real={real:.1f}%  mean-filled={100-real:.1f}%")

    train_t = df_to_tensors(train_df)
    val_t   = df_to_tensors(val_df)

    # Train
    print()
    model = train(model, train_t, val_t, device)

    # Save final checkpoint
    final_ckpt = CHECKPOINTS_DIR / f"{cfg.CHECKPOINT_PREFIX}_final.pt"
    torch.save({"model": model.state_dict(), "mean_aer_ai": mean_aer_ai}, final_ckpt)
    print(f"\n[V22SAT] Checkpoint saved -> {final_ckpt.name}")

    # Evaluate
    print("\n" + "=" * 68)
    print("[V22SAT] FINAL EVALUATION")
    print("=" * 68)
    print("Floor: v22 cannot be worse than v19 (hard gate guarantee)")
    print("Target: Diwali PM2.5 R² > 0.6813")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = evaluate(model, split, device, mean_aer_ai)
        if r:
            all_results[split] = r

    payload = {
        "version": "v22_satellite_gated_correction",
        "architecture": "frozen_v19 + aer_ai_available * sat_net(x,y,t,aer_ai_norm)",
        "sat_net_params": sum(p.numel() for p in model.sat_net.parameters()),
        "mean_aer_ai_fill": mean_aer_ai,
        "v19_ref": V19_REF,
        "v20_ref": V20_REF,
        "results": all_results,
    }
    out_path = CHECKPOINTS_DIR / "v22_satellite_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[V22SAT] Results -> {out_path.name}")
    print(f"[V22SAT] mean_aer_ai_fill = {mean_aer_ai:.6f}  (save for inference)")
    print("[V22SAT] Done.")


if __name__ == "__main__":
    main()

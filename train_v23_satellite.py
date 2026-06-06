"""
train_v23_satellite.py  —  Enhanced Satellite-Gated Residual Correction
─────────────────────────────────────────────────────────────────────────
Builds on v22 (frozen v19 + hard-gated sat_net).

v22 sat_net used only [x, y, t, aer_ai_norm] — 4 inputs, 13K params.
v23 improvements:

1. Richer sat_net inputs (4 → 7):
   + u_wind_norm, v_wind_norm  — wind tells you WHERE aerosol came from
   + fire_count_7day_norm      — FIRMS 7-day rolling fire count (stubble burning)

2. Larger sat_net (64 → 128 hidden, 4 → 6 layers):
   13K → ~115K trainable params. Still tiny vs 527K frozen base.

3. Longer training (20K → 50K Adam epochs):
   Still ~3-5 min. sat_net is small enough to handle this.

Why wind matters:
   High AER_AI upwind of a station → PM2.5 will rise at that station.
   High AER_AI downwind → irrelevant. v22 couldn't distinguish these.

Why fire counts matter:
   Diwali = fireworks + stubble burning. FIRMS fire counts peak exactly
   during Diwali/Winter test periods. This is a direct emission proxy
   that v19 (using only wind/position) cannot capture.

Gate: aer_ai_available × sat_net(x,y,t,u,v,aer_ai,fire7d)
   When no satellite (summer months): gate=0, output=v19 exactly.
   When satellite available (Oct-Jan): gate=1, correction applied.
   Fire counts included in sat_net inputs (gated alongside AER_AI).
   Oct-Jan is exactly when fires peak → gate is appropriate.

Baselines:
   v19:  Diwali PM2.5=0.6813, NO2=0.4763, O3=0.4935, SO2=0.7013
         Winter  PM2.5=0.5967, NO2=0.5496, O3=0.2921, SO2=0.6326
   v22:  Diwali PM2.5=0.6779, NO2=0.5298, O3=0.5348, SO2=0.7070
         Winter  PM2.5=0.6058, NO2=0.5686, O3=0.3683, SO2=0.6369

Target: Diwali PM2.5 R² > 0.70 (push past v19).
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

cfg.INPUT_DIM         = 11   # v19 base
cfg.CHECKPOINT_PREFIX = "aeras_v23_satellite"

from src.config import CHECKPOINTS_DIR, DEVICE
from src.models.pinn import AerasPINN

SPLITS_DIR = Path("/kaggle/input/datasets/b1042rudrakumar/kaggle-dataset-fixed")
V19_CKPT   = Path("/kaggle/working/v19_checkpoints/aeras_v19_da4_final.pt")

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS           = 50_000
LR               = 1e-3
BATCH_SIZE       = 4096
LBFGS_MAX_ITER   = 1_000
LBFGS_CHUNK      = 20_000
CHECKPOINT_EVERY = 5_000

# ── Column definitions ────────────────────────────────────────────────────────
INPUT_COLS_BASE = [
    "x_norm", "y_norm", "t_norm",
    "u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm",
    "pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm", "so2_lag1h_norm",
]
# sat_net gets 7 inputs: position + wind + satellite + fire
SAT_FEAT_COLS = [
    "x_norm", "y_norm", "t_norm",
    "u_wind_norm", "v_wind_norm",
    "aer_ai_norm",
    "fire_count_7day_norm",
]
ALL_COLS    = INPUT_COLS_BASE + ["aer_ai_norm", "aer_ai_available", "fire_count_7day_norm"]
TARGET_COLS = ["pm25_norm", "no2_norm", "o3_norm", "so2_norm"]

_PHYS_NORM = {
    "pm25": (0.03, 509.75), "no2": (0.01, 278.21),
    "o3":   (0.01,  66.17), "so2": (0.01, 500.00),
}

V19_REF = {
    "diwali": {"pm25": 0.6813, "no2": 0.4763, "o3": 0.4935, "so2": 0.7013},
    "winter": {"pm25": 0.5967, "no2": 0.5496, "o3": 0.2921, "so2": 0.6326},
}
V22_REF = {
    "diwali": {"pm25": 0.6779, "no2": 0.5298, "o3": 0.5348, "so2": 0.7070},
    "winter": {"pm25": 0.6058, "no2": 0.5686, "o3": 0.3683, "so2": 0.6369},
}

SAT_IN_DIM = len(SAT_FEAT_COLS)   # 7


# ── Enhanced satellite correction network ─────────────────────────────────────
class SatCorrectionNet(nn.Module):
    """
    7-input MLP: [x, y, t, u_wind, v_wind, aer_ai_norm, fire_7day] -> 4 corrections.

    v23 vs v22:
      - 4 inputs -> 7 (adds wind + fire)
      - 64 hidden -> 128 hidden
      - 4 layers  -> 6 layers
      - 13K params -> ~115K params
    """
    def __init__(self, in_dim: int = SAT_IN_DIM, hidden: int = 128, out_dim: int = 4,
                 num_layers: int = 6):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        # Zero-init output: corrections start at 0, model at v19 floor
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full model ────────────────────────────────────────────────────────────────
class SatelliteCorrectedPINN(nn.Module):
    """
    output = frozen_v19(base[:11]) + aer_ai_available * sat_net(x,y,t,u,v,aer_ai,fire7d)

    inputs layout (14 features):
        [:11]  base v19 inputs
        [11]   aer_ai_norm          (mean-filled when missing)
        [12]   aer_ai_available     (1=real, 0=filled)
        [13]   fire_count_7day_norm (always real, from FIRMS)
    """
    def __init__(self, v19_ckpt: Path):
        super().__init__()

        self.base = AerasPINN()
        raw = torch.load(v19_ckpt, map_location="cpu")
        self.base.load_state_dict(raw["model"])
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        n_frozen = sum(p.numel() for p in self.base.parameters())
        print(f"[V23SAT] v19 frozen: {n_frozen:,} params")

        self.sat_net = SatCorrectionNet()
        n_train = sum(p.numel() for p in self.sat_net.parameters())
        print(f"[V23SAT] sat_net trainable: {n_train:,} params")
        print(f"[V23SAT] sat_net inputs: {SAT_FEAT_COLS}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_in   = inputs[:, :11]     # v19 base features
        aer_norm  = inputs[:, 11:12]   # aer_ai_norm
        available = inputs[:, 12:13]   # hard gate
        fire_norm = inputs[:, 13:14]   # fire_count_7day_norm

        # sat_net: [x, y, t, u, v, aer_ai, fire_7day]
        sat_in = torch.cat([
            inputs[:, 0:3],    # x, y, t
            inputs[:, 3:5],    # u_wind, v_wind
            aer_norm,
            fire_norm,
        ], dim=1)

        base_pred  = self.base(base_in)
        correction = self.sat_net(sat_in)

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
    if "fire_count_7day_norm" not in df.columns:
        print("[V23SAT] WARNING: fire_count_7day_norm not found, filling 0")
        df["fire_count_7day_norm"] = 0.0
    df["fire_count_7day_norm"] = df["fire_count_7day_norm"].fillna(0.0)
    return df


def df_to_tensors(df: pd.DataFrame) -> dict:
    inputs  = torch.tensor(df[ALL_COLS].fillna(0).values,    dtype=torch.float32)
    targets = torch.tensor(df[TARGET_COLS].fillna(0).values, dtype=torch.float32)
    mask    = torch.tensor(df[TARGET_COLS].notna().values,    dtype=torch.float32)

    ev = pd.Series(1.0, index=df.index)
    if "is_holiday" in df.columns:
        ev.loc[df["is_holiday"].astype(bool)] *= 3.0
    if "timestamp" in df.columns:
        ev.loc[df["timestamp"].dt.month.isin([12, 1])] *= 2.0
    if "pm25_norm" in df.columns:
        ev.loc[df["pm25_norm"] > 0.4] *= 2.0
    ew = torch.tensor(ev.values, dtype=torch.float32).unsqueeze(1)

    return {"inputs": inputs, "targets": targets, "mask": mask, "event_weight": ew}


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, train_t, val_t, device):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10_000, T_mult=2, eta_min=1e-6
    )

    inputs  = train_t["inputs"].to(device)
    targets = train_t["targets"].to(device)
    mask    = train_t["mask"].to(device)
    ew      = train_t["event_weight"].to(device)
    v_inp   = val_t["inputs"].to(device)
    v_tgt   = val_t["targets"].to(device)
    v_msk   = val_t["mask"].to(device)
    N       = len(inputs)
    t0      = time.time()

    print(f"[V23SAT] Training {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")

    for epoch in range(EPOCHS + 1):
        model.train(); model.base.eval()

        idx = torch.randperm(N, device=device)[:BATCH_SIZE]
        xb, yb, mb, wb = inputs[idx], targets[idx], mask[idx], ew[idx]
        denom = (mb * wb).sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss = ((model(xb) - yb) ** 2 * mb * wb).sum() / denom
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % CHECKPOINT_EVERY == 0:
            model.eval()
            with torch.no_grad():
                vp    = model(v_inp)
                vloss = ((vp - v_tgt).abs() * v_msk).sum() / v_msk.sum().clamp(min=1)
            elapsed = (time.time() - t0) / 60
            print(f"Epoch {epoch:>6} | Loss: {loss.item():.4e} | Val MAE: {vloss.item():.4e} | {elapsed:.1f}min")
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       CHECKPOINTS_DIR / f"{cfg.CHECKPOINT_PREFIX}_epoch_{epoch}.pt")

    # L-BFGS
    print("\n[V23SAT] L-BFGS fine-tuning...")
    idx  = torch.randperm(N, device=device)[:LBFGS_CHUNK]
    lx, ly, lm, lw = inputs[idx], targets[idx], mask[idx], ew[idx]
    lden = (lm * lw).sum().clamp(min=1.0)
    lbfgs = torch.optim.LBFGS(model.trainable_parameters(), max_iter=LBFGS_MAX_ITER,
                                lr=0.1, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        model.train(); model.base.eval()
        loss = ((model(lx) - ly) ** 2 * lm * lw).sum() / lden
        loss.backward()
        return loss

    lbfgs.step(closure)
    print("[V23SAT] L-BFGS done.")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, name, device, mean_aer_ai):
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[V23SAT] {name} not found"); return None

    df = prepare_df(df, mean_aer_ai)
    X  = torch.tensor(df[ALL_COLS].fillna(0).values, dtype=torch.float32).to(device)
    T  = df[TARGET_COLS].values.astype(float)
    M  = df[TARGET_COLS].notna().values

    model.eval()
    with torch.no_grad():
        P = model(X).cpu().numpy()

    key_label = "diwali" if "diwali" in name else "winter" if "winter" in name else None
    ref19 = V19_REF.get(key_label, {})
    ref22 = V22_REF.get(key_label, {})

    out  = {}
    sat  = df["aer_ai_available"].mean() * 100
    fire = df["fire_count_7day_norm"].mean()
    print(f"\n[{name}]  rows={len(df):,}  aer_ai={sat:.1f}%  fire_7d_mean={fire:.4f}")
    print(f"  {'Poll':<6} {'R2':>8} {'MAE':>10}  vs v19   vs v22")

    for i, key in enumerate(["pm25", "no2", "o3", "so2"]):
        v = M[:, i]
        if not v.any(): continue
        p, t = P[v, i], T[v, i]
        r2  = 1.0 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
        lo, hi = _PHYS_NORM[key]
        mae = float(np.mean(np.abs(p * (hi-lo) + lo - (t * (hi-lo) + lo))))
        d19 = f"{r2-ref19[key]:+.4f}" if key in ref19 else "   n/a"
        d22 = f"{r2-ref22[key]:+.4f}" if key in ref22 else "   n/a"
        print(f"  {key.upper():<6} {r2:>8.4f} {mae:>10.2f}  {d19}   {d22}")
        out[key] = {"r2": r2, "mae_ug_m3": mae, "n": int(v.sum())}
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.backends.cudnn.benchmark = True
    device = DEVICE

    print(f"[V23SAT] Device      : {device}")
    print(f"[V23SAT] V19 ckpt    : {V19_CKPT}")
    print(f"[V23SAT] sat_net in  : {SAT_FEAT_COLS}  ({SAT_IN_DIM} features)")
    print(f"[V23SAT] sat_net arch: {SAT_IN_DIM}->128x6->4  (~115K params)")
    print(f"[V23SAT] Guarantee   : output = v19 when aer_ai_available=0")

    model = SatelliteCorrectedPINN(V19_CKPT)

    print("\n[V23SAT] Loading splits...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    mean_aer_ai = float(train_df["aer_ai_norm"].mean())
    if np.isnan(mean_aer_ai): mean_aer_ai = 0.0
    print(f"[V23SAT] mean_aer_ai fill = {mean_aer_ai:.6f}")

    train_df = prepare_df(train_df, mean_aer_ai)
    val_df   = prepare_df(val_df,   mean_aer_ai)

    for name, df in [("train", train_df), ("val", val_df)]:
        sat  = df["aer_ai_available"].mean() * 100
        fire = df["fire_count_7day_norm"].mean()
        print(f"[V23SAT] {name}: sat={sat:.1f}%  fire_7d_mean={fire:.4f}")

    train_t = df_to_tensors(train_df)
    val_t   = df_to_tensors(val_df)

    print()
    model = train(model, train_t, val_t, device)

    final_ckpt = CHECKPOINTS_DIR / f"{cfg.CHECKPOINT_PREFIX}_final.pt"
    torch.save({"model": model.state_dict(), "mean_aer_ai": mean_aer_ai}, final_ckpt)
    print(f"\n[V23SAT] Saved -> {final_ckpt.name}")

    print("\n" + "=" * 68)
    print("[V23SAT] FINAL EVALUATION")
    print("=" * 68)
    print("Floor: output = v19 when gate=0 (hard guarantee)")
    print("Target: Diwali PM2.5 R² > 0.70")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = evaluate(model, split, device, mean_aer_ai)
        if r: all_results[split] = r

    payload = {
        "version": "v23_satellite_wind_fire",
        "architecture": "frozen_v19 + aer_ai_available * sat_net(x,y,t,u,v,aer_ai,fire7d)",
        "sat_net_inputs": SAT_FEAT_COLS,
        "sat_net_params": sum(p.numel() for p in model.sat_net.parameters()),
        "mean_aer_ai_fill": mean_aer_ai,
        "v19_ref": V19_REF,
        "v22_ref": V22_REF,
        "results": all_results,
    }
    out_path = CHECKPOINTS_DIR / "v23_satellite_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[V23SAT] Results -> {out_path.name}")
    print(f"[V23SAT] mean_aer_ai_fill = {mean_aer_ai:.6f}")
    print("[V23SAT] Done.")


if __name__ == "__main__":
    main()

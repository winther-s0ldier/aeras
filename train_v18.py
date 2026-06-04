"""
train_v18.py  —  Two-regime Chemistry + Full Multi-Pollutant DA (incl. SO2)
─────────────────────────────────────────────────────────────────────────────
Builds directly on v17's findings:

  v17 DA3_OFF  : DA alone rescued NO2/O3 (lag features)
  v17 CHEMDA_ON: Leighton helped PM2.5 + NO2 winter, HURT O3 (null-cycle)
  v16 ablation : PDE adds +0.12 R² over pure DA on Diwali

What v18 adds:
  1. Titration chemistry (TitrationChemistryModule) — two-regime:
       Daytime : NO2 photolysis → O3 produced      (Leighton was right here)
       Nighttime: NOx titration → O3 consumed       (Leighton was blind here)
     This is the correct mechanism for Diwali (nighttime NOx + O3 titration).
     Target: beat DA3_OFF on O3 Diwali (0.562) — which Leighton failed to do.

  2. SO2 lag feature (so2_lag1h_norm) → INPUT_DIM = 11
     Same recipe that rescued PM2.5 (0.05→0.71), NO2 and O3.
     SO2 has been at R²≈0 / negative all runs — this should rescue it.

Baselines to beat (v17 DA3_OFF, the honest control):
  PM2.5 Diwali: 0.709  |  NO2 Diwali: 0.501  |  O3 Diwali: 0.562
  SO2 Diwali  : -0.002 ← main SO2 target

Run on Kaggle (2× T4):
  CUDA_VISIBLE_DEVICES=0 python train_v18.py
"""

import os, sys, json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import src.config as cfg

# ── config overrides (must be set BEFORE importing AerasPINN / trainer) ───────
cfg.INPUT_DIM          = 11   # 7 base + pm25/no2/o3/so2 lags
cfg.CHECKPOINT_PREFIX  = "aeras_v18_titration"
cfg.BATCH_SIZE         = 4096
cfg.NUM_COLLOCATION    = 200_000
cfg.CHECKPOINT_EVERY   = 2_500
cfg.LBFGS_MAX_ITER     = 1_000
cfg.LBFGS_DATA_CHUNK   = 20_000
cfg.LBFGS_COLLOC_CHUNK = 15_000

from src.config import CHECKPOINTS_DIR, POLLUTANTS, DEVICE
from src.training.trainer import AerasTrainer
from src.models.pinn import AerasPINN
from src.models.chemistry import TitrationChemistryModule, DELHI_LATITUDE, IST_OFFSET_HOURS

SPLITS_DIR = Path("/kaggle/input/datasets/b1042rudrakumar/aeras-data/splits")

INPUT_COLS = [
    "x_norm", "y_norm", "t_norm",
    "u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm",
    "pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm",
    "so2_lag1h_norm",   # new — rescues SO2
]
TARGET_COLS = ["pm25_norm", "no2_norm", "o3_norm", "so2_norm"]

_NORM = {
    "pm25": {"min": 0.03,  "max": 509.75},
    "no2":  {"min": 0.01,  "max": 278.21},
    "o3":   {"min": 0.01,  "max": 66.17},
    "so2":  {"min": 0.01,  "max": 500.0},
}

# v17 DA3_OFF baselines — the honest control to beat
V17_REF = {
    "diwali": {"pm25": 0.7088, "no2": 0.5012, "o3": 0.5624, "so2": -0.0020},
    "winter": {"pm25": 0.6093, "no2": 0.5500, "o3": 0.1939, "so2": -0.1366},
}


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag-1h per station for ALL four pollutants. First row per station → 0.
    NEVER falls back to current value (target leak prevention).
    """
    df = df.sort_values(["station", "timestamp"]).copy()
    g  = df.groupby("station", observed=True)
    df["pm25_lag1h_norm"] = g["pm25_norm"].shift(1).fillna(0.0)
    df["no2_lag1h_norm"]  = g["no2_norm"].shift(1).fillna(0.0)
    df["o3_lag1h_norm"]   = g["o3_norm"].shift(1).fillna(0.0)
    df["so2_lag1h_norm"]  = g["so2_norm"].shift(1).fillna(0.0)
    return df


def df_to_tensors(df: pd.DataFrame, include_ic: bool = False) -> dict:
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values,  dtype=torch.float32)
    targets = torch.tensor(df[TARGET_COLS].fillna(0).values, dtype=torch.float32)
    mask    = torch.tensor(df[TARGET_COLS].notna().values,   dtype=torch.float32)

    ev = pd.Series(1.0, index=df.index)
    if "is_holiday" in df.columns:
        ev.loc[df["is_holiday"].astype(bool)] *= 3.0
    if "timestamp" in df.columns:
        ev.loc[df["timestamp"].dt.month.isin([12, 1])] *= 2.0
    if "pm25_norm" in df.columns:
        ev.loc[df["pm25_norm"] > 0.4] *= 2.0
    event_weight = torch.tensor(ev.values, dtype=torch.float32).unsqueeze(1)

    result = {"inputs": inputs, "targets": targets,
              "mask": mask, "event_weight": event_weight}

    if include_ic:
        ic_rows = (df.dropna(subset=TARGET_COLS)
                     .sort_values("t_norm")
                     .groupby("station", observed=True).first()
                     .reset_index())
        if len(ic_rows) > 0:
            result["ic_inputs"]  = torch.tensor(
                ic_rows[INPUT_COLS].fillna(0).values, dtype=torch.float32)
            result["ic_targets"] = torch.tensor(
                ic_rows[TARGET_COLS].fillna(0).values, dtype=torch.float32)
            print(f"[V18] IC points: {len(ic_rows)}")
    return result


def evaluate_test(trainer, name: str, device: str, chem):
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[V18] {name}.parquet not found, skipping")
        return None

    df      = add_lag_features(df)
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values, dtype=torch.float32).to(device)
    targets = df[TARGET_COLS].values.astype(float)
    mask    = df[TARGET_COLS].notna().values

    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(inputs).cpu().numpy()

    out = {}
    print(f"\n[{name}]")
    print(f"{'Poll':<6} {'R2':>8} {'MAE (ug/m3)':>13} {'n':>8}  vs v17_DA3_OFF")
    ref = V17_REF.get("diwali" if "diwali" in name else "winter" if "winter" in name else None, {})

    for i, key in enumerate(["pm25", "no2", "o3", "so2"]):
        v = mask[:, i]
        if v.sum() == 0:
            continue
        p, t   = pred[v, i], targets[v, i]
        mae_n  = float(np.mean(np.abs(p - t)))
        ss_r   = float(np.sum((p - t) ** 2))
        ss_t   = float(np.sum((t - t.mean()) ** 2))
        r2     = 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")
        vmin, vmax = _NORM[key]["min"], _NORM[key]["max"]
        mae_phys   = float(np.mean(np.abs(
            p * (vmax - vmin) + vmin - (t * (vmax - vmin) + vmin)
        )))
        delta = f"  {r2 - ref[key]:+.4f}" if key in ref else ""
        print(f"{key.upper():<6} {r2:>8.4f} {mae_phys:>13.2f} {int(v.sum()):>8}{delta}")
        out[key] = {"mae_norm": mae_n, "r2": r2, "mae_ug_m3": mae_phys, "n": int(v.sum())}

    # Log coverage of lag features (diagnostic)
    for col in ["no2_lag1h_norm", "o3_lag1h_norm", "so2_lag1h_norm"]:
        if col in df.columns:
            nonzero = (df[col] != 0).mean() * 100
            print(f"  lag coverage {col}: {nonzero:.1f}% non-zero")

    return out


def main():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

    device = DEVICE
    print(f"[V18] Device      : {device}")
    print(f"[V18] INPUT_DIM   : {cfg.INPUT_DIM}  (7 base + pm25/no2/o3/so2 lags)")
    print(f"[V18] Chemistry   : TitrationChemistryModule (two-regime: photolysis + NOx titration)")
    print(f"[V18] Prefix      : {cfg.CHECKPOINT_PREFIX}")

    # ── safety assertions ────────────────────────────────────────────────────
    assert POLLUTANTS == ["PM2.5", "NO2", "O3", "SO2"], (
        f"Channel order changed! Chemistry assumes 0=PM2.5,1=NO2,2=O3. Got {POLLUTANTS}"
    )
    _probe = AerasPINN()
    assert _probe.fourier.B.shape[0] == 11, (
        f"AerasPINN built with input_dim={_probe.fourier.B.shape[0]}, expected 11. "
        f"cfg.INPUT_DIM must be set before importing the trainer."
    )
    del _probe
    print("[V18] Architecture assertions passed.")

    # ── load data ─────────────────────────────────────────────────────────────
    print("\n[V18] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    print("[V18] Computing lag-1h features (pm25/no2/o3/so2)...")
    train_df = add_lag_features(train_df)
    val_df   = add_lag_features(val_df)

    for c in ["pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm", "so2_lag1h_norm"]:
        nz = (train_df[c] != 0).mean() * 100
        print(f"[V18]   {c}: {nz:.1f}% non-zero")

    train_data = df_to_tensors(train_df, include_ic=True)
    val_data   = df_to_tensors(val_df)

    # ── chemistry module ──────────────────────────────────────────────────────
    ts      = train_df["timestamp"]
    ts_utc  = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    t_min   = ts_utc.min().timestamp()
    t_range = ts_utc.max().timestamp() - t_min

    chem = TitrationChemistryModule(
        t_min_unix=t_min,
        t_range_sec=t_range,
        lat_deg=getattr(cfg, "DELHI_LATITUDE", DELHI_LATITUDE),
        tz_offset_hours=getattr(cfg, "IST_OFFSET_HOURS", IST_OFFSET_HOURS),
        log_J_amp_init=getattr(cfg, "LOG_J_AMP_INIT", -0.5),
        log_k_tit_init=getattr(cfg, "LOG_K_TIT_INIT", -2.0),
    )
    print(f"[V18] TitrationChemistry: J_amp_init={chem.J_amp.item():.4f}  "
          f"k_tit_init={chem.k_tit.item():.4f}")

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[V18] Training (INPUT_DIM=11, two-regime chemistry)...")
    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
        chemistry_module=chem,
    )
    trainer.train()

    # ── learned params ────────────────────────────────────────────────────────
    print(f"\n[V18] Learned J_amp  = {chem.J_amp.item():.6e}  "
          f"(init {torch.exp(torch.tensor(-0.5)).item():.4f})")
    print(f"[V18] Learned k_tit  = {chem.k_tit.item():.6e}  "
          f"(init {torch.exp(torch.tensor(-2.0)).item():.4f})")

    if chem.k_tit.item() < 1e-5:
        print("[V18] WARNING: k_tit collapsed near zero — titration did not engage.")
    if chem.J_amp.item() < 0.05:
        print("[V18] WARNING: J_amp collapsed — photolysis did not engage.")

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("[V18] FINAL EVALUATION")
    print("=" * 64)
    print("Delta shown vs v17 DA3_OFF (the honest control).")
    print(f"  v17 O3 Diwali: 0.5624  ← target to beat with titration")
    print(f"  v17 SO2 Diwali: -0.002 ← target to rescue with SO2 lag")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = evaluate_test(trainer, split, device, chem)
        if r:
            all_results[split] = r

    payload = {
        "version": "v18_titration",
        "input_dim": cfg.INPUT_DIM,
        "chemistry": "TitrationChemistryModule",
        "learned": {
            "J_amp":  chem.J_amp.item(),
            "k_tit":  chem.k_tit.item(),
        },
        "v17_da3_off_ref": V17_REF,
        "results": all_results,
    }

    out_path = CHECKPOINTS_DIR / "v18_titration_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[V18] Results saved -> {out_path.name}")
    print("[V18] Done.")


if __name__ == "__main__":
    main()

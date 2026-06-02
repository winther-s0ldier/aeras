"""
train_chem.py
───────────
Coupled NO2/O3 Photochemistry PINN.

Run on GPU 0:
  CUDA_VISIBLE_DEVICES=0 python train_chem.py
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

# ── critical: set INPUT_DIM=8 BEFORE importing AerasPINN ─────────────────────
cfg.INPUT_DIM = 8
cfg.CHECKPOINT_PREFIX = "aeras_v14_chem"
cfg.USE_CHEMISTRY = True
cfg.BATCH_SIZE        = 4096
cfg.NUM_COLLOCATION   = 200_000
cfg.CHECKPOINT_EVERY  = 2_500
cfg.LBFGS_MAX_ITER    = 1_000
cfg.LBFGS_DATA_CHUNK  = 20_000
cfg.LBFGS_COLLOC_CHUNK = 15_000

from src.config import CHECKPOINTS_DIR
from src.training.trainer import AerasTrainer

SPLITS_DIR = Path("/kaggle/input/datasets/rudrakumar21/aeras-splits")

INPUT_COLS  = ["x_norm","y_norm","t_norm","u_wind_norm","v_wind_norm",
               "temp_norm","blh_norm","pm25_lag1h_norm"]
TARGET_COLS = ["pm25_norm","no2_norm","o3_norm","so2_norm"]

_NORM = {
    "pm25": {"min": 0.03,  "max": 509.75},
    "no2":  {"min": 0.01,  "max": 278.21},
    "o3":   {"min": 0.01,  "max": 66.17},
    "so2":  {"min": 0.01,  "max": 500.0},
}


def add_lag_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pm25_lag1h_norm column: PM2.5 from previous hour at the same station.
    For each station, sort by time and shift(1). Missing lag values are filled
    with current PM2.5 (effectively no DA signal for first observation).
    """
    df = df.sort_values(["station", "timestamp"]).copy()
    df["pm25_lag1h_norm"] = df.groupby(
        "station", observed=True
    )["pm25_norm"].shift(1)

    # Fill the first row of each station (no prior obs) with 0.
    # IMPORTANT: do NOT fall back to df["pm25_norm"] — that leaks the target
    # into the input for first-row-per-station observations.
    df["pm25_lag1h_norm"] = df["pm25_lag1h_norm"].fillna(0.0)
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
            print(f"[DA] IC points: {len(ic_rows)}")

    return result


def evaluate_test(trainer, name: str, device: str):
    """Evaluate on one test split and return results dict."""
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[CHEM] {name}.parquet not found, skipping")
        return None

    df = add_lag_feature(df)
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values, dtype=torch.float32).to(device)
    targets = df[TARGET_COLS].values.astype(float)
    mask    = df[TARGET_COLS].notna().values

    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(inputs).cpu().numpy()

    out = {}
    print(f"\n[{name}]")
    print(f"{'Poll':<6} {'R²':>8} {'MAE (μg/m³)':>13} {'n':>8}")
    for i, key in enumerate(["pm25","no2","o3","so2"]):
        v = mask[:,i]
        if v.sum() == 0:
            continue
        p, t = pred[v,i], targets[v,i]
        mae_n  = float(np.mean(np.abs(p - t)))
        ss_r   = float(np.sum((p - t)**2))
        ss_t   = float(np.sum((t - t.mean())**2))
        r2     = 1.0 - ss_r / ss_t if ss_t > 0 else float('nan')
        vmin, vmax = _NORM[key]["min"], _NORM[key]["max"]
        mae_phys = float(np.mean(np.abs(
            p * (vmax - vmin) + vmin - (t * (vmax - vmin) + vmin)
        )))
        print(f"{key.upper():<6} {r2:>8.4f} {mae_phys:>13.2f} {int(v.sum()):>8}")
        out[key] = {"mae_norm": mae_n, "r2": r2, "mae_ug_m3": mae_phys, "n": int(v.sum())}
    return out


def main():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

    device = cfg.DEVICE
    print(f"[CHEM] Device : {device}")
    print(f"[CHEM] INPUT_DIM : {cfg.INPUT_DIM}  (7 base + 1 PM2.5 lag-1h)")
    print(f"[CHEM] Output prefix : {cfg.CHECKPOINT_PREFIX}")

    print("\n[CHEM] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    
    # Ensure UTC interpretation. CPCB data may be naive (no tz attached).
    # If naive, assume UTC (ERA5 standard) — chemistry module will apply IST offset internally.
    ts_col = train_df["timestamp"]
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize("UTC")
    else:
        ts_col = ts_col.dt.tz_convert("UTC")

    T_MIN_UNIX = ts_col.min().timestamp()
    T_MAX_UNIX = ts_col.max().timestamp()
    T_RANGE_SEC = T_MAX_UNIX - T_MIN_UNIX

    print(f"[CHEM] T_MIN_UNIX = {T_MIN_UNIX:.1f} ({ts_col.min()})")
    print(f"[CHEM] T_RANGE_SEC = {T_RANGE_SEC:.1f} ({T_RANGE_SEC / 86400:.1f} days)")

    # Instantiate chemistry module:
    from src.models.chemistry import ChemistryModule
    chem = ChemistryModule(
        t_min_unix=T_MIN_UNIX,
        t_range_sec=T_RANGE_SEC,
        lat_deg=cfg.DELHI_LATITUDE,
        tz_offset_hours=cfg.IST_OFFSET_HOURS,
        log_J_amp_init=cfg.LOG_J_AMP_INIT,
        log_leighton_eps_init=cfg.LOG_LEIGHTON_EPS_INIT,
    )

    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    print("[CHEM] Computing PM2.5 lag-1h feature...")
    train_df = add_lag_feature(train_df)
    val_df   = add_lag_feature(val_df)

    lag_mean = train_df["pm25_lag1h_norm"].mean()
    lag_std  = train_df["pm25_lag1h_norm"].std()
    print(f"[CHEM] Lag feature stats: mean={lag_mean:.4f}  std={lag_std:.4f}")
    print(f"[CHEM] Non-null lag rows: {train_df['pm25_lag1h_norm'].notna().sum():,} "
          f"/ {len(train_df):,}")

    train_data = df_to_tensors(train_df, include_ic=True)
    val_data   = df_to_tensors(val_df)

    print(f"\n[CHEM] Training forward PINN with INPUT_DIM=8...")
    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
        chemistry_module=chem,
    )
    trainer.train()

    # After training, print learned chemistry params:
    print(f"\n[CHEM] Learned J_amp        = {chem.J_amp.item():.6e}")
    print(f"[CHEM] Learned leighton_eps = {chem.leighton_eps.item():.6e}")

    # Save chemistry params to JSON for paper:
    chem_state = {
        "J_amp": chem.J_amp.item(),
        "leighton_eps": chem.leighton_eps.item(),
        "log_J_amp": chem.log_J_amp.item(),
        "log_leighton_eps": chem.log_leighton_eps.item(),
        "T_MIN_UNIX": T_MIN_UNIX,
        "T_RANGE_SEC": T_RANGE_SEC,
    }
    with open(CHECKPOINTS_DIR / "chemistry_learned_params.json", "w") as f:
        json.dump(chem_state, f, indent=2)

    print("\n" + "=" * 60)
    print("[CHEM] FINAL EVALUATION — Chem vs DA vs Baseline")
    print("=" * 60)
    
    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = evaluate_test(trainer, split, device)
        if r:
            all_results[split] = r

    out_path = CHECKPOINTS_DIR / "chem_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[CHEM] Results saved → {out_path.name}")
    print("[CHEM] Done.")


if __name__ == "__main__":
    main()

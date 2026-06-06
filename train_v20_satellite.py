"""
train_v20_satellite.py  —  Satellite-Augmented Forward PINN
─────────────────────────────────────────────────────────────
Builds on v19 DA4 (all-4-pollutant, 4 lags, no chemistry).

What v20 adds:
    INPUT_DIM = 12:  v19's 11 features + aer_ai_norm (12th)
    aer_ai_norm: Sentinel-5P Absorbing Aerosol Index, scaled to [0,1]
    Zero-filled when missing (34% non-zero in train, ~100% Diwali/Winter)

Hypothesis:
    AER_AI is a column-integrated smoke/dust proxy independent of PM2.5 sensors.
    Diwali: fireworks + stubble burning -> elevated AER_AI over Delhi.
    The PINN uses this as spatial constraint where surface sensors are absent.

V19 DA4 baselines:
    Diwali: PM2.5=0.6813, NO2=0.4763, O3=0.4935, SO2=0.7013
    Winter: PM2.5=0.5967, NO2=0.5496, O3=0.2921, SO2=0.6326
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

# ── Config (must be set BEFORE importing AerasPINN / trainer) ────────────────
cfg.INPUT_DIM         = 12   # 7 base + 4 lags + aer_ai_norm
cfg.CHECKPOINT_PREFIX = "aeras_v20_satellite"
cfg.BATCH_SIZE        = 4096
cfg.NUM_COLLOCATION   = 200_000
cfg.CHECKPOINT_EVERY  = 2_500
cfg.LBFGS_MAX_ITER    = 1_000
cfg.LBFGS_DATA_CHUNK  = 20_000
cfg.LBFGS_COLLOC_CHUNK = 15_000

from src.config import CHECKPOINTS_DIR, POLLUTANTS, DEVICE
from src.training.trainer import AerasTrainer
from src.models.pinn import AerasPINN

SPLITS_DIR = Path("/kaggle/input/datasets/b1042rudrakumar/kaggle-dataset-fixed")

INPUT_COLS = [
    "x_norm", "y_norm", "t_norm",
    "u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm",
    "pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm", "so2_lag1h_norm",
    "aer_ai_norm",      # Sentinel-5P Absorbing Aerosol Index
]
TARGET_COLS = ["pm25_norm", "no2_norm", "o3_norm", "so2_norm"]

_PHYS_NORM = {
    "pm25": (0.03, 509.75), "no2": (0.01, 278.21),
    "o3":   (0.01,  66.17), "so2": (0.01, 500.00),
}

V19_REF = {
    "diwali": {"pm25": 0.6813, "no2": 0.4763, "o3": 0.4935, "so2": 0.7013},
    "winter": {"pm25": 0.5967, "no2": 0.5496, "o3": 0.2921, "so2": 0.6326},
}


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "timestamp"]).copy()
    g  = df.groupby("station", observed=True)
    df["pm25_lag1h_norm"] = g["pm25_norm"].shift(1).fillna(0.0)
    df["no2_lag1h_norm"]  = g["no2_norm"].shift(1).fillna(0.0)
    df["o3_lag1h_norm"]   = g["o3_norm"].shift(1).fillna(0.0)
    df["so2_lag1h_norm"]  = g["so2_norm"].shift(1).fillna(0.0)
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    if "aer_ai_norm" not in df.columns:
        df["aer_ai_norm"] = 0.0
    df["aer_ai_norm"] = df["aer_ai_norm"].fillna(0.0)
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
            print(f"[V20SAT] IC points: {len(ic_rows)}")
    return result


def evaluate_test(trainer, name: str, device: str) -> dict:
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[V20SAT] {name}.parquet not found, skipping")
        return None

    df = prepare_df(df)
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values,
                           dtype=torch.float32).to(device)
    targets = df[TARGET_COLS].values.astype(float)
    mask    = df[TARGET_COLS].notna().values

    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(inputs).cpu().numpy()

    out = {}
    print(f"\n[{name}]")
    key_label = ("diwali" if "diwali" in name
                 else "winter" if "winter" in name else None)
    ref = V19_REF.get(key_label, {})
    print(f"{'Poll':<6} {'R2':>8} {'MAE(ug/m3)':>12} {'n':>8}  vs v19")

    for i, key in enumerate(["pm25", "no2", "o3", "so2"]):
        v = mask[:, i]
        if v.sum() == 0:
            continue
        p, t   = pred[v, i], targets[v, i]
        ss_r   = float(np.sum((p - t) ** 2))
        ss_t   = float(np.sum((t - t.mean()) ** 2))
        r2     = 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")
        vmin, vmax = _PHYS_NORM[key]
        mae_phys   = float(np.mean(np.abs(
            p * (vmax - vmin) + vmin - (t * (vmax - vmin) + vmin)
        )))
        delta = f"  {r2 - ref[key]:+.4f}" if key in ref else ""
        print(f"{key.upper():<6} {r2:>8.4f} {mae_phys:>12.2f} {int(v.sum()):>8}{delta}")
        out[key] = {"r2": r2, "mae_ug_m3": mae_phys, "n": int(v.sum())}

    sat_cov = (df["aer_ai_norm"] > 0).mean() * 100
    print(f"  [sat] aer_ai_norm coverage: {sat_cov:.1f}%")
    return out


def main():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    device = DEVICE

    print(f"[V20SAT] Device     : {device}")
    print(f"[V20SAT] INPUT_DIM  : {cfg.INPUT_DIM}  (7 base + 4 lags + aer_ai_norm)")
    print(f"[V20SAT] Prefix     : {cfg.CHECKPOINT_PREFIX}")
    print(f"[V20SAT] Data dir   : {SPLITS_DIR}")

    assert POLLUTANTS == ["PM2.5", "NO2", "O3", "SO2"], f"Channel order wrong: {POLLUTANTS}"
    _probe = AerasPINN()
    assert _probe.fourier.B.shape[0] == 12, (
        f"AerasPINN input_dim={_probe.fourier.B.shape[0]}, expected 12."
    )
    del _probe
    print("[V20SAT] Architecture assertions passed (INPUT_DIM=12).")

    print("\n[V20SAT] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    print("[V20SAT] Preparing features (lags + satellite zero-fill)...")
    train_df = prepare_df(train_df)
    val_df   = prepare_df(val_df)

    for split_name, df in [("train", train_df), ("val", val_df)]:
        sat_cov = (df["aer_ai_norm"] > 0).mean() * 100
        print(f"[V20SAT] {split_name} aer_ai_norm non-zero: {sat_cov:.1f}%")

    train_data = df_to_tensors(train_df, include_ic=True)
    val_data   = df_to_tensors(val_df)

    print(f"\n[V20SAT] Training (INPUT_DIM=12, satellite-augmented)...")
    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
    )
    trainer.train()

    print("\n" + "=" * 64)
    print("[V20SAT] FINAL EVALUATION")
    print("=" * 64)
    print("Delta shown vs v19 DA4 (baseline to beat).")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = evaluate_test(trainer, split, device)
        if r:
            all_results[split] = r

    payload = {
        "version": "v20_satellite",
        "input_dim": cfg.INPUT_DIM,
        "features": INPUT_COLS,
        "v19_ref": V19_REF,
        "results": all_results,
    }
    out_path = CHECKPOINTS_DIR / "v20_satellite_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[V20SAT] Results saved -> {out_path.name}")
    print("[V20SAT] Done.")


if __name__ == "__main__":
    main()

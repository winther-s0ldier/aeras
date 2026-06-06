"""
train_v21_satellite.py  —  Satellite-Augmented Forward PINN (indicator fix)
─────────────────────────────────────────────────────────────────────────────
Builds on v20 (INPUT_DIM=12). Fixes the zero-fill distribution shift that
caused v20 to underperform v19.

What v20 did wrong:
    aer_ai_norm zero-filled when missing (77% of training rows).
    Model learned "aer_ai≈0 is normal" and ignored it.
    At Diwali/Winter test (100% real satellite), model couldn't use it.

What v21 fixes:
    INPUT_DIM = 13: adds aer_ai_available binary indicator (13th feature)
        0 = value was missing, filled with training mean
        1 = real Sentinel-5P value
    Mean-fill instead of zero-fill: missing aer_ai gets training-set mean,
        not 0, so fill value sits in the middle of the distribution.
    Model now learns: "when indicator=1, trust aer_ai_norm."

Why this works:
    Training has real satellite for Oct-Jan (same seasons as Diwali/Winter test).
    Model sees indicator=1 during high-pollution Diwali-like months in training
    (2018, 2020, 2021, 2022 Oct-Nov). At test on 2019 Diwali, indicator=1 again
    -> model knows to use the satellite signal.

Baselines:
    v19 DA4:  Diwali PM2.5=0.6813, NO2=0.4763, O3=0.4935, SO2=0.7013
              Winter  PM2.5=0.5967, NO2=0.5496, O3=0.2921, SO2=0.6326
    v20 sat:  Diwali PM2.5=0.5827, NO2=0.2807, O3=0.0427, SO2=0.5007
              Winter  PM2.5=0.4389, NO2=0.3223, O3=-0.5482, SO2=0.4416
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
cfg.INPUT_DIM         = 13   # 7 base + 4 lags + aer_ai_norm + aer_ai_available
cfg.CHECKPOINT_PREFIX = "aeras_v21_satellite"
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
    "aer_ai_norm",        # Sentinel-5P AER_AI (mean-filled when missing)
    "aer_ai_available",   # 1 = real satellite value, 0 = mean-filled
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
V20_REF = {
    "diwali": {"pm25": 0.5827, "no2": 0.2807, "o3": 0.0427, "so2": 0.5007},
    "winter": {"pm25": 0.4389, "no2": 0.3223, "o3": -0.5482, "so2": 0.4416},
}


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "timestamp"]).copy()
    g  = df.groupby("station", observed=True)
    df["pm25_lag1h_norm"] = g["pm25_norm"].shift(1).fillna(0.0)
    df["no2_lag1h_norm"]  = g["no2_norm"].shift(1).fillna(0.0)
    df["o3_lag1h_norm"]   = g["o3_norm"].shift(1).fillna(0.0)
    df["so2_lag1h_norm"]  = g["so2_norm"].shift(1).fillna(0.0)
    return df


def prepare_df(df: pd.DataFrame, mean_aer_ai: float) -> pd.DataFrame:
    """Add lag features, indicator flag, and mean-fill aer_ai."""
    df = add_lag_features(df)
    if "aer_ai_norm" not in df.columns:
        df["aer_ai_norm"] = np.nan
    # Binary indicator: 1 = real S5P value, 0 = filled
    df["aer_ai_available"] = df["aer_ai_norm"].notna().astype(float)
    # Mean-fill: missing gets training mean, not zero
    df["aer_ai_norm"] = df["aer_ai_norm"].fillna(mean_aer_ai)
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
            print(f"[V21SAT] IC points: {len(ic_rows)}")
    return result


def evaluate_test(trainer, name: str, device: str, mean_aer_ai: float) -> dict:
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[V21SAT] {name}.parquet not found, skipping")
        return None

    df = prepare_df(df, mean_aer_ai)
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
    ref19 = V19_REF.get(key_label, {})
    ref20 = V20_REF.get(key_label, {})

    sat_cov  = df["aer_ai_available"].mean() * 100
    sat_mean = df.loc[df["aer_ai_available"] == 1, "aer_ai_norm"].mean()
    print(f"  aer_ai coverage={sat_cov:.1f}%  mean(real)={sat_mean:.4f}  fill={mean_aer_ai:.4f}")
    print(f"  {'Poll':<6} {'R2':>8} {'MAE(ug/m3)':>12} {'n':>8}  vs v19   vs v20")

    for i, key in enumerate(["pm25", "no2", "o3", "so2"]):
        v = mask[:, i]
        if v.sum() == 0:
            continue
        p, t   = pred[v, i], targets[v, i]
        ss_r   = float(np.sum((p - t) ** 2))
        ss_t   = float(np.sum((t - t.mean()) ** 2))
        r2     = 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")
        vmin, vmax = _PHYS_NORM[key]
        mae    = float(np.mean(np.abs(
            p * (vmax - vmin) + vmin - (t * (vmax - vmin) + vmin)
        )))
        d19 = f"{r2 - ref19[key]:+.4f}" if key in ref19 else "  n/a  "
        d20 = f"{r2 - ref20[key]:+.4f}" if key in ref20 else "  n/a  "
        print(f"  {key.upper():<6} {r2:>8.4f} {mae:>12.2f} {int(v.sum()):>8}  {d19}  {d20}")
        out[key] = {"r2": r2, "mae_ug_m3": mae, "n": int(v.sum())}

    return out


def main():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    device = DEVICE

    print(f"[V21SAT] Device      : {device}")
    print(f"[V21SAT] INPUT_DIM   : {cfg.INPUT_DIM}  (11 DA + aer_ai_norm + aer_ai_available)")
    print(f"[V21SAT] Prefix      : {cfg.CHECKPOINT_PREFIX}")
    print(f"[V21SAT] Data dir    : {SPLITS_DIR}")
    print(f"[V21SAT] Fix         : mean-fill + binary indicator (no zero-fill)")

    assert POLLUTANTS == ["PM2.5", "NO2", "O3", "SO2"], f"Channel order wrong: {POLLUTANTS}"
    _probe = AerasPINN()
    assert _probe.fourier.B.shape[0] == 13, (
        f"AerasPINN input_dim={_probe.fourier.B.shape[0]}, expected 13. "
        f"cfg.INPUT_DIM={cfg.INPUT_DIM}"
    )
    del _probe
    print("[V21SAT] Architecture assertions passed (INPUT_DIM=13).")

    print("\n[V21SAT] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    # Compute mean from TRAINING SET ONLY (no leakage into val/test)
    mean_aer_ai = float(train_df["aer_ai_norm"].mean()) if "aer_ai_norm" in train_df.columns else 0.0
    if np.isnan(mean_aer_ai):
        mean_aer_ai = 0.0
    print(f"[V21SAT] Training aer_ai mean (non-null): {mean_aer_ai:.6f}")

    print("[V21SAT] Preparing features (lags + indicator + mean-fill)...")
    train_df = prepare_df(train_df, mean_aer_ai)
    val_df   = prepare_df(val_df, mean_aer_ai)

    for split_name, df in [("train", train_df), ("val", val_df)]:
        real_cov  = df["aer_ai_available"].mean() * 100
        zero_fill = (df["aer_ai_available"] == 0).mean() * 100
        print(f"[V21SAT] {split_name}: aer_ai real={real_cov:.1f}%  mean-filled={zero_fill:.1f}%")

    train_data = df_to_tensors(train_df, include_ic=True)
    val_data   = df_to_tensors(val_df)

    print(f"\n[V21SAT] Training (INPUT_DIM=13, indicator fix)...")
    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
    )
    trainer.train()

    print("\n" + "=" * 68)
    print("[V21SAT] FINAL EVALUATION")
    print("=" * 68)
    print("Targets: v21 R2 > v19 (0.68 Diwali PM2.5) and > v20 (0.58)")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = evaluate_test(trainer, split, device, mean_aer_ai)
        if r:
            all_results[split] = r

    payload = {
        "version": "v21_satellite_indicator",
        "input_dim": cfg.INPUT_DIM,
        "features": INPUT_COLS,
        "mean_aer_ai_fill": mean_aer_ai,
        "v19_ref": V19_REF,
        "v20_ref": V20_REF,
        "results": all_results,
    }
    out_path = CHECKPOINTS_DIR / "v21_satellite_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[V21SAT] Results saved -> {out_path.name}")
    print(f"[V21SAT] mean_aer_ai_fill={mean_aer_ai:.6f}  (save this for inference)")
    print("[V21SAT] Done.")


if __name__ == "__main__":
    main()

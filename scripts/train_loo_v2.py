"""
train_loo_v2.py
───────────────
Leave-One-Out spatial validation — v2 fix.

v1 picked NISE Gwal Pahari (IMD weather station, no PM2.5 data) by accident.
v2 filters held-out candidates to stations with >50% PM2.5 coverage.

Run on GPU 1:
  CUDA_VISIBLE_DEVICES=1 python train_loo_v2.py
"""

import os, sys, json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import src.config as cfg
from src.config import CHECKPOINTS_DIR
from src.training.trainer import AerasTrainer

SPLITS_DIR    = Path("/kaggle/input/datasets/rudrakumar21/aeras-splits")
OUTPUT_PREFIX = "aeras_v12_loo_v2"

INPUT_COLS  = ["x_norm","y_norm","t_norm","u_wind_norm","v_wind_norm","temp_norm","blh_norm"]
TARGET_COLS = ["pm25_norm","no2_norm","o3_norm","so2_norm"]
POLL_KEYS   = ["pm25", "no2", "o3", "so2"]

_NORM_PARAMS_FALLBACK = {
    "pm25": {"min": 0.03,  "max": 509.75},
    "no2":  {"min": 0.01,  "max": 278.21},
    "o3":   {"min": 0.01,  "max": 66.17},
    "so2":  {"min": 0.01,  "max": 500.0},
}

CENTROID_X = 0.5
CENTROID_Y = 0.5
MIN_PM25_COVERAGE = 0.5


def pick_held_out_station(df: pd.DataFrame) -> str:
    """
    Pick most-central station that HAS PM2.5 data.

    v1 picked Gwal Pahari (IMD) which only had NO2 — eval crashed.
    v2 requires PM2.5 coverage > 50% so the LOO test is meaningful.
    """
    stats = df.groupby("station").agg(
        x_norm=("x_norm", "mean"),
        y_norm=("y_norm", "mean"),
        pm25_count=("pm25_norm", "count"),
        total=("pm25_norm", "size"),
    ).dropna()
    stats["pm25_coverage"] = stats["pm25_count"] / stats["total"]
    stats["dist"] = ((stats["x_norm"] - CENTROID_X)**2 +
                     (stats["y_norm"] - CENTROID_Y)**2)**0.5

    valid = stats[stats["pm25_coverage"] > MIN_PM25_COVERAGE]
    print(f"[LOO-v2] {len(valid)} stations with PM2.5 coverage > {MIN_PM25_COVERAGE*100:.0f}%")

    chosen = valid["dist"].idxmin()
    s = stats.loc[chosen]
    print(f"[LOO-v2] Held-out station : '{chosen}'")
    print(f"[LOO-v2]   x_norm={s['x_norm']:.4f}  y_norm={s['y_norm']:.4f}")
    print(f"[LOO-v2]   dist from centroid = {s['dist']:.4f}")
    print(f"[LOO-v2]   PM2.5 readings: {int(s['pm25_count'])} / {int(s['total'])} "
          f"({s['pm25_coverage']*100:.1f}%)")
    return chosen


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
            print(f"[LOO-v2] IC points: {len(ic_rows)}")

    return result


def main():
    cfg.BATCH_SIZE          = 4096
    cfg.NUM_COLLOCATION     = 200_000
    cfg.CHECKPOINT_EVERY    = 2_500
    cfg.LBFGS_MAX_ITER      = 1_000
    cfg.LBFGS_DATA_CHUNK    = 20_000
    cfg.LBFGS_COLLOC_CHUNK  = 15_000
    cfg.CHECKPOINT_PREFIX   = OUTPUT_PREFIX

    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

    device = cfg.DEVICE
    print(f"[LOO-v2] Device : {device}")
    print(f"[LOO-v2] Output prefix : {OUTPUT_PREFIX}")

    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    held_out     = pick_held_out_station(train_df)
    held_out_df  = train_df[train_df["station"] == held_out].copy()
    train_loo_df = train_df[train_df["station"] != held_out].copy()

    if "station" in val_df.columns:
        val_loo_df = val_df[val_df["station"] != held_out].copy()
    else:
        val_loo_df = val_df

    print(f"[LOO-v2] Train rows  : {len(train_df):,} → {len(train_loo_df):,}")
    print(f"[LOO-v2] Val rows    : {len(val_df):,} → {len(val_loo_df):,}")
    print(f"[LOO-v2] Held-out    : {len(held_out_df):,} rows (never seen)")
    print(f"[LOO-v2] PM2.5 in held-out: {held_out_df['pm25_norm'].notna().sum()} rows")

    with open(CHECKPOINTS_DIR / "loo_v2_station.json", "w") as f:
        json.dump({
            "held_out_station": held_out,
            "held_out_rows": len(held_out_df),
            "pm25_rows": int(held_out_df["pm25_norm"].notna().sum()),
        }, f, indent=2)

    train_data = df_to_tensors(train_loo_df, include_ic=True)
    val_data   = df_to_tensors(val_loo_df)

    print(f"\n[LOO-v2] Training fresh forward PINN...")
    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
    )
    trainer.train()

    print(f"\n[LOO-v2] Evaluating at held-out station '{held_out}'...")
    trainer.model.eval()

    held_inputs  = torch.tensor(
        held_out_df[INPUT_COLS].fillna(0).values, dtype=torch.float32
    ).to(device)
    held_targets = held_out_df[TARGET_COLS].values.astype(float)
    held_mask    = held_out_df[TARGET_COLS].notna().values

    with torch.no_grad():
        held_pred = trainer.model(held_inputs).cpu().numpy()

    results = {"held_out_station": held_out, "pollutants": {}}
    print(f"\n{'Pollutant':<10} {'R² (norm)':>10} {'MAE (μg/m³)':>13} {'n':>8}")
    print("-" * 45)

    for i, key in enumerate(POLL_KEYS):
        valid = held_mask[:, i]
        if valid.sum() == 0:
            print(f"{key.upper():<10} {'(no data)':>10}")
            continue

        pred_n = held_pred[valid, i]
        tgt_n  = held_targets[valid, i]

        mae_n  = float(np.mean(np.abs(pred_n - tgt_n)))
        ss_res = float(np.sum((pred_n - tgt_n)**2))
        ss_tot = float(np.sum((tgt_n - tgt_n.mean())**2))
        r2_n   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        vmin, vmax = _NORM_PARAMS_FALLBACK[key]["min"], _NORM_PARAMS_FALLBACK[key]["max"]
        mae_phys = float(np.mean(np.abs(
            pred_n * (vmax - vmin) + vmin - (tgt_n * (vmax - vmin) + vmin)
        )))

        print(f"{key.upper():<10} {r2_n:>10.4f} {mae_phys:>13.2f} {valid.sum():>8}")
        results["pollutants"][key] = {
            "mae_normalised":  mae_n,
            "r2_normalised":   r2_n,
            "mae_ug_m3":       mae_phys,
            "n":               int(valid.sum()),
        }

    out_path = CHECKPOINTS_DIR / "loo_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[LOO-v2] Results saved → {out_path.name}")

    pm25_r2 = results["pollutants"].get("pm25", {}).get("r2_normalised", float("nan"))
    print(f"\n→ PM2.5 R² at unseen station = {pm25_r2:.4f}")
    if pm25_r2 > 0.1:
        print("  ✓ SPATIAL GENERALISATION CONFIRMED")
    elif pm25_r2 > 0:
        print("  ~ Weak positive")
    else:
        print("  ✗ Not confirmed — document as limitation")


if __name__ == "__main__":
    main()

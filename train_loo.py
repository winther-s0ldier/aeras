"""
train_loo.py
────────────
Leave-One-Out (LOO) spatial validation.

Removes the station closest to the Delhi centroid from training, trains a fresh
forward PINN, then measures prediction accuracy at that station's location.

If R² > 0 at the held-out station, the PINN demonstrably generalises in space —
it is not just memorising values at known sensor locations. This is the experiment
that validates the core spatial-interpolation claim.

Run on GPU 1:
  CUDA_VISIBLE_DEVICES=1 python train_loo.py
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

# ── config ────────────────────────────────────────────────────────────────────
SPLITS_DIR    = Path("/kaggle/input/datasets/rudrakumar21/aeras-splits")
OUTPUT_PREFIX = "aeras_v12_loo"

INPUT_COLS  = ["x_norm","y_norm","t_norm","u_wind_norm","v_wind_norm","temp_norm","blh_norm"]
TARGET_COLS = ["pm25_norm","no2_norm","o3_norm","so2_norm"]
POLL_KEYS   = ["pm25", "no2", "o3", "so2"]

# Hardcoded PM2.5 normalization params as fallback (from data/processed/normalized_params.json)
_NORM_PARAMS_FALLBACK = {
    "pm25": {"min": 0.03,  "max": 509.75},
    "no2":  {"min": 0.01,  "max": 278.21},
    "o3":   {"min": 0.01,  "max": 66.17},
    "so2":  {"min": 0.01,  "max": 500.0},
}

# Delhi centroid in normalised coordinates.
# lat [27.5, 29.0] lon [76.5, 77.8]. Centroid ≈ 28.25N, 77.15E → (0.5, 0.5).
CENTROID_X = 0.5
CENTROID_Y = 0.5


# ── station selection ─────────────────────────────────────────────────────────
def pick_held_out_station(df: pd.DataFrame) -> str:
    """
    Return the station closest to Delhi centroid (most interior station).
    Interior means surrounded by other stations on all sides — strongest spatial
    interpolation test because the PINN has constraints nearby in every direction.
    """
    means = df.groupby("station")[["x_norm", "y_norm"]].mean().dropna()
    dist  = ((means["x_norm"] - CENTROID_X)**2 +
             (means["y_norm"] - CENTROID_Y)**2) ** 0.5
    chosen = dist.idxmin()
    x, y   = means.loc[chosen, ["x_norm", "y_norm"]]
    print(f"[LOO] Held-out station : '{chosen}'")
    print(f"[LOO]   x_norm={x:.4f}  y_norm={y:.4f}")
    print(f"[LOO]   dist from centroid = {dist[chosen]:.4f} (normalised)")
    return chosen


# ── tensor builder (same logic as notebook cell 4) ────────────────────────────
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
            print(f"[LOO] IC points: {len(ic_rows)}")

    return result


# ── normalization params ───────────────────────────────────────────────────────
def load_norm_params() -> dict:
    candidates = [
        SPLITS_DIR / "normalized_params.json",
        REPO / "data" / "processed" / "normalized_params.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                print(f"[LOO] Norm params loaded from {p}")
                return json.load(f)
    print("[LOO] normalized_params.json not found — using hardcoded fallback")
    return _NORM_PARAMS_FALLBACK


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # T4 16GB config — match notebook cell 2
    cfg.BATCH_SIZE          = 4096
    cfg.NUM_COLLOCATION     = 200_000
    cfg.CHECKPOINT_EVERY    = 2_500
    cfg.LBFGS_MAX_ITER      = 1_000
    cfg.LBFGS_DATA_CHUNK    = 20_000
    cfg.LBFGS_COLLOC_CHUNK  = 15_000
    cfg.CHECKPOINT_PREFIX   = OUTPUT_PREFIX

    import torch
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

    device = cfg.DEVICE
    print(f"[LOO] Device : {device}")
    print(f"[LOO] Output prefix : {OUTPUT_PREFIX}")

    # ── load raw data ──────────────────────────────────────────────────────────
    print("[LOO] Loading data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    if "station" not in train_df.columns:
        raise RuntimeError(
            "train.parquet has no 'station' column — cannot perform LOO experiment."
        )

    # ── select held-out station ────────────────────────────────────────────────
    held_out     = pick_held_out_station(train_df)
    held_out_df  = train_df[train_df["station"] == held_out].copy()
    train_loo_df = train_df[train_df["station"] != held_out].copy()

    # Remove from validation too (avoid data leakage in val metrics)
    if "station" in val_df.columns:
        val_loo_df = val_df[val_df["station"] != held_out].copy()
    else:
        val_loo_df = val_df

    print(f"[LOO] Train rows  : {len(train_df):,} → {len(train_loo_df):,} after removal")
    print(f"[LOO] Val rows    : {len(val_df):,} → {len(val_loo_df):,}")
    print(f"[LOO] Held-out    : {len(held_out_df):,} rows (never seen during training)")

    # Save station info so we can reference it later
    with open(CHECKPOINTS_DIR / "loo_station.json", "w") as f:
        json.dump({"held_out_station": held_out,
                   "held_out_rows": len(held_out_df)}, f, indent=2)

    # ── build tensors ──────────────────────────────────────────────────────────
    train_data = df_to_tensors(train_loo_df, include_ic=True)
    val_data   = df_to_tensors(val_loo_df)

    # ── train ──────────────────────────────────────────────────────────────────
    print(f"\n[LOO] Training fresh forward PINN (no WandB)...")
    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
    )
    trainer.train()

    # ── evaluate at held-out station ───────────────────────────────────────────
    print(f"\n[LOO] Evaluating at held-out station '{held_out}'...")
    trainer.model.eval()

    held_inputs  = torch.tensor(
        held_out_df[INPUT_COLS].fillna(0).values, dtype=torch.float32
    ).to(device)
    held_targets = held_out_df[TARGET_COLS].values.astype(float)
    held_mask    = held_out_df[TARGET_COLS].notna().values

    with torch.no_grad():
        held_pred = trainer.model(held_inputs).cpu().numpy()

    norm_params = load_norm_params()

    results = {"held_out_station": held_out, "pollutants": {}}

    print(f"\n{'Pollutant':<10} {'MAE (norm)':>12} {'R² (norm)':>10} "
          f"{'MAE (μg/m³)':>13} {'n':>8}")
    print("-" * 58)

    for i, key in enumerate(POLL_KEYS):
        valid = held_mask[:, i]
        if valid.sum() == 0:
            print(f"{key.upper():<10} {'—':>12} {'—':>10} {'—':>13} {'0':>8}")
            continue

        pred_n = held_pred[valid, i]
        tgt_n  = held_targets[valid, i]

        mae_n  = float(np.mean(np.abs(pred_n - tgt_n)))
        ss_res = float(np.sum((pred_n - tgt_n)**2))
        ss_tot = float(np.sum((tgt_n - tgt_n.mean())**2))
        r2_n   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Denormalise
        np_key = norm_params.get(key, _NORM_PARAMS_FALLBACK.get(key, {"min": 0, "max": 1}))
        vmin, vmax = np_key["min"], np_key["max"]
        mae_phys = float(np.mean(np.abs(
            pred_n * (vmax - vmin) + vmin - (tgt_n * (vmax - vmin) + vmin)
        )))

        print(f"{key.upper():<10} {mae_n:>12.4f} {r2_n:>10.4f} "
              f"{mae_phys:>13.2f} {valid.sum():>8}")

        results["pollutants"][key] = {
            "mae_normalised":  mae_n,
            "r2_normalised":   r2_n,
            "mae_ug_m3":       mae_phys,
            "n":               int(valid.sum()),
        }

    out_path = CHECKPOINTS_DIR / "loo_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[LOO] Results saved → {out_path.name}")

    # Summary verdict
    pm25_r2 = results["pollutants"].get("pm25", {}).get("r2_normalised", float("nan"))
    if pm25_r2 > 0.1:
        print(f"\n✓ SPATIAL GENERALISATION CONFIRMED: PM2.5 R²={pm25_r2:.4f} at "
              f"an unseen station. The PINN predicts at locations it was never trained on.")
    elif pm25_r2 > 0:
        print(f"\n~ WEAK SPATIAL GENERALISATION: PM2.5 R²={pm25_r2:.4f}. "
              f"Positive but low — physics constraint helps but physics alone is insufficient.")
    else:
        print(f"\n✗ SPATIAL GENERALISATION NOT CONFIRMED: PM2.5 R²={pm25_r2:.4f}. "
              f"Document this as a limitation — sparse sensors constrain the inverse problem.")

    print(f"[LOO] Done.")


if __name__ == "__main__":
    main()

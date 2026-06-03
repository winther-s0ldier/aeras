"""
train_ablation_pdeweight.py  —  v16 PHYSICS-OFF CONTROL
───────────────────────────────────────────────────────
THE central-claim ablation. This is v13 DA's exact twin with the physics
gradient turned OFF, so we can measure what the advection-diffusion PDE
actually contributes to PM2.5 forecast skill.

  v13 DA  = lag + physics (PDE ON )  → Diwali R² = 0.712   (already trained)
  v16     = lag + physics (PDE OFF)  → this run             (the control)

  gap (v13 − v16) = the TRUE physics contribution to PM2.5.

How "physics OFF" is done (clean + bulletproof):
  We keep EVERYTHING identical to train_da.py — same INPUT_DIM=8, same lag
  feature, same architecture, collocation, RAR, curriculum, optimizer,
  adaptive weighting — and ONLY set the PDE and BC loss weights to 0 on the
  already-constructed loss object. So data fidelity + IC + non-negativity
  still train the network; the advection-diffusion residual contributes zero
  gradient. The PDE residual is still *computed* (identical code path) but
  weighted out — guaranteeing the only difference vs v13 is the physics term.

  base_weights index map (see src/models/loss.py):
    [0]=data  [1]=PDE  [2]=BC  [3]=IC  [4]=sparsity  [5]=nonneg

Interpretation of the result:
  - v16 Diwali ≪ v13 0.712  → physics adds real skill. Headline survives.
  - v16 Diwali ≈ v13 0.712  → physics adds ~nothing to pointwise accuracy;
                              the lag (DA) does the work. Honest reframe:
                              physics' value is interpolation + source ID,
                              not accuracy at known stations.
  - v16 Diwali  >  v13 0.712 → physics slightly hurts. Also a clean finding.

DO NOT re-run v13. v13 is saved. This is its physics-off twin.

Run on GPU 0:
  CUDA_VISIBLE_DEVICES=0 python train_ablation_pdeweight.py
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

# ── must match v13 DA exactly, EXCEPT the checkpoint name ─────────────────────
cfg.INPUT_DIM         = 8
cfg.CHECKPOINT_PREFIX = "aeras_v16_noPDE"
cfg.BATCH_SIZE        = 4096
cfg.NUM_COLLOCATION   = 200_000
cfg.CHECKPOINT_EVERY  = 2_500
cfg.LBFGS_MAX_ITER    = 1_000
cfg.LBFGS_DATA_CHUNK  = 20_000
cfg.LBFGS_COLLOC_CHUNK = 15_000

from src.config import CHECKPOINTS_DIR
from src.training.trainer import AerasTrainer
from src.models.pinn import AerasPINN

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

# v13 DA reference (physics ON) — for side-by-side printing at the end
V13_REF = {
    "test_random": {"pm25": 0.444},
    "test_diwali": {"pm25": 0.712},
    "test_winter": {"pm25": 0.630},
}


def add_lag_feature(df: pd.DataFrame) -> pd.DataFrame:
    """PM2.5 lag-1h per station; first row per station → 0 (no target leak)."""
    df = df.sort_values(["station", "timestamp"]).copy()
    df["pm25_lag1h_norm"] = df.groupby(
        "station", observed=True
    )["pm25_norm"].shift(1).fillna(0.0)
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
            print(f"[NOPDE] IC points: {len(ic_rows)}")

    return result


def evaluate_test(trainer, name: str, device: str):
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[NOPDE] {name}.parquet not found, skipping")
        return None

    df = add_lag_feature(df)
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values, dtype=torch.float32).to(device)
    targets = df[TARGET_COLS].values.astype(float)
    mask    = df[TARGET_COLS].notna().values

    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(inputs).cpu().numpy()

    out = {}
    v13 = V13_REF.get(name, {})
    print(f"\n[{name}]")
    print(f"{'Poll':<6} {'R² (v16)':>9} {'R² (v13)':>9} {'Δ phys':>8} {'MAE μg/m³':>11} {'n':>8}")
    for i, key in enumerate(["pm25","no2","o3","so2"]):
        v = mask[:, i]
        if v.sum() == 0:
            continue
        p, t = pred[v, i], targets[v, i]
        mae_n  = float(np.mean(np.abs(p - t)))
        ss_r   = float(np.sum((p - t) ** 2))
        ss_t   = float(np.sum((t - t.mean()) ** 2))
        r2     = 1.0 - ss_r / ss_t if ss_t > 0 else float('nan')
        vmin, vmax = _NORM[key]["min"], _NORM[key]["max"]
        mae_phys = float(np.mean(np.abs(
            p * (vmax - vmin) + vmin - (t * (vmax - vmin) + vmin)
        )))
        ref = v13.get(key)
        delta_str = f"{(ref - r2):+.3f}" if ref is not None else "  —  "
        ref_str   = f"{ref:.4f}" if ref is not None else "  —  "
        print(f"{key.upper():<6} {r2:>9.4f} {ref_str:>9} {delta_str:>8} {mae_phys:>11.2f} {int(v.sum()):>8}")
        out[key] = {"mae_norm": mae_n, "r2": r2, "mae_ug_m3": mae_phys, "n": int(v.sum())}
    return out


def main():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

    device = cfg.DEVICE
    print(f"[NOPDE] Device : {device}")
    print(f"[NOPDE] INPUT_DIM : {cfg.INPUT_DIM}  (same as v13 DA)")
    print(f"[NOPDE] Prefix : {cfg.CHECKPOINT_PREFIX}")

    # Verify INPUT_DIM propagated into the model before training
    _probe = AerasPINN()
    assert _probe.fourier.B.shape[0] == 8, (
        f"AerasPINN built with input_dim={_probe.fourier.B.shape[0]}, expected 8."
    )
    del _probe

    print("\n[NOPDE] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    print("[NOPDE] Computing PM2.5 lag-1h feature...")
    train_df = add_lag_feature(train_df)
    val_df   = add_lag_feature(val_df)

    train_data = df_to_tensors(train_df, include_ic=True)
    val_data   = df_to_tensors(val_df)

    print("\n[NOPDE] Building trainer (identical to v13 DA)...")
    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
    )

    # ── THE ablation knob: zero the physics-residual weights ─────────────────
    # base_weights = [data, PDE, BC, IC, sparsity, nonneg]
    before = trainer.loss_fn.base_weights.tolist()
    trainer.loss_fn.base_weights[1] = 0.0   # PDE residual OFF
    trainer.loss_fn.base_weights[2] = 0.0   # BC (no-flux) OFF — also physics
    after = trainer.loss_fn.base_weights.tolist()
    print(f"[NOPDE] base_weights before: {['%.3f'%w for w in before]}")
    print(f"[NOPDE] base_weights after : {['%.3f'%w for w in after]}")
    assert trainer.loss_fn.base_weights[1].item() == 0.0, "PDE weight not zeroed!"
    assert trainer.loss_fn.base_weights[2].item() == 0.0, "BC weight not zeroed!"
    print("[NOPDE] Physics gradient DISABLED. Data + IC + non-negativity only.\n")

    trainer.train()

    print("\n" + "=" * 64)
    print("[NOPDE] FINAL EVALUATION — v16 (physics OFF) vs v13 DA (physics ON)")
    print("=" * 64)
    print("Δ phys = v13 − v16  (positive = physics helped; ~0 = lag did the work)")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = evaluate_test(trainer, split, device)
        if r:
            all_results[split] = r

    # Headline verdict on PM2.5 Diwali
    if "test_diwali" in all_results and "pm25" in all_results["test_diwali"]:
        v16_diwali = all_results["test_diwali"]["pm25"]["r2"]
        delta = V13_REF["test_diwali"]["pm25"] - v16_diwali
        print("\n" + "-" * 64)
        print(f"[VERDICT] PM2.5 Diwali  v13={V13_REF['test_diwali']['pm25']:.3f}  "
              f"v16={v16_diwali:.3f}  Δphysics={delta:+.3f}")
        if delta > 0.05:
            print("  → Physics adds real skill. 'Physics helps' claim SURVIVES.")
        elif delta > -0.02:
            print("  → Physics adds ~nothing to accuracy. Lag (DA) does the work.")
            print("    Honest reframe: physics' value is interpolation + source ID.")
        else:
            print("  → Physics slightly HURTS pointwise accuracy. Clean finding.")
        print("-" * 64)

    out_path = CHECKPOINTS_DIR / "noPDE_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "v13_reference": V13_REF}, f, indent=2)
    print(f"\n[NOPDE] Results saved → {out_path.name}")
    print("[NOPDE] Done.")


if __name__ == "__main__":
    main()

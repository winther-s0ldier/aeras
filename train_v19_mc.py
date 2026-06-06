"""
train_v19_mc.py  —  MC Dropout Uncertainty Quantification
──────────────────────────────────────────────────────────
Adds Monte Carlo Dropout to the v19 DA4 setup for uncertainty estimation.
Same data, same INPUT_DIM=11, same loss — only change is Dropout(0.15)
inserted after every hidden layer.

At inference: model.train() keeps dropout ACTIVE, 50 stochastic forward
passes give a distribution: mean = final prediction, std = uncertainty.

Uncertainty metrics:
    PICP90: fraction of true values inside 90% CI (target: ~0.90)
    MeanUnc: mean std across test points (normalised units)

Runs on GPU 1 while v20 Satellite runs on GPU 0.
"""

import os, sys, json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import src.config as cfg

cfg.INPUT_DIM         = 11
cfg.CHECKPOINT_PREFIX = "aeras_v19_mc"
cfg.BATCH_SIZE        = 4096
cfg.NUM_COLLOCATION   = 200_000
cfg.CHECKPOINT_EVERY  = 2_500
cfg.LBFGS_MAX_ITER    = 1_000
cfg.LBFGS_DATA_CHUNK  = 20_000
cfg.LBFGS_COLLOC_CHUNK = 15_000

from src.config import (
    CHECKPOINTS_DIR, POLLUTANTS, DEVICE,
    HIDDEN_LAYERS, HIDDEN_DIM, OUTPUT_DIM, LEARNING_RATE,
)
from src.training.trainer import AerasTrainer
from src.models.pinn import AerasPINN

SPLITS_DIR   = Path("/kaggle/input/datasets/b1042rudrakumar/kaggle-dataset-fixed")
MC_PASSES    = 50
DROPOUT_RATE = 0.15

INPUT_COLS = [
    "x_norm", "y_norm", "t_norm",
    "u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm",
    "pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm", "so2_lag1h_norm",
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


# ── MC Dropout PINN ──────────────────────────────────────────────────────────
class MCDropoutPINN(AerasPINN):
    """AerasPINN with Dropout(p) after every hidden layer."""
    def __init__(self, dropout_rate: float = DROPOUT_RATE, **kwargs):
        super().__init__(**kwargs)
        layers = []
        in_dim = self.fourier.output_dim
        for _ in range(HIDDEN_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_DIM))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = HIDDEN_DIM
        layers.append(nn.Linear(HIDDEN_DIM, OUTPUT_DIM))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def mc_predict(self, x: torch.Tensor, n: int = MC_PASSES):
        """N stochastic passes -> (mean, std) each [N_data, 4]."""
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n):
                preds.append(self(x).cpu().numpy())
        stack = np.stack(preds, axis=0)
        return stack.mean(axis=0), stack.std(axis=0)


# ── Data helpers ─────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
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
            print(f"[V19MC] IC points: {len(ic_rows)}")
    return result


def mc_evaluate(model: MCDropoutPINN, name: str, device: str) -> dict:
    try:
        df = pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
    except FileNotFoundError:
        print(f"[V19MC] {name}.parquet not found, skipping")
        return None

    df = add_lag_features(df)
    inputs  = torch.tensor(df[INPUT_COLS].fillna(0).values,
                           dtype=torch.float32).to(device)
    targets = df[TARGET_COLS].values.astype(float)
    mask    = df[TARGET_COLS].notna().values

    print(f"\n[{name}] Running {MC_PASSES} MC passes...")
    mean_pred, std_pred = model.mc_predict(inputs, n=MC_PASSES)

    out = {}
    key_label = ("diwali" if "diwali" in name
                 else "winter" if "winter" in name else None)
    ref = V19_REF.get(key_label, {})
    print(f"{'Poll':<6} {'R2':>8} {'MAE':>8} {'PICP90':>8} {'MeanUnc':>9}  vs v19")

    for i, key in enumerate(["pm25", "no2", "o3", "so2"]):
        v = mask[:, i]
        if v.sum() == 0:
            continue
        p_mean, p_std = mean_pred[v, i], std_pred[v, i]
        t = targets[v, i]
        ss_r = float(np.sum((p_mean - t) ** 2))
        ss_t = float(np.sum((t - t.mean()) ** 2))
        r2   = 1.0 - ss_r / ss_t if ss_t > 0 else float("nan")
        vmin, vmax = _PHYS_NORM[key]
        mae_phys   = float(np.mean(np.abs(
            p_mean * (vmax - vmin) + vmin - (t * (vmax - vmin) + vmin)
        )))
        z    = 1.645
        picp = float(np.mean((t >= p_mean - z * p_std) & (t <= p_mean + z * p_std)))
        unc  = float(p_std.mean())
        delta = f"  {r2 - ref[key]:+.4f}" if key in ref else ""
        print(f"{key.upper():<6} {r2:>8.4f} {mae_phys:>8.2f} {picp:>8.3f} {unc:>9.5f}{delta}")
        out[key] = {
            "r2": r2, "mae_ug_m3": mae_phys,
            "picp_90": picp, "mean_unc_norm": unc,
            "n": int(v.sum()),
        }
    return out


def main():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    device = DEVICE

    print(f"[V19MC] Device      : {device}")
    print(f"[V19MC] INPUT_DIM   : {cfg.INPUT_DIM}")
    print(f"[V19MC] Dropout     : {DROPOUT_RATE}")
    print(f"[V19MC] MC passes   : {MC_PASSES}")
    print(f"[V19MC] Prefix      : {cfg.CHECKPOINT_PREFIX}")
    print(f"[V19MC] Data dir    : {SPLITS_DIR}")

    print("\n[V19MC] Loading training data...")
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")
    train_df = add_lag_features(train_df)
    val_df   = add_lag_features(val_df)
    train_data = df_to_tensors(train_df, include_ic=True)
    val_data   = df_to_tensors(val_df)

    trainer = AerasTrainer(
        train_data=train_data,
        val_data=val_data,
        inverse_mode=False,
        use_wandb=False,
    )
    mc_model = MCDropoutPINN(dropout_rate=DROPOUT_RATE).to(device)
    trainer.model = mc_model
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    trainer.optimizer = Adam(
        list(mc_model.parameters()) + list(trainer.loss_fn.parameters()),
        lr=LEARNING_RATE,
    )
    trainer.scheduler = CosineAnnealingWarmRestarts(
        trainer.optimizer, T_0=10000, T_mult=2, eta_min=1e-6
    )

    print(f"[V19MC] MC model params: {sum(p.numel() for p in mc_model.parameters()):,}")
    print("[V19MC] Training MCDropoutPINN...")
    trainer.train()

    print("\n" + "=" * 64)
    print("[V19MC] FINAL MC EVALUATION")
    print("=" * 64)
    print("PICP90 target: ~0.90 (well-calibrated model)")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        r = mc_evaluate(mc_model, split, device)
        if r:
            all_results[split] = r

    payload = {
        "version": "v19_mc_dropout",
        "input_dim": cfg.INPUT_DIM,
        "dropout_rate": DROPOUT_RATE,
        "mc_passes": MC_PASSES,
        "v19_ref": V19_REF,
        "results": all_results,
    }
    out_path = CHECKPOINTS_DIR / "v19_mc_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[V19MC] Results saved -> {out_path.name}")
    print("[V19MC] Done.")


if __name__ == "__main__":
    main()

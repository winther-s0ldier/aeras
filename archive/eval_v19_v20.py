"""
eval_v19_v20.py  —  Fair side-by-side comparison on the SAME splits
─────────────────────────────────────────────────────────────────────
Evaluates both v19 and v20 checkpoints on the current local splits
(post-satellite-merge). This is the only fair comparison since the
Kaggle eval ran v20 on different split sizes than v19 was trained on.
"""

import sys, json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

SPLITS_DIR  = REPO / "data" / "splits"
CKPT_V19    = REPO / "checkpoints" / "aeras_v19_da4_final.pt"
CKPT_V20    = REPO / "checkpoints" / "v20" / "aeras_v20_satellite_final.pt"

_PHYS_NORM = {
    "pm25": (0.03, 509.75), "no2": (0.01, 278.21),
    "o3":   (0.01,  66.17), "so2": (0.01, 500.00),
}

INPUT_COLS_V19 = [
    "x_norm", "y_norm", "t_norm",
    "u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm",
    "pm25_lag1h_norm", "no2_lag1h_norm", "o3_lag1h_norm", "so2_lag1h_norm",
]
INPUT_COLS_V20 = INPUT_COLS_V19 + ["aer_ai_norm"]
TARGET_COLS    = ["pm25_norm", "no2_norm", "o3_norm", "so2_norm"]


def add_lag_features(df):
    df = df.sort_values(["station", "timestamp"]).copy()
    g  = df.groupby("station", observed=True)
    df["pm25_lag1h_norm"] = g["pm25_norm"].shift(1).fillna(0.0)
    df["no2_lag1h_norm"]  = g["no2_norm"].shift(1).fillna(0.0)
    df["o3_lag1h_norm"]   = g["o3_norm"].shift(1).fillna(0.0)
    df["so2_lag1h_norm"]  = g["so2_norm"].shift(1).fillna(0.0)
    return df


def load_model(ckpt_path, input_dim):
    import src.config as cfg
    cfg.INPUT_DIM = input_dim
    # Re-import to pick up new INPUT_DIM
    import importlib
    import src.models.pinn as pinn_mod
    importlib.reload(pinn_mod)
    model = pinn_mod.AerasPINN()
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate(model, df, input_cols):
    if "aer_ai_norm" in input_cols:
        if "aer_ai_norm" not in df.columns:
            df["aer_ai_norm"] = 0.0
        df["aer_ai_norm"] = df["aer_ai_norm"].fillna(0.0)

    inputs  = torch.tensor(df[input_cols].fillna(0).values, dtype=torch.float32)
    targets = df[TARGET_COLS].values.astype(float)
    mask    = df[TARGET_COLS].notna().values

    with torch.no_grad():
        pred = model(inputs).numpy()

    results = {}
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
            p * (vmax-vmin) + vmin - (t * (vmax-vmin) + vmin)
        )))
        results[key] = {"r2": r2, "mae": mae, "n": int(v.sum())}
    return results


def print_comparison(split_name, r19, r20):
    print(f"\n{'-'*58}")
    print(f"  {split_name}")
    print(f"{'-'*58}")
    print(f"{'Poll':<6} {'v19 R2':>8} {'v20 R2':>8} {'Delta':>8}  "
          f"{'v19 MAE':>9} {'v20 MAE':>9}")
    for key in ["pm25", "no2", "o3", "so2"]:
        if key not in r19 or key not in r20:
            continue
        d = r20[key]["r2"] - r19[key]["r2"]
        sign = "+" if d >= 0 else ""
        print(f"{key.upper():<6} {r19[key]['r2']:>8.4f} {r20[key]['r2']:>8.4f} "
              f"{sign}{d:>7.4f}  "
              f"{r19[key]['mae']:>9.2f} {r20[key]['mae']:>9.2f}")


def main():
    device = "cpu"  # local eval, no GPU needed

    print("Loading checkpoints...")
    print(f"  v19: {CKPT_V19}")
    print(f"  v20: {CKPT_V20}")

    model_v19 = load_model(CKPT_V19, input_dim=11)
    model_v20 = load_model(CKPT_V20, input_dim=12)
    print("Checkpoints loaded.")

    all_results = {}
    for split in ["test_random", "test_diwali", "test_winter"]:
        path = SPLITS_DIR / f"{split}.parquet"
        if not path.exists():
            print(f"Skipping {split} (not found)")
            continue

        df = pd.read_parquet(path)
        df = add_lag_features(df)

        sat_cov = (df["aer_ai_norm"].fillna(0) > 0).mean() * 100 if "aer_ai_norm" in df.columns else 0
        print(f"\n[{split}]  rows={len(df):,}  aer_ai coverage={sat_cov:.1f}%")

        r19 = evaluate(model_v19, df.copy(), INPUT_COLS_V19)
        r20 = evaluate(model_v20, df.copy(), INPUT_COLS_V20)
        print_comparison(split, r19, r20)
        all_results[split] = {"v19": r19, "v20": r20}

    out = REPO / "checkpoints" / "v19_v20_comparison.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out.name}")


if __name__ == "__main__":
    main()

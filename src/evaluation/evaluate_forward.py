import sys
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import SPLITS_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, DEVICE, CHECKPOINT_PREFIX
from src.models.pinn import AerasPINN
from src.evaluation.metrics import (
    compute_all_forward_metrics,
    sparse_sensor_evaluation,
)


def load_test_data(split_name: str) -> dict:
    path = SPLITS_DIR / f"{split_name}.parquet"
    if not path.exists():
        print(f"[EVAL] {path} not found.")
        return None

    df = pd.read_parquet(path)


    input_cols = ["x_norm", "y_norm", "t_norm",
                  "u_wind_norm", "v_wind_norm",
                  "temp_norm", "blh_norm"]
    target_cols = ["pm25_norm", "no2_norm", "o3_norm", "so2_norm"]

    available_inputs = [c for c in input_cols if c in df.columns]
    if len(available_inputs) < 3:
        print(f"[EVAL] Missing columns. Available: {df.columns.tolist()}")
        return None


    for col in ["u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm"]:
        if col not in df.columns:
            df[col] = 0.0

    inputs = torch.tensor(df[input_cols].values, dtype=torch.float32)
    targets = torch.tensor(df[target_cols].values, dtype=torch.float32)  # [N, 4]
    mask = torch.tensor(df[target_cols].notna().values, dtype=torch.float32)  # [N, 4]

    return {"inputs": inputs, "targets": targets, "mask": mask}


def evaluate_all(checkpoint_name: str = "final"):

    print("  AerasPINN Forward Evaluation")

    model = AerasPINN().to(DEVICE)
    ckpt_path = CHECKPOINTS_DIR / f"{CHECKPOINT_PREFIX}_{checkpoint_name}.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"[EVAL] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[EVAL] Checkpoint not found: {ckpt_path}")
        return


    norm_path = PROCESSED_DIR / "normalized_params.json"
    norm_params = None
    if norm_path.exists():
        with open(norm_path) as f:
            norm_params = json.load(f)


    test_sets = {
        "random": "test_random",
        "diwali_2019": "test_diwali",
        "winter_2019": "test_winter",
        "full_test": "test",
    }

    all_results = {}
    model.eval()

    POLLUTANT_NAMES = ["pm25", "no2", "o3", "so2"]
    DENORM_KEYS = {"pm25": "pm25", "no2": "no2", "o3": "o3", "so2": "so2"}

    for name, split in test_sets.items():
        data = load_test_data(split)
        if data is None:
            continue

        inputs = data["inputs"].to(DEVICE)
        targets = data["targets"].numpy()  # [N, 4]
        mask = data["mask"].numpy()  # [N, 4]

        with torch.no_grad():
            pred = model(inputs).cpu().numpy()  # [N, 4]

        all_results[name] = {}
        print(f"\n[{name}]")
        for i, poll in enumerate(POLLUTANT_NAMES):
            valid = mask[:, i].astype(bool)
            if valid.sum() < 100:
                print(f"  {poll}: <100 valid samples, skipping")
                continue

            p = pred[valid, i]
            t = targets[valid, i]

            # Denormalize using saved params
            if norm_params and DENORM_KEYS[poll] in norm_params:
                pmin = norm_params[DENORM_KEYS[poll]]["min"]
                pmax = norm_params[DENORM_KEYS[poll]]["max"]
                p_denorm = p * (pmax - pmin) + pmin
                t_denorm = t * (pmax - pmin) + pmin
            else:
                p_denorm = p
                t_denorm = t

            metrics = compute_all_forward_metrics(p_denorm, t_denorm, norm_params=None)
            all_results[name][poll] = metrics

            print(f"  {poll.upper()}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f} (n={valid.sum()})")


    print("Sparse Sensor Degradation Test")

    full_test = load_test_data("test")
    if full_test:
        sparse_results = sparse_sensor_evaluation(model, full_test, device=DEVICE)
        all_results["sparse_sensor"] = {
            str(k): v for k, v in sparse_results.items()
        }


    results_path = CHECKPOINTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[EVAL] Results saved to {results_path}")


    def _param_to_list(p):
        if p.numel() == 1:
            return [p.flatten()[0].item()]
        return [p[i].item() for i in range(p.numel())]

    POLLUTANT_NAMES = ["pm25", "no2", "o3", "so2"]
    dx_vals = _param_to_list(model.Dx)
    dy_vals = _param_to_list(model.Dy)
    lam_vals = _param_to_list(model.lambda_dep)

    print("Learned Physics Parameters")
    if len(dx_vals) == 1:
        print(f"  Dx (diffusion x): {dx_vals[0]:.6f}")
        print(f"  Dy (diffusion y): {dy_vals[0]:.6f}")
        print(f"  lambda (deposition): {lam_vals[0]:.6f}")
    else:
        for i in range(len(dx_vals)):
            poll = POLLUTANT_NAMES[i] if i < len(POLLUTANT_NAMES) else f"out{i}"
            print(f"  {poll.upper()}: Dx={dx_vals[i]:.6f}, Dy={dy_vals[i]:.6f}, lambda_dep={lam_vals[i]:.6f}")

    return all_results


if __name__ == "__main__":
    evaluate_all()

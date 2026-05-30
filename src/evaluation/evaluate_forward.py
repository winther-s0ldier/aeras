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
    target_col = "pm25_norm"

    available_inputs = [c for c in input_cols if c in df.columns]
    if len(available_inputs) < 3 or target_col not in df.columns:
        print(f"[EVAL] Missing columns. Available: {df.columns.tolist()}")
        return None


    for col in ["u_wind_norm", "v_wind_norm", "temp_norm", "blh_norm"]:
        if col not in df.columns:
            df[col] = 0.0


    df = df.dropna(subset=[target_col])

    inputs = torch.tensor(df[input_cols].values, dtype=torch.float32)
    targets = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

    return {"inputs": inputs, "targets": targets}


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

    for name, split in test_sets.items():
        data = load_test_data(split)
        if data is None:
            continue

        inputs = data["inputs"].to(DEVICE)
        targets = data["targets"]

        with torch.no_grad():
            pred = model(inputs).cpu()

        metrics = compute_all_forward_metrics(
            pred.numpy().flatten(),
            targets.numpy().flatten(),
            norm_params=norm_params,
        )

        all_results[name] = metrics
        print(f"\n[{name}]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


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


    print("Learned Physics Parameters")

    print(f"  Dx (diffusion x): {model.Dx.item():.6f}")
    print(f"  Dy (diffusion y): {model.Dy.item():.6f}")
    print(f"  lambda (deposition):  {model.lambda_dep.item():.6f}")

    return all_results


if __name__ == "__main__":
    evaluate_all()

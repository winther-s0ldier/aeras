import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CHECKPOINTS_DIR


ABLATION_CONFIGS = {
    "full": {
        "description": "VayuPINN with all components (baseline)",
        "w_pde": 0.1,
        "w_data": 1.0,
        "w_bc": 0.01,
        "w_ic": 0.1,
        "fourier_features": 64,
        "curriculum": True,
        "adaptive_weights": True,
    },
    "no_pde": {
        "description": "Remove PDE loss — pure data-driven regression",
        "w_pde": 0.0,
        "w_data": 1.0,
        "w_bc": 0.0,
        "w_ic": 0.0,
        "fourier_features": 64,
        "curriculum": False,
        "adaptive_weights": True,
    },
    "no_data": {
        "description": "Remove data loss — pure physics simulation",
        "w_pde": 1.0,
        "w_data": 0.0,
        "w_bc": 0.1,
        "w_ic": 1.0,
        "fourier_features": 64,
        "curriculum": False,
        "adaptive_weights": True,
    },
    "no_fourier": {
        "description": "Remove Fourier features — test spectral bias fix",
        "w_pde": 0.1,
        "w_data": 1.0,
        "w_bc": 0.01,
        "w_ic": 0.1,
        "fourier_features": 0,
        "curriculum": True,
        "adaptive_weights": True,
    },
    "no_curriculum": {
        "description": "No curriculum — all losses from epoch 0",
        "w_pde": 0.1,
        "w_data": 1.0,
        "w_bc": 0.01,
        "w_ic": 0.1,
        "fourier_features": 64,
        "curriculum": False,
        "adaptive_weights": True,
    },
    "no_adaptive": {
        "description": "Fixed loss weights — no adaptive balancing",
        "w_pde": 0.1,
        "w_data": 1.0,
        "w_bc": 0.01,
        "w_ic": 0.1,
        "fourier_features": 64,
        "curriculum": True,
        "adaptive_weights": False,
    },
}


def print_ablation_table(results: Dict[str, Dict[str, float]]):
    print("\n## Ablation Study Results\n")
    header = "| Config | MAE | RMSE | Diwali MAE | Sparse MAE (50%) | Physics Residual |"
    sep = "|---|---|---|---|---|---|"
    print(header)
    print(sep)

    for name, metrics in results.items():
        config = ABLATION_CONFIGS.get(name, {})
        desc = config.get("description", name)
        row = (
            f"| {desc} | "
            f"{metrics.get('mae', 'N/A'):.2f} | "
            f"{metrics.get('rmse', 'N/A'):.2f} | "
            f"{metrics.get('diwali_mae', 'N/A'):.2f} | "
            f"{metrics.get('sparse_mae_50', 'N/A'):.2f} | "
            f"{metrics.get('physics_residual', 'N/A'):.4f} |"
        )
        print(row)


if __name__ == "__main__":
    print("Ablation study configurations:")
    for name, config in ABLATION_CONFIGS.items():
        print(f"\n  [{name}]: {config['description']}")
    print("\nRun each config through the trainer to generate results.")
    print("See experiments/ directory for individual experiment scripts.")

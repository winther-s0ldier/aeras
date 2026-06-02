import sys
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CHECKPOINTS_DIR, RAW_DIR, DEVICE, CHECKPOINT_PREFIX
from src.models.pinn import AerasPINN
from src.models.source_net import SourceNet, SourceGrid
from src.evaluation.metrics import source_location_error, source_magnitude_error


def load_edgar_baseline(resolution: int = 50) -> np.ndarray:
    edgar_file_raw = RAW_DIR / "edgar" / "edgar_delhi_pm25.npy"
    edgar_file_ckpt = CHECKPOINTS_DIR / "edgar_delhi_pm25.npy"

    if edgar_file_raw.exists():
        edgar_file = edgar_file_raw
    elif edgar_file_ckpt.exists():
        edgar_file = edgar_file_ckpt
    else:
        edgar_file = None

    if edgar_file is not None:
        edgar = np.load(edgar_file)

        from scipy.ndimage import zoom
        scale_y = resolution / edgar.shape[0]
        scale_x = resolution / edgar.shape[1]
        return zoom(edgar, (scale_y, scale_x))

    print("[EDGAR] Preprocessed file not found. Generating synthetic baseline.")
    print(f"  Expected: {edgar_file}")
    print("  Download EDGAR v8.0 PM2.5 gridded emissions from:")
    print("  https://edgar.jrc.ec.europa.eu/")


    synthetic = np.zeros((resolution, resolution))

    synthetic[30:35, 35:40] = 0.8

    synthetic[25:30, 20:25] = 0.6

    synthetic[20:25, 25:30] = 0.4

    synthetic[15:40, 15:40] += 0.1

    return synthetic


def evaluate_inverse(
    checkpoint_name: str = "final",
    source_type: str = "net",
    resolution: int = 50,
):

    print("  aeras Inverse Problem Evaluation")

    ckpt_path = CHECKPOINTS_DIR / f"{CHECKPOINT_PREFIX}_{checkpoint_name}.pt"
    if not ckpt_path.exists():
        print(f"[EVAL] Checkpoint not found: {ckpt_path}")
        return

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if "source_model" not in state:
        print("[EVAL] No source model in checkpoint. Was inverse mode used?")
        return


    if source_type == "grid":
        source_model = SourceGrid().to(DEVICE)
    else:
        source_model = SourceNet().to(DEVICE)
    source_model.load_state_dict(state["source_model"])
    source_model.eval()


    time_points = [0.25, 0.5, 0.75]
    pred_maps = {}

    for t in time_points:
        if isinstance(source_model, SourceGrid):
            t_idx = int(t * (source_model.nt - 1))
            source_map = source_model.get_source_map(t_idx)
        else:
            source_map = source_model.get_source_map(
                t_value=t, resolution=resolution, device=DEVICE
            )
        pred_maps[t] = source_map


    edgar_map = load_edgar_baseline(resolution)


    results = {}
    for t, pred_map in pred_maps.items():
        loc_error = source_location_error(pred_map, edgar_map)
        mag_error = source_magnitude_error(pred_map, edgar_map)

        results[f"t={t}"] = {
            "location_error_km": loc_error,
            "magnitude_error": mag_error,
        }
        print(f"\n[t={t:.2f}] Location error: {loc_error:.1f} km | "
              f"Magnitude error: {mag_error:.2%}")


    results_path = CHECKPOINTS_DIR / "inverse_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EVAL] Results saved to {results_path}")


    maps_path = CHECKPOINTS_DIR / "source_maps.npz"
    np.savez(maps_path, **{f"t_{t}": m for t, m in pred_maps.items()}, edgar=edgar_map)
    print(f"[EVAL] Source maps saved to {maps_path}")

    return results


if __name__ == "__main__":
    evaluate_inverse()

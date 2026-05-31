import torch
import numpy as np
from typing import Dict, Optional


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    denominator = (np.abs(pred) + np.abs(true)) / 2.0
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    return float(np.mean(np.abs(pred - true) / denominator) * 100)


def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def physics_residual_norm(
    model,
    collocation_inputs: torch.Tensor,
    compute_residual_fn,
    device: str = "cpu",
) -> float:
    model.eval()
    with torch.no_grad():
        inputs = collocation_inputs.to(device).requires_grad_(True)


    residual = compute_residual_fn(model, inputs)
    return float(residual.abs().mean().item())


def compute_all_forward_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    norm_params: Optional[dict] = None,
) -> Dict[str, float]:
    if norm_params is not None:
        pm_min = norm_params["pm25"]["min"]
        pm_max = norm_params["pm25"]["max"]
        pred = pred * (pm_max - pm_min) + pm_min
        true = true * (pm_max - pm_min) + pm_min

    return {
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "smape": smape(pred, true),
        "r2": r_squared(pred, true),
    }


def source_location_error(
    pred_sources: np.ndarray,
    true_sources: np.ndarray,
    lat_range: tuple = (27.5, 29.0),
    lon_range: tuple = (76.5, 77.8),
) -> float:

    from scipy.ndimage import label

    def find_peaks(field, threshold_frac=0.5):
        threshold = field.max() * threshold_frac
        binary = field > threshold
        labeled, n_features = label(binary)
        peaks = []
        for i in range(1, n_features + 1):
            region = np.where(labeled == i)
            cy = np.mean(region[0]) / field.shape[0]
            cx = np.mean(region[1]) / field.shape[1]
            peaks.append((cy, cx))
        return peaks

    pred_peaks = find_peaks(pred_sources)
    true_peaks = find_peaks(true_sources)

    if not pred_peaks or not true_peaks:
        return float("nan")


    def norm_to_latlon(peaks):
        return [
            (p[0] * (lat_range[1] - lat_range[0]) + lat_range[0],
             p[1] * (lon_range[1] - lon_range[0]) + lon_range[0])
            for p in peaks
        ]

    pred_ll = norm_to_latlon(pred_peaks)
    true_ll = norm_to_latlon(true_peaks)


    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon / 2) ** 2)
        return R * 2 * np.arcsin(np.sqrt(a))


    errors = []
    for plat, plon in pred_ll:
        min_dist = min(haversine(plat, plon, tlat, tlon) for tlat, tlon in true_ll)
        errors.append(min_dist)

    return float(np.mean(errors))


def source_magnitude_error(
    pred_sources: np.ndarray,
    true_sources: np.ndarray,
) -> float:
    pred_total = pred_sources.sum()
    true_total = true_sources.sum()
    if true_total < 1e-8:
        return float("nan")
    return float(abs(pred_total - true_total) / true_total)


def sparse_sensor_evaluation(
    model,
    full_data: Dict[str, torch.Tensor],
    dropout_rates: list = None,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[float, Dict[str, float]]:
    if dropout_rates is None:
        dropout_rates = [0.0, 0.25, 0.50, 0.75]

    rng = np.random.default_rng(seed)
    results = {}

    inputs = full_data["inputs"]
    targets = full_data["targets"]

    for rate in dropout_rates:
        n_total = len(inputs)
        n_keep = int(n_total * (1 - rate))
        keep_idx = rng.choice(n_total, n_keep, replace=False)
        keep_idx.sort()

        sparse_inputs = inputs[keep_idx].to(device)
        # .cpu() required — targets may be GPU-resident; .numpy() fails on CUDA tensors
        sparse_targets = targets[keep_idx].cpu().numpy()

        model.eval()
        with torch.no_grad():
            pred = model(sparse_inputs).cpu().numpy()

        # Extract PM2.5 channel (index 0) only — consistent with Run F sparse baseline.
        # Flattening all 4 channels together would mix pollutant scales and corrupt metrics.
        if pred.ndim == 2 and pred.shape[1] > 1:
            pred_pm25 = pred[:, 0]
            true_pm25 = sparse_targets[:, 0]
        else:
            pred_pm25 = pred.flatten()
            true_pm25 = sparse_targets.flatten()

        # Mask out NaN rows — targets may have NaN for stations missing PM2.5 readings
        valid = ~np.isnan(true_pm25)
        pred_pm25 = pred_pm25[valid]
        true_pm25 = true_pm25[valid]

        metrics = compute_all_forward_metrics(pred_pm25, true_pm25)
        metrics["n_points"] = n_keep
        results[rate] = metrics

        print(f"[SPARSE] Dropout {rate:.0%}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

    return results

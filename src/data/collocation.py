import numpy as np
import torch
from typing import Optional


def uniform_collocation(n_points: int, dim: int = 3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n_points, dim)).astype(np.float32)


def latin_hypercube(n_points: int, dim: int = 3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.zeros((n_points, dim), dtype=np.float32)

    for d in range(dim):

        intervals = np.linspace(0, 1, n_points + 1)
        for i in range(n_points):
            points[i, d] = rng.uniform(intervals[i], intervals[i + 1])

        rng.shuffle(points[:, d])

    return points


def residual_adaptive_resample(
    model: torch.nn.Module,
    current_points: torch.Tensor,
    compute_residual_fn,
    n_new: int = 5000,
    top_fraction: float = 0.3,
    noise_std: float = 0.02,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()


    with torch.enable_grad():
        residual = compute_residual_fn(model, current_points.to(device))
        residual_mag = residual.detach().abs().squeeze()


    n_top = int(len(current_points) * top_fraction)
    top_idx = residual_mag.argsort(descending=True)[:n_top]
    top_points = current_points[top_idx].cpu()


    n_perturbed = n_new // 2
    selected = top_points[np.random.choice(len(top_points), n_perturbed, replace=True)]
    perturbed = selected + torch.randn_like(selected) * noise_std
    perturbed = perturbed.clamp(0, 1)


    n_random = n_new - n_perturbed
    fresh = torch.rand(n_random, current_points.shape[1])

    new_points = torch.cat([perturbed, fresh], dim=0)
    model.train()

    return new_points

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
    chunk_size: int = 10_000,
) -> torch.Tensor:
    model.eval()

    # Compute residuals in chunks to avoid OOM on large collocation pools
    residual_chunks = []
    for i in range(0, len(current_points), chunk_size):
        chunk = current_points[i: i + chunk_size].to(device)
        with torch.enable_grad():
            res = compute_residual_fn(model, chunk)
            mag = res.detach().abs()
            if mag.dim() > 1 and mag.shape[1] > 1:
                mag = mag.mean(dim=1)
            residual_chunks.append(mag.squeeze().cpu())
    residual_mag = torch.cat(residual_chunks, dim=0)



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

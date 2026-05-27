import torch
import numpy as np
from typing import Tuple


def time_marching_curriculum(
    total_time_range: Tuple[float, float] = (0.0, 1.0),
    n_phases: int = 4,
    current_phase: int = 0,
) -> Tuple[float, float]:
    t_min, t_max = total_time_range
    phase_frac = (current_phase + 1) / n_phases
    t_current_max = t_min + (t_max - t_min) * phase_frac
    return t_min, t_current_max


def causal_weighting(
    t_values: torch.Tensor,
    epsilon: float = 1.0,
    pde_residuals: torch.Tensor = None,
) -> torch.Tensor:
    if pde_residuals is not None:

        sorted_idx = t_values.argsort()
        sorted_residuals = pde_residuals[sorted_idx]


        cumsum = torch.cumsum(sorted_residuals ** 2, dim=0)
        weights = torch.exp(-epsilon * cumsum)


        unsorted_weights = torch.zeros_like(weights)
        unsorted_weights[sorted_idx] = weights
        return unsorted_weights
    else:

        return torch.exp(-epsilon * t_values)

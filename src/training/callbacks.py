import torch
import numpy as np
from typing import Optional


class EarlyStopping:
    def __init__(self, patience: int = 5000, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float("inf")
        self.best_physics = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float, physics_residual: float) -> bool:
        improved = False

        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            improved = True

        if physics_residual < self.best_physics - self.min_delta:
            self.best_physics = physics_residual
            improved = True

        if improved:
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop


class GradientMonitor:
    def __init__(self):
        self.stats = {}

    def compute_grad_stats(
        self,
        model: torch.nn.Module,
        loss_terms: dict,
    ) -> dict:
        stats = {}
        for name, loss in loss_terms.items():
            if loss.requires_grad:
                model.zero_grad()
                loss.backward(retain_graph=True)
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                stats[name] = total_norm ** 0.5
            else:
                stats[name] = 0.0

        self.stats = stats
        return stats

    def check_health(self) -> list:
        warnings = []
        for name, norm in self.stats.items():
            if norm < 1e-8:
                warnings.append(f"[WARN] {name}: vanishing gradients ({norm:.2e})")
            elif norm > 1e3:
                warnings.append(f"[WARN] {name}: exploding gradients ({norm:.2e})")


        norms = [v for v in self.stats.values() if v > 0]
        if len(norms) >= 2:
            ratio = max(norms) / (min(norms) + 1e-10)
            if ratio > 1000:
                warnings.append(
                    f"[WARN] Gradient imbalance: max/min ratio = {ratio:.0f}. "
                    f"Consider adjusting loss weights."
                )

        return warnings

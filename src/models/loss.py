import torch
import torch.nn as nn
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import W_DATA, W_PDE, W_BC, W_IC, W_SOURCE_SPARSITY


class AdaptiveLossWeights(nn.Module):
    def __init__(self, n_losses: int = 5, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.ones(n_losses))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if not self.initialized:


                active = losses.detach() > 1e-10
                init_vals = torch.where(active,
                                        losses.detach().clamp(min=1e-8),
                                        self.running_mean)
                self.running_mean.copy_(init_vals)
                self.initialized.fill_(True)
            else:
                updated = (
                    self.momentum * self.running_mean +
                    (1 - self.momentum) * losses.detach().clamp(min=1e-8)
                )
                active = losses.detach() > 1e-10
                self.running_mean.copy_(
                    torch.where(active, updated, self.running_mean)
                )


        weights = 1.0 / self.running_mean
        weights = weights / weights.sum() * len(weights)
        return weights


class AerasLoss(nn.Module):
    def __init__(
        self,
        w_data: float = W_DATA,
        w_pde: float = W_PDE,
        w_bc: float = W_BC,
        w_ic: float = W_IC,
        w_sparsity: float = W_SOURCE_SPARSITY,
        w_nonneg: float = 0.1,   # non-negativity constraint weight (jPINN 2025)
        adaptive: bool = True,
    ):
        super().__init__()
        self.base_weights = torch.tensor([w_data, w_pde, w_bc, w_ic, w_sparsity, w_nonneg])
        self.adaptive = adaptive
        if adaptive:
            self.adaptive_weights = AdaptiveLossWeights(n_losses=4)

    def data_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        underpred_alpha: float = 3.0,
    ) -> torch.Tensor:
        diff = pred - target
        asymm = torch.where(diff < 0, underpred_alpha, 1.0)
        sq = asymm * (diff ** 2)

        if mask is not None:
            sq = sq * mask
        if weight is not None:
            sq = sq * weight

        if mask is not None and weight is not None:
            denom = (mask * weight).sum().clamp(min=1)
        elif mask is not None:
            denom = mask.sum().clamp(min=1)
        elif weight is not None:
            denom = weight.sum().clamp(min=1)
        else:
            denom = torch.tensor(float(diff.numel()), device=diff.device)

        return sq.sum() / denom

    def pde_loss(self, residual: torch.Tensor) -> torch.Tensor:
        return (residual ** 2).mean()

    def bc_loss(self, bc_residual: torch.Tensor) -> torch.Tensor:
        return (bc_residual ** 2).mean()

    def ic_loss(
        self,
        pred_t0: torch.Tensor,
        target_t0: torch.Tensor,
    ) -> torch.Tensor:
        return ((pred_t0 - target_t0) ** 2).mean()

    def sparsity_loss(self, source_output: torch.Tensor) -> torch.Tensor:
        return source_output.abs().mean()

    def nonnegativity_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Penalise negative concentration predictions (jPINN, 2025).
        Physical constraint: all pollutant concentrations >= 0.
        Uses ReLU so only negative predictions incur a penalty; positive
        predictions contribute zero gradient — no interference with data loss.
        """
        return torch.nn.functional.relu(-pred).mean()

    def forward(
        self,
        losses_dict: Dict[str, torch.Tensor],
        epoch: int = 0,
        curriculum_data_epochs: int = 5000,
        curriculum_rampup_epochs: int = 10000,
    ) -> Dict[str, torch.Tensor]:
        L_data     = losses_dict.get("data",     torch.tensor(0.0))
        L_pde      = losses_dict.get("pde",      torch.tensor(0.0))
        L_bc       = losses_dict.get("bc",       torch.tensor(0.0))
        L_ic       = losses_dict.get("ic",       torch.tensor(0.0))
        L_sparsity = losses_dict.get("sparsity", torch.tensor(0.0))
        L_nonneg   = losses_dict.get("nonneg",   torch.tensor(0.0))

        raw_losses = torch.stack([L_data, L_pde, L_bc, L_ic, L_sparsity, L_nonneg])


        if epoch < curriculum_data_epochs:
            pde_scale = 0.0
        elif epoch < curriculum_data_epochs + curriculum_rampup_epochs:
            progress = (epoch - curriculum_data_epochs) / curriculum_rampup_epochs
            pde_scale = progress
        else:
            pde_scale = 1.0


        weights = self.base_weights.to(raw_losses.device).clone()
        weights[1] *= pde_scale
        weights[2] *= pde_scale
        weights[3] *= pde_scale


        if self.adaptive and pde_scale > 0 and raw_losses[:4].sum() > 0:
            adaptive_w_core = self.adaptive_weights(raw_losses[:4])
            adaptive_w = torch.cat([adaptive_w_core, torch.ones(2, device=raw_losses.device)])
            weights = weights * adaptive_w

        total = (weights * raw_losses).sum()

        return {
            "total": total,
            "data": L_data.item(),
            "pde": L_pde.item(),
            "bc": L_bc.item(),
            "ic": L_ic.item(),
            "sparsity": L_sparsity.item(),
            "nonneg": L_nonneg.item(),
            "pde_scale": pde_scale,
            "weights": weights.detach().cpu().tolist(),
        }



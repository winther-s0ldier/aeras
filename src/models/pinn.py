import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    INPUT_DIM, OUTPUT_DIM, HIDDEN_LAYERS, HIDDEN_DIM,
    ACTIVATION, FOURIER_FEATURES, FOURIER_SIGMA, DEVICE,
    LOG_DX_INIT, LOG_DY_INIT, LAMBDA_DEP_INIT,
    SOURCE_NET_HIDDEN, SOURCE_NET_LAYERS,
)


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, num_frequencies: int = 64, sigma: float = 10.0):
        super().__init__()

        B = torch.randn(input_dim, num_frequencies) * sigma
        self.register_buffer("B", B)
        self.output_dim = num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x @ self.B
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class AerasPINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D Spatiotemporal Air Quality.
    Predicts PM2.5 concentration C(x,y,t).
    """
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_layers: int = HIDDEN_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        fourier_features: int = FOURIER_FEATURES,
        fourier_sigma: float = FOURIER_SIGMA,
        inverse_mode: bool = False,
    ):
        super().__init__()
        self.inverse_mode = inverse_mode

        self.fourier = FourierEmbedding(input_dim, fourier_features, fourier_sigma)

        layers = []
        in_dim = self.fourier.output_dim

        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, OUTPUT_DIM))
        self.network = nn.Sequential(*layers)

        self.log_Dx = nn.Parameter(torch.tensor(LOG_DX_INIT))
        self.log_Dy = nn.Parameter(torch.tensor(LOG_DY_INIT))
        self.log_lambda_dep = nn.Parameter(torch.tensor(float(np.log(LAMBDA_DEP_INIT))))

        if self.inverse_mode:
            from src.models.source_net import SourceNet
            self.source_net = SourceNet(
                hidden_dim=SOURCE_NET_HIDDEN,
                num_layers=SOURCE_NET_LAYERS
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def Dx(self) -> torch.Tensor:
        return torch.exp(self.log_Dx)

    @property
    def Dy(self) -> torch.Tensor:
        return torch.exp(self.log_Dy)

    @property
    def lambda_dep(self) -> torch.Tensor:
        return torch.exp(self.log_lambda_dep)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.fourier(inputs)
        return self.network(embedded)

# Backwards compatibility — remove after Phase 5
VayuPINN = AerasPINN


def compute_pde_residual(
    model: AerasPINN,
    inputs: torch.Tensor,
    source_term: torch.Tensor = None,
) -> torch.Tensor:

    inputs = inputs.requires_grad_(True)


    C = model(inputs)


    u_wind = inputs[:, 3:4]
    v_wind = inputs[:, 4:5]


    grad_C = torch.autograd.grad(
        C, inputs,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True,
    )[0]

    dC_dx = grad_C[:, 0:1]
    dC_dy = grad_C[:, 1:2]
    dC_dt = grad_C[:, 2:3]


    d2C_dx2 = torch.autograd.grad(
        dC_dx, inputs,
        grad_outputs=torch.ones_like(dC_dx),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]

    d2C_dy2 = torch.autograd.grad(
        dC_dy, inputs,
        grad_outputs=torch.ones_like(dC_dy),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]


    advection = u_wind * dC_dx + v_wind * dC_dy


    diffusion = model.Dx * d2C_dx2 + model.Dy * d2C_dy2


    S = source_term if source_term is not None else 0.0


    deposition = model.lambda_dep * C


    residual = dC_dt + advection - diffusion - S + deposition

    return residual


def compute_boundary_residual(
    model: AerasPINN,
    boundary_points: torch.Tensor,
    normal_direction: int = 0,
) -> torch.Tensor:
    boundary_points = boundary_points.requires_grad_(True)
    C = model(boundary_points)

    grad_C = torch.autograd.grad(
        C, boundary_points,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
    )[0]

    dC_dn = grad_C[:, normal_direction:normal_direction + 1]
    return dC_dn

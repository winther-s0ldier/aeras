import torch
import torch.nn as nn
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import SOURCE_NET_HIDDEN, SOURCE_NET_LAYERS


class SourceGrid(nn.Module):
    def __init__(self, nx: int = 50, ny: int = 50, nt: int = 24):
        super().__init__()

        self.raw_grid = nn.Parameter(torch.zeros(nt, ny, nx))
        self.nx, self.ny, self.nt = nx, ny, nt

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        grid = nn.functional.softplus(self.raw_grid)


        x = inputs[:, 0]
        y = inputs[:, 1]
        t = inputs[:, 2]


        ix = x * (self.nx - 1)
        iy = y * (self.ny - 1)
        it = t * (self.nt - 1)


        grid_coords = torch.stack([
            2 * x - 1,
            2 * y - 1,
            2 * t - 1,
        ], dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        grid_5d = grid.unsqueeze(0).unsqueeze(0)

        S = nn.functional.grid_sample(
            grid_5d, grid_coords,
            mode="bilinear", padding_mode="border", align_corners=True
        )

        return S.reshape(-1, 1)

    def get_source_map(self, t_index: int = 0) -> np.ndarray:
        with torch.no_grad():
            grid = nn.functional.softplus(self.raw_grid)
            return grid[t_index].cpu().numpy()


class SourceNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,


        hidden_dim: int = SOURCE_NET_HIDDEN,
        num_layers: int = SOURCE_NET_LAYERS,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim


        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def get_source_map(
        self,
        t_value: float,
        hour_value: float = 0.5,
        dow_value: float = 0.5,
        resolution: int = 50,
        device: str = "cpu",
    ) -> np.ndarray:
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        points = torch.stack([
            xx.flatten(),
            yy.flatten(),
            torch.full((resolution ** 2,), t_value),
            torch.full((resolution ** 2,), hour_value),
            torch.full((resolution ** 2,), dow_value),
        ], dim=1).to(device)

        with torch.no_grad():
            S = self.network(points)

        return S.reshape(resolution, resolution).cpu().numpy()


def source_sparsity_loss(source_output: torch.Tensor) -> torch.Tensor:
    return source_output.abs().mean()

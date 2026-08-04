"""GPS MLP encoder. Input: (B, in_dim) standardized [dlat_m, dlon_m, speed]
-> (B, out_dim)."""
from __future__ import annotations

import torch
import torch.nn as nn


class GPSEncoder(nn.Module):
    def __init__(self, out_dim: int, in_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

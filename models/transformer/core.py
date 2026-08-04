"""Causal Transformer temporal core — the matched baseline for the SSM.

Pre-norm Transformer encoder with a causal (subsequent) mask so it only attends
to past frames, mirroring the SSM's causal recurrence. Learned positional
embeddings up to `max_len` (window sizes are small: 8/32/128). (B,T,D)->(B,T,D).

Sliding-window inference (window in {8,32,128}) is applied at eval time by the
caller feeding fixed-length windows; this module itself is window-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerCore(nn.Module):
    def __init__(self, dim: int, n_layers: int = 4, nhead: int = 4,
                 dim_feedforward: int = 192, dropout: float = 0.0, max_len: int = 128):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm_f = nn.LayerNorm(dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = x + self.pos[:, :t]
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=mask)
        return self.norm_f(x)

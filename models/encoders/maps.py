"""Small CNN encoder for 2-D "map" modalities (LiDAR BEV, radar RA/RV maps).

Same depthwise-separable, ONNX-safe design as the camera encoder but for smaller
multi-channel maps (e.g. 3×128×128 LiDAR BEV, 2×128×128 radar). One frame ->
(B, out_dim). Reused for both LiDAR and radar with different in_ch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .camera import _DWSepBlock, _conv_bn_act


class MapEncoder(nn.Module):
    def __init__(self, out_dim: int, in_ch: int, width: int = 32):
        super().__init__()
        w = width
        self.stem = _conv_bn_act(in_ch, w, stride=2)            # /2
        self.blocks = nn.Sequential(
            _DWSepBlock(w, w * 2, stride=2),                    # /4
            _DWSepBlock(w * 2, w * 4, stride=2),                # /8
            _DWSepBlock(w * 4, w * 4, stride=1),
            _DWSepBlock(w * 4, w * 8, stride=2),                # /16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(w * 8, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

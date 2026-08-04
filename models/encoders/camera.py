"""Small CNN camera encoder (MobileNetV3-class weight, edge-friendly).

Compact conv stem -> a few stride-2 blocks -> global average pool -> linear to
`out_dim`. Only ONNX/TensorRT-safe ops (conv, BN, ReLU6, GAP, linear) so the
Phase-2 export path never breaks. Input: (B, 3, H, W) -> (B, out_dim).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_act(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU6(inplace=True),
    )


class _DWSepBlock(nn.Module):
    """Depthwise-separable conv block (MobileNet-style), stride on the depthwise."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class CameraEncoder(nn.Module):
    def __init__(self, out_dim: int, width: int = 32, in_ch: int = 3):
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
        # NHWC memory format for faster Ampere conv kernels (weights are already
        # channels_last via model.to(memory_format=...)).
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

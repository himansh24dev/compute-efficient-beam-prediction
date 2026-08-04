"""ImageNet-pretrained camera encoder (transfer learning).

Training our custom CNN from scratch on ~15k frames is data-starved; a pretrained
MobileNetV3-small backbone typically gives a solid accuracy boost at modest extra
cost (~1.0 M params, ~0.06 GFLOPs/frame) — still ~11x lighter than the SSM SOTA.
torchvision is imported lazily so the package works without it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PretrainedCameraEncoder(nn.Module):
    """MobileNetV3-small features (optionally ImageNet-pretrained) -> out_dim.

    Expects ImageNet-normalised 3x224x224 input, which our camera preprocessing
    already produces.
    """

    def __init__(self, out_dim: int, pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = mobilenet_v3_small(weights=weights)
        except Exception:  # offline / no weights -> random init, still trainable
            net = mobilenet_v3_small(weights=None)
        self.features = net.features          # -> (B, 576, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(576, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

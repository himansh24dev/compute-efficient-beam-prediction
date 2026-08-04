"""Faithful BeMamba-*architecture* replica for the on-device LATENCY comparison.

We do NOT reproduce BeMamba's accuracy (no public code; their vehicle-enhancement /
virtual-point preprocessing is the accuracy driver). But latency depends on the
ARCHITECTURE (operator types + sizes), not on trained weights or data, so we
rebuild their published compute profile:
  - ~16.19 M parameters, ~23.4 GFLOPs per inference (W=5 snapshot),
  - a compute-heavy multimodal conv encoder stack,
  - Mamba / selective-SSM temporal+cross-modal fusion (our S6 core), and
  - a large (~13.3 M-param) multi-layer fully-connected head.
Knobs (`W_ENC`, head dims) are tuned so params+FLOPs match the reported budget;
run this file to print the current params / GFLOPs and adjust.

This replaces the ResNet-50 stand-in in Fig. 5 with a model that has BeMamba's
actual operator mix, giving a defensible edge-latency contrast.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.ssm.s6 import SSMCore


def conv(ci, co, k=3, s=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class HeavyImageEncoder(nn.Module):
    """Compute-heavy, param-light conv backbone (few channels at high resolution)
    to reach BeMamba's per-frame FLOP scale with modest parameters."""

    def __init__(self, w=40, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            conv(3, w, k=3, s=2),          # 224 -> 112
            conv(w, w, k=3, s=1),
            conv(w, 2 * w, k=3, s=1),
            nn.MaxPool2d(2),               # 112 -> 56
            conv(2 * w, 2 * w, k=3, s=1),
            conv(2 * w, 4 * w, k=3, s=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(4 * w, out_dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class MapEncoder(nn.Module):
    def __init__(self, in_ch, out_dim=256, w=32):
        super().__init__()
        self.net = nn.Sequential(conv(in_ch, w, 3, 2), conv(w, 2 * w, 3, 2),
                                 conv(2 * w, 4 * w, 3, 2), nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(4 * w, out_dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class BeMambaReplica(nn.Module):
    """Multimodal (camera+LiDAR+radar+GPS) Mamba-fusion model with a large FC head,
    built to BeMamba's reported param/FLOP budget."""

    def __init__(self, dim=256, num_beams=64, window=5, head_hidden=(4096, 1856),
                 img_w=82, ssm_layers=4):
        super().__init__()
        self.window = window
        self.cam = HeavyImageEncoder(w=img_w, out_dim=dim)
        self.lidar = MapEncoder(3, dim)
        self.radar = MapEncoder(2, dim)
        self.gps = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, dim))
        self.fuse = nn.Linear(4 * dim, dim)
        self.core = SSMCore(dim, n_layers=ssm_layers)
        h1, h2 = head_hidden
        self.head = nn.Sequential(
            nn.Linear(window * dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
            nn.Linear(h2, num_beams))

    def _enc(self, enc, x):
        b, t = x.shape[:2]
        return enc(x.flatten(0, 1)).view(b, t, -1)

    def forward(self, camera, gps, lidar, radar):
        ec = self._enc(self.cam, camera)
        el = self._enc(self.lidar, lidar)
        er = self._enc(self.radar, radar)
        eg = self._enc(self.gps, gps)
        u = self.fuse(torch.cat([ec, el, er, eg], dim=-1))    # (B,W,dim)
        h = self.core(u)                                      # (B,W,dim)
        return self.head(h.flatten(1))                        # (B,64)


def _report():
    W = 5
    m = BeMambaReplica(window=W).eval()
    n = sum(p.numel() for p in m.parameters())
    cam = torch.randn(1, W, 3, 224, 224)
    gps = torch.randn(1, W, 3)
    lid = torch.randn(1, W, 3, 128, 128)
    rad = torch.randn(1, W, 2, 128, 128)
    with torch.no_grad():
        out = m(cam, gps, lid, rad)
    print(f"params: {n/1e6:.2f} M   (target 16.19 M)")
    print(f"head params: {sum(p.numel() for p in m.head.parameters())/1e6:.2f} M (target ~13.3 M)")
    try:
        from fvcore.nn import FlopCountAnalysis
        g = FlopCountAnalysis(m, (cam, gps, lid, rad)).total() / 1e9
        print(f"GFLOPs/inference: {g:.1f}   (target 23.4)")
    except Exception as e:
        print("fvcore unavailable:", e)
    print("output:", tuple(out.shape))


if __name__ == "__main__":
    _report()

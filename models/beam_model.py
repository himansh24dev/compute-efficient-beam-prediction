"""Assembled streaming beam predictor.

    inputs[modality] : (B, T, ...)   ->   encoders (per frame)
                                     ->   gated fusion            (B, T, D)
                                     ->   temporal core (SSM|XF)  (B, T, D)
                                     ->   beam head (last frame)  (B, num_beams)

Encoders / fusion / head are IDENTICAL across the SSM and Transformer variants;
only `core_kind` changes. The snapshot task predicts the CURRENT beam (label at
the last frame of the window). Optional auxiliary future-beam heads (t+1..t+H)
support the tracking framing but are off by default for the Phase-0 spike.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import CameraEncoder, GPSEncoder, MapEncoder
from .fusion import build_fusion
from .temporal import build_temporal_core


class BeamModel(nn.Module):
    def __init__(self, modalities: list[str], num_beams: int, core_kind: str,
                 model_cfg: dict):
        super().__init__()
        dim = int(model_cfg["dim"])
        self.modalities = list(modalities)
        self.num_beams = num_beams
        self.core_kind = core_kind

        # Channel counts for the 2-D "map" modalities (LiDAR BEV / radar maps).
        map_ch = {"lidar": 3, "radar": 2}
        map_ch.update(model_cfg.get("map_channels", {}))
        map_width = int(model_cfg.get("map", {}).get("width", 32))

        enc = {}
        if "camera" in modalities:
            cam_cfg = model_cfg.get("camera", {})
            if cam_cfg.get("pretrained", False):
                from .encoders.pretrained import PretrainedCameraEncoder
                enc["camera"] = PretrainedCameraEncoder(dim, pretrained=True)
            else:
                enc["camera"] = CameraEncoder(dim, width=int(cam_cfg.get("width", 32)))
        if "gps" in modalities:
            enc["gps"] = GPSEncoder(dim, in_dim=int(model_cfg.get("gps", {}).get("in_dim", 3)),
                                    hidden=int(model_cfg.get("gps", {}).get("hidden", 128)))
        if "lidar" in modalities:
            enc["lidar"] = MapEncoder(dim, in_ch=int(map_ch["lidar"]), width=map_width)
        if "radar" in modalities:
            enc["radar"] = MapEncoder(dim, in_ch=int(map_ch["radar"]), width=map_width)
        self.encoders = nn.ModuleDict(enc)

        self.fusion = build_fusion(model_cfg.get("fusion_kind", "gated"), modalities, dim,
                                    modality_dropout_p=float(model_cfg.get("modality_dropout_p", 0.0)))
        self.core = build_temporal_core(core_kind, dim, int(model_cfg["core_layers"]), model_cfg)
        self.head = nn.Linear(dim, num_beams)

        # Optional train-time GPS-noise augmentation (reviewer C7 mitigation): add
        # Gaussian noise (standardized units) to the GPS stream during training only.
        self.gps_noise_std = float(model_cfg.get("gps_noise_std", 0.0))

        self.future_horizons = list(model_cfg.get("future_horizons", []))
        self.future_heads = nn.ModuleList(
            [nn.Linear(dim, num_beams) for _ in self.future_horizons]
        ) if self.future_horizons else None

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = {}
        for m, enc in self.encoders.items():
            x = inputs[m]                          # (B, T, ...)
            b, t = x.shape[:2]
            x = x.flatten(0, 1)                    # (B*T, ...)
            e = enc(x)                             # (B*T, D)
            feats[m] = e.view(b, t, -1)            # (B, T, D)
        return feats

    def forward(self, inputs: dict[str, torch.Tensor]):
        if self.training and self.gps_noise_std > 0 and "gps" in inputs:
            g = inputs["gps"]
            inputs = {**inputs, "gps": g + torch.randn_like(g) * self.gps_noise_std}
        feats = self.encode(inputs)
        fused = self.fusion(feats)                 # (B, T, D)
        h = self.core(fused)                       # (B, T, D)
        last = h[:, -1]                            # current-beam readout (B, D)
        logits = self.head(last)                   # (B, num_beams)
        if self.future_heads is None:
            return logits
        future = torch.stack([hd(last) for hd in self.future_heads], dim=1)  # (B, H, num_beams)
        return logits, future

    # convenience for the parity check
    def core_module(self) -> nn.Module:
        return self.core

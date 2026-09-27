"""Export a GPS-ONLY W=2 SSM predictor to ONNX so its on-device latency can be measured
as a second Pareto point (reviewer S2). Latency is architecture-determined, not
weight-determined, so a freshly-built model gives a faithful latency; the GPS-only
ACCURACY (DBA 0.853) comes separately from the modality ablation.

  .venv/bin/python -m experiments.edge_deploy.export_gps_only
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

from models.beam_model import BeamModel


class GpsWrap(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, gps):
        out = self.model({"gps": gps})
        return out[0] if isinstance(out, tuple) else out


def main():
    W = 2
    out_dir = "experiments/edge_deploy/artifacts/gps_only"
    os.makedirs(out_dir, exist_ok=True)
    model_cfg = {"dim": 128, "core_layers": 4,
                 "gps": {"in_dim": 3, "hidden": 128},
                 "ssm": {"d_state": 16, "d_conv": 4, "expand": 2, "dt_rank": 8}}
    model = BeamModel(["gps"], 64, "ssm", model_cfg).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"GPS-only model: {n:,} params ({n/1e6:.3f}M)")

    wrap = GpsWrap(model).eval()
    gps = torch.randn(1, W, 3)
    with torch.no_grad():
        ref = wrap(gps).numpy()
    path = os.path.join(out_dir, "beam_gps_fp32.onnx")
    torch.onnx.export(wrap, (gps,), path, input_names=["gps"], output_names=["logits"],
                      dynamic_axes={"gps": {0: "batch"}}, opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print(f"exported -> {path} ({os.path.getsize(path)/1e3:.0f} KB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"gps": gps.numpy()})[0]
    print(f"PyTorch vs ONNX max_abs_err={float(np.abs(o-ref).max()):.2e}")


if __name__ == "__main__":
    raise SystemExit(main())

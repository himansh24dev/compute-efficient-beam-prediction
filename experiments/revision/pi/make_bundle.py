"""Assemble the Raspberry Pi 4 bundle for the revision's on-device runs.

models/ : deploy_fp32 / deploy_int8 / deploy_int8_selective (validation-selected
          A_ssm_2M_i1337, from experiments/revision/int8_eval.py), gps_fp32
          (A_ssm_gps_i1337), replica_full (23.41 GMACs) and replica_half (11.58 GMACs,
          the published "23.4 GFLOPs" read as true FLOPs; only the camera-encoder width
          changes, 82 -> 57 channels base).
frames/ : 120 real 960x540 DeepSense JPEGs, 30 from one test pass of each scenario.
Scripts : bench_pipeline.py (revised processing pipeline), bench_auto.py (forward pass),
          bench_profile.py (per-operator), run_pi.sh (runs everything).

  .venv/bin/python -m experiments.revision.pi.make_bundle
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from models.beam_model import BeamModel
from experiments.edge_deploy.bemamba_replica import BeMambaReplica

B = Path("experiments/revision/pi/bundle")
R = Path("experiments/revision/results")


class GpsWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, gps):
        o = self.m({"gps": gps}); return o[0] if isinstance(o, tuple) else o


def export(model, args, names, path, dyn):
    torch.onnx.export(model, args, str(path), input_names=names, output_names=["logits"],
                      dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(*args).numpy()
    got = s.run(None, {n: a.numpy() for n, a in zip(names, args)})[0]
    err = float(np.abs(got - ref).max())
    assert err < 1e-3, err
    return err


def main():
    (B / "models").mkdir(parents=True, exist_ok=True)
    info = {}
    src = R / "onnx" / "s1337"
    for f, dst in (("beam_ssm_fp32.onnx", "deploy_fp32.onnx"), ("beam_ssm_int8.onnx", "deploy_int8.onnx"),
                   ("beam_ssm_int8_selective.onnx", "deploy_int8_selective.onnx")):
        shutil.copy(src / f, B / "models" / dst)
    o = torch.load(R / "ckpt" / "A_ssm_gps_i1337.pt", map_location="cpu")
    m = BeamModel(o["modalities"], 64, o["core_kind"], o["model_cfg"]).eval()
    m.load_state_dict(o["model"])
    info["gps_fp32_max_err"] = export(GpsWrap(m).eval(), (torch.randn(1, 2, 3),), ["gps"],
                                      B / "models" / "gps_fp32.onnx", {"gps": {0: "batch"}})
    class CamWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, camera):
            o = self.m({"camera": camera}); return o[0] if isinstance(o, tuple) else o
    o = torch.load(R / "ckpt" / "A_ssm_cam_i1337.pt", map_location="cpu")
    m = BeamModel(o["modalities"], 64, o["core_kind"], o["model_cfg"]).eval()
    m.load_state_dict(o["model"])
    info["cam_fp32_max_err"] = export(CamWrap(m).eval(), (torch.randn(1, 2, 3, 224, 224),), ["camera"],
                                      B / "models" / "cam_fp32.onnx", {"camera": {0: "batch"}})
    # the four alternative temporal cores (same encoders/head), to measure whether the
    # core changes on-device latency
    from experiments.edge_deploy.export_onnx import ExportWrapper
    for core in ("transformer", "gru", "lstm", "mlp"):
        o = torch.load(R / "ckpt" / f"A_{core}_2M_i1337.pt", map_location="cpu")
        m = BeamModel(o["modalities"], 64, o["core_kind"], o["model_cfg"]).eval()
        m.load_state_dict(o["model"])
        info[f"core_{core}_max_err"] = export(ExportWrapper(m).eval(),
                                              (torch.randn(1, 2, 3), torch.randn(1, 2, 3, 224, 224)),
                                              ["gps", "camera"], B / "models" / f"core_{core}_fp32.onnx",
                                              {"gps": {0: "batch"}, "camera": {0: "batch"}})
    W = 5
    args = (torch.randn(1, W, 3, 224, 224), torch.randn(1, W, 3),
            torch.randn(1, W, 3, 128, 128), torch.randn(1, W, 2, 128, 128))
    from fvcore.nn import FlopCountAnalysis
    for tag, w in (("replica_full", 82), ("replica_half", 57)):
        torch.manual_seed(0)
        rep = BeMambaReplica(window=W, img_w=w).eval()
        fa = FlopCountAnalysis(rep, args); fa.unsupported_ops_warnings(False); fa.uncalled_modules_warnings(False)
        info[tag] = {"img_w": w, "params": sum(p.numel() for p in rep.parameters()),
                     "gmacs": round(fa.total() / 1e9, 3),
                     "max_err": export(rep, args, ["camera", "gps", "lidar", "radar"],
                                       B / "models" / f"{tag}.onnx", None)}
    for s in ("experiments/revision/pi/bench_pipeline.py", "experiments/edge_deploy/bench_auto.py",
              "experiments/edge_deploy/bench_profile.py", "experiments/revision/pi/run_pi.sh"):
        shutil.copy(s, B / Path(s).name)
    json.dump(info, open(B / "models" / "models_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())

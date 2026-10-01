"""Second Raspberry Pi batch: forward-pass latency of every remaining trained variant,
so that no on-device number in the paper comes from the pre-revision weights and no
ablation row is "not measured".

  window sweep : W = 4, 5, 8, 16 (SSM, camera+GPS; W=2 is in the main bundle)
  ablations    : + LiDAR, + LiDAR + radar, ImageNet-pretrained encoder, 2 and 6 SSM
                 layers, mean and concat fusion (all W = 2)

  .venv/bin/python -m experiments.revision.pi.make_bundle_extra
writes experiments/revision/pi/bundle_extra/{models/*.onnx, run_extra.sh}
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from models.beam_model import BeamModel

R = Path("experiments/revision/results/ckpt")
B = Path("experiments/revision/pi/bundle_extra")
SHAPE = {"gps": (3,), "camera": (3, 224, 224), "lidar": (3, 128, 128), "radar": (2, 128, 128)}
ORDER = ("gps", "camera", "lidar", "radar")

JOBS = [("win_w4", "E_w4_i1337"), ("win_w5", "C_w5_episode-random_s1337"), ("win_w8", "E_w8_i1337"),
        ("win_w16", "E_w16_i1337"), ("abl_lidar", "B_ssm_3M_i1337"), ("abl_lidar_radar", "B_ssm_4M_i1337"),
        ("abl_pretrained", "B_pretrained_i1337"), ("abl_L2", "B_ssm_L2_i1337"), ("abl_L6", "B_ssm_L6_i1337"),
        ("abl_fuse_mean", "B_fuse_mean_i1337"), ("abl_fuse_concat", "B_fuse_concat_i1337")]


class Wrap(torch.nn.Module):
    def __init__(self, m, mods):
        super().__init__(); self.m = m; self.mods = mods

    def forward(self, *xs):
        o = self.m(dict(zip(self.mods, xs)))
        return o[0] if isinstance(o, tuple) else o


def main():
    (B / "models").mkdir(parents=True, exist_ok=True)
    info, lines = {}, []
    for tag, ck in JOBS:
        o = torch.load(R / f"{ck}.pt", map_location="cpu")
        mods = [m for m in ORDER if m in o["modalities"]]
        m = BeamModel(o["modalities"], 64, o["core_kind"], o["model_cfg"]).eval()
        m.load_state_dict(o["model"])
        W = int(o["window"])
        args = tuple(torch.randn(1, W, *SHAPE[k]) for k in mods)
        w = Wrap(m, mods).eval()
        path = B / "models" / f"{tag}.onnx"
        torch.onnx.export(w, args, str(path), input_names=mods, output_names=["logits"],
                          dynamic_axes={k: {0: "batch"} for k in mods}, opset_version=17,
                          do_constant_folding=True, dynamo=False)
        import onnxruntime as ort
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = w(*args).numpy()
        err = float(np.abs(s.run(None, {k: a.numpy() for k, a in zip(mods, args)})[0] - ref).max())
        assert err < 1e-3, (tag, err)
        info[tag] = {"ckpt": ck, "window": W, "modalities": mods, "max_err": err,
                     "params": sum(p.numel() for p in m.parameters())}
        runs = 100 if W >= 8 or "pretrained" in tag else 300
        lines.append(f"$PY bench_auto.py --model models/{tag}.onnx --window {W} --threads 4 --runs {runs} "
                     f"--out results_extra/fwd_{tag}_t4.json")
    json.dump(info, open(B / "models" / "models_info.json", "w"), indent=1)
    shutil.copy("experiments/edge_deploy/bench_auto.py", B / "bench_auto.py")
    (B / "run_extra.sh").write_text(
        "#!/usr/bin/env bash\n# second Pi batch (forward pass of every remaining variant)\nset -euo pipefail\n"
        "PY=${PY:-python3}\nmkdir -p results_extra\n"
        + "\nsleep 20\n".join(lines) + "\necho ALL EXTRA DONE\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())

"""Static (calibrated) INT8 quantization of the deployment ONNX model.

Static quant needs a handful of REAL input windows to calibrate activation
ranges -> it can then quantize the conv-heavy camera encoder (dynamic quant
can't), which is where the ARM speedup comes from. We also report how much INT8
changes the top-1 beam vs fp32 (the quantization-robustness ablation).

  .venv/bin/python -m experiments.edge_deploy.quantize_int8 \
      --fp32 experiments/edge_deploy/artifacts/deploy_w2_fp32.onnx \
      --out  experiments/edge_deploy/artifacts/deploy_w2_int8.onnx \
      --protocol episode-random --ncal 200
"""
from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort

from data.config import load_data_config
from utils.yaml_io import load_yaml
from experiments.bemamba_compare.splits_compare import build_compare_loaders


class RealWindowReader:
    """CalibrationDataReader: yields real (gps, camera) single-window feeds."""

    def __init__(self, loader, n: int):
        self.samples = []
        for batch in loader:
            inp = batch["inputs"]
            g = inp["gps"].numpy().astype(np.float32)
            c = inp["camera"].numpy().astype(np.float32)
            for i in range(g.shape[0]):
                self.samples.append({"gps": g[i:i + 1], "camera": c[i:i + 1]})
                if len(self.samples) >= n:
                    break
            if len(self.samples) >= n:
                break
        self._it = iter(self.samples)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/deploy_w2.yaml")
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="episode-random")
    ap.add_argument("--ncal", type=int, default=200)
    ap.add_argument("--op-types", nargs="*", default=None,
                    help="restrict quantisation to these op types, e.g. Conv "
                         "(=> selective: encoders INT8, SSM/head fp32)")
    args = ap.parse_args()

    exp = load_yaml(args.config)
    cfg = load_data_config(exp["data_config"])
    seed = int(exp["sweep"]["seeds"][0])
    tr, te, _ = build_compare_loaders(cfg, exp, ["gps", "camera"], args.protocol, seed)

    reader = RealWindowReader(tr, args.ncal)
    print(f"calibrating on {len(reader.samples)} real windows ...")

    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process
    prep = args.fp32.replace(".onnx", "_prep.onnx")
    try:
        quant_pre_process(args.fp32, prep, skip_symbolic_shape=False)
        src = prep
    except Exception as e:
        print(f"[pre_process skipped: {type(e).__name__}] quantizing raw graph")
        src = args.fp32
    qkw = {}
    if args.op_types:
        qkw["op_types_to_quantize"] = args.op_types
        print(f"selective quantisation: op types {args.op_types} only")
    quantize_static(src, args.out, reader, quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8, **qkw)
    import os
    print(f"INT8 saved -> {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)")

    # quantization robustness: top-1 agreement vs fp32 on the test set
    s32 = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    agree = tot = 0
    for batch in te:
        inp = batch["inputs"]
        g = inp["gps"].numpy().astype(np.float32); c = inp["camera"].numpy().astype(np.float32)
        for i in range(g.shape[0]):
            f = {"gps": g[i:i + 1], "camera": c[i:i + 1]}
            if int(s32.run(None, f)[0].argmax()) == int(s8.run(None, f)[0].argmax()):
                agree += 1
            tot += 1
        if tot >= 500:
            break
    print(f"INT8 vs fp32 top-1 agreement: {agree}/{tot} = {100*agree/tot:.1f}%")


if __name__ == "__main__":
    main()

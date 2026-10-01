"""Evaluate an ONNX model's real beam-prediction accuracy (Top-k + exact DBA) on
the test set, against ground truth. Used to quantify the INT8 accuracy cost with
the metric that matters (DBA/Top-3), not fp32-agreement.

  .venv/bin/python -m experiments.edge_deploy.eval_onnx \
      --models experiments/edge_deploy/artifacts/deploy_w2/beam_ssm_fp32.onnx \
               experiments/edge_deploy/artifacts/deploy_w2/beam_ssm_int8.onnx \
      --protocol episode-random
"""
from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

from data.config import load_data_config
from utils.yaml_io import load_yaml
from experiments.bemamba_compare.splits_compare import build_compare_loaders
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics


def eval_onnx(path, te):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    acc = new_accumulator()
    for batch in te:
        inp = batch["inputs"]; tgt = batch["beam"]
        g = inp["gps"].numpy().astype(np.float32); c = inp["camera"].numpy().astype(np.float32)
        logits = [sess.run(None, {"gps": g[i:i + 1], "camera": c[i:i + 1]})[0][0]
                  for i in range(g.shape[0])]
        add(acc, full_metrics(torch.tensor(np.array(logits)), tgt))
    return finalize(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/deploy_w2.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--protocol", default="episode-random")
    args = ap.parse_args()

    exp = load_yaml(args.config)
    cfg = load_data_config(exp["data_config"])
    seed = int(exp["sweep"]["seeds"][0])
    _, _, te, _ = build_compare_loaders(cfg, exp, ["gps", "camera"], args.protocol, seed)

    print(f"protocol={args.protocol}")
    for m in args.models:
        r = eval_onnx(m, te)
        tag = "INT8" if "int8" in m else ("fp32" if "fp32" in m else m.split("/")[-1])
        print(f"  {tag:5s}: top1={r['top1']:.4f} top3={r['top3']:.4f} "
              f"top5={r['top5']:.4f} dba={r['dba']:.4f}")


if __name__ == "__main__":
    main()

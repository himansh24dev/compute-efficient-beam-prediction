"""INT8 accuracy for the revised models (reviewer comment 7 + Table 9 precision rows).

For each of the five validation-selected camera+GPS W=2 checkpoints: export fp32 ONNX
(numerical-equivalence checked), quantize statically (QDQ, per-channel INT8 weights,
per-tensor UINT8 activations, 200 calibration windows from the TRAINING partition) as
(a) whole-model and (b) Conv-only ("selective"), and score all three with onnxruntime on
the held-out test passes. Also records which operator types were quantized, so the
description of the selective model can be checked against the graph.

  .venv/bin/python -m experiments.revision.int8_eval
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from data.config import load_data_config
from utils.yaml_io import load_yaml
from experiments.bemamba_compare.splits_compare import build_compare_loaders
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics

SEEDS = [1337, 2024, 7, 42, 123]
ROOT = Path("experiments/revision/results")
PY = sys.executable


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def quantized_ops(path):
    """Operator types that consume a DequantizeLinear output of a quantized weight."""
    g = onnx.load(path).graph
    deq = {n.output[0] for n in g.node if n.op_type == "DequantizeLinear"}
    return dict(Counter(n.op_type for n in g.node
                        if n.op_type != "QuantizeLinear" and n.op_type != "DequantizeLinear"
                        and any(i in deq for i in n.input)))


def score(path, batches, save=None, meta=None):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    acc = new_accumulator(); P = []
    for g, c, t in batches:
        logits = np.concatenate([sess.run(None, {"gps": g[i:i + 1], "camera": c[i:i + 1]})[0]
                                 for i in range(len(g))])
        P.append(torch.softmax(torch.from_numpy(logits).float(), -1).half().numpy())
        add(acc, full_metrics(torch.from_numpy(logits), torch.from_numpy(t)))
    if save is not None:                      # same format as run_revision's .npz
        np.savez_compressed(save, prob=np.concatenate(P), **meta)
    r = finalize(acc)
    return {k: r[k] for k in ("dba", "top1", "top3", "top5")}


def main():
    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    exp["split"]["val_frac"] = 0.15
    cfg = load_data_config(exp["data_config"])
    _, _, te, info = build_compare_loaders(cfg, exp, ["gps", "camera"], "episode-random", 1337)
    batches, T, S, Rw = [], [], [], []
    for b in te:
        batches.append((b["inputs"]["gps"].numpy().astype(np.float32),
                        b["inputs"]["camera"].numpy().astype(np.float32), b["beam"].numpy()))
        T.append(b["beam"].numpy()); S.append(b["scenario_id"].numpy()); Rw.append(b["row"].numpy())
    key = info["episode_key"]
    meta = {"target": np.concatenate(T), "scenario_id": np.concatenate(S), "row": np.concatenate(Rw)}
    meta["passes"] = np.array([key(a, b) for a, b in zip(meta["scenario_id"], meta["row"])])
    RUNS = ROOT / "runs"
    res = {}
    for s in SEEDS:
        ck = ROOT / "ckpt" / f"A_ssm_2M_i{s}.pt"
        out = ROOT / "onnx" / f"s{s}"
        out.mkdir(parents=True, exist_ok=True)
        fp32 = out / "beam_ssm_fp32.onnx"
        if not (out / "beam_ssm_int8_selective.onnx").is_file():
            run([PY, "-m", "experiments.edge_deploy.export_onnx", "--ckpt", str(ck), "--window", "2",
                 "--out", str(out)])
            run([PY, "-m", "experiments.edge_deploy.quantize_int8", "--fp32", str(fp32),
                 "--out", str(out / "beam_ssm_int8.onnx")])
            run([PY, "-m", "experiments.edge_deploy.quantize_int8", "--fp32", str(fp32),
                 "--out", str(out / "beam_ssm_int8_selective.onnx"), "--op-types", "Conv"])
        res[s] = {"fp32": score(fp32, batches),
                  "int8_whole": score(out / "beam_ssm_int8.onnx", batches,
                                      RUNS / f"Q_int8_whole_i{s}.npz", meta),
                  "int8_selective": score(out / "beam_ssm_int8_selective.onnx", batches,
                                          RUNS / f"Q_int8_selective_i{s}.npz", meta),
                  "quantized_ops_whole": quantized_ops(str(out / "beam_ssm_int8.onnx")),
                  "quantized_ops_selective": quantized_ops(str(out / "beam_ssm_int8_selective.onnx"))}
        print(s, json.dumps(res[s]), flush=True)
    summ = {}
    for v in ("fp32", "int8_whole", "int8_selective"):
        summ[v] = {m: round(float(np.mean([res[s][v][m] for s in SEEDS])), 4)
                   for m in ("dba", "top1", "top3", "top5")}
        summ[v]["dba_sd"] = round(float(np.std([res[s][v]["dba"] for s in SEEDS], ddof=1)), 4)
    json.dump({"per_seed": res, "mean": summ}, open(ROOT / "int8_eval.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())

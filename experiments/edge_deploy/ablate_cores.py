"""Temporal-core comparison at the DEPLOYMENT operating point (cam+GPS, W=2,
leakage-free episode-random split). Trains the *same* network with each temporal
core — SSM (ours), Transformer, GRU, LSTM, MLP (no sequence model) — matched on
depth/width, and reports accuracy + parameter count + compute. Answers the
reviewer's "why a selective SSM and not a GRU / a plain MLP?"

  .venv/bin/python -m experiments.edge_deploy.ablate_cores \
      --cores ssm transformer gru lstm mlp --seeds 1337

Results append to results/ablate_cores.json keyed by "<core>_s<seed>" so partial
runs survive an interruption; re-running skips finished (core, seed) pairs unless
--force.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from data.config import load_data_config
from utils.device import get_device
from utils.yaml_io import load_yaml, save_json
from experiments.bemamba_compare.train_compare import train_one

OUT = "experiments/edge_deploy/results/ablate_cores.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", nargs="+", default=["ssm", "transformer", "gru", "lstm", "mlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337])
    ap.add_argument("--protocol", default="episode-random")
    ap.add_argument("--config", default="configs/experiments/deploy_w2.yaml")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    exp = load_yaml(args.config)
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    combo = ["gps", "camera"]

    results = {}
    if os.path.exists(args.out) and not args.force:
        results = json.load(open(args.out))

    for seed in args.seeds:
        for core in args.cores:
            key = f"{core}_s{seed}"
            if key in results and not args.force:
                print(f"skip {key} (done)", flush=True); continue
            print(f"\n=== {core} | {args.protocol} | seed {seed} ===", flush=True)
            t0 = time.time()
            r = train_one(cfg, exp, core, combo, args.protocol, seed, device=device)
            t = r["test"]
            row = {"core": core, "seed": seed, "protocol": args.protocol,
                   "top1": float(t["top1"]), "top3": float(t["top3"]),
                   "top5": float(t["top5"]), "dba": float(t["dba"]),
                   "params": int(r["params"]),
                   "gflops": r["efficiency"].get("gflops"),
                   "efficiency": {k: v for k, v in r["efficiency"].items() if k != "gflops"},
                   "train_time_s": round(r["train_time_s"], 1),
                   "wall_s": round(time.time() - t0, 1)}
            results[key] = row
            save_json(results, args.out)   # incremental
            print(f"  DONE {core}: dba={row['dba']:.4f} top1={row['top1']:.4f} "
                  f"top3={row['top3']:.4f} | {row['params']:,} params | "
                  f"{row['gflops']} GFLOPs | {row['wall_s']}s", flush=True)

    # readable summary table
    print("\n==== TEMPORAL CORE COMPARISON (episode-random, W=2, cam+GPS) ====")
    print(f"{'core':12s} {'seed':>5s} {'params':>10s} {'GFLOPs':>7s} {'DBA':>7s} {'Top-1':>7s} {'Top-3':>7s}")
    for key in sorted(results):
        r = results[key]
        g = f"{r['gflops']:.3f}" if isinstance(r.get("gflops"), (int, float)) else "  n/a"
        print(f"{r['core']:12s} {r['seed']:5d} {r['params']:10,d} {g:>7s} "
              f"{r['dba']:.4f} {r['top1']:.4f} {r['top3']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Accuracy-efficiency Pareto sweep + multi-seed rigor for the deployment model.
Trains cam+gps SSM at W=2 (episode-random, honest split) over:
  - a capacity sweep (model dim -> params/FLOPs) for the Pareto curve, and
  - extra seeds at dim=128 for mean+/-std.
Records (dim, seed, params, gflops, top1/3, dba) incrementally.
"""
from __future__ import annotations

import copy
from pathlib import Path

from data.config import load_data_config
from utils.device import get_device
from utils.yaml_io import load_yaml, save_json
from experiments.bemamba_compare.train_compare import train_one

# (dim, seed, tag). dim=128/seed=1337 already exists (deploy_w2) -> reused, not re-run.
POINTS = [
    (64, 1337, "pareto"), (96, 1337, "pareto"),
    (192, 1337, "pareto"), (256, 1337, "pareto"),
    (128, 2024, "seed"), (128, 7, "seed"),
]


def main():
    exp0 = load_yaml("configs/experiments/deploy_w2.yaml")
    cfg = load_data_config(exp0["data_config"])
    device = get_device(cfg)
    out = Path(exp0["results_dir"]); out.mkdir(parents=True, exist_ok=True)

    results = []
    for dim, seed, tag in POINTS:
        exp = copy.deepcopy(exp0)
        exp["model"]["dim"] = dim
        exp["sweep"]["seeds"] = [seed]
        print(f"\n=== dim={dim} seed={seed} ({tag}) | W=2 episode-random ===", flush=True)
        r = train_one(cfg, exp, "ssm", ["gps", "camera"], "episode-random", seed, device=device)
        t = r["test"]; e = r["efficiency"]
        row = {"dim": dim, "seed": seed, "tag": tag, "params": int(r["params"]),
               "gflops": (round(e["gflops"], 4) if e.get("gflops") == e.get("gflops") else None),
               "top1": float(t["top1"]), "top3": float(t["top3"]), "dba": float(t["dba"])}
        results.append(row)
        print(f"  RESULT dim={dim} seed={seed}: params={r['params']:,} "
              f"gflops={row['gflops']} dba={t['dba']:.4f} top3={t['top3']:.4f}", flush=True)
        save_json(results, out / "sweep_capacity_seeds.json")   # incremental
    print("\nsweep done.", flush=True)


if __name__ == "__main__":
    main()

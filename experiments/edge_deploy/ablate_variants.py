"""Architecture-variant ablations at the deployment point (SSM, cam+GPS, W=2,
episode-random). Two axes a reviewer asks about:

  * SSM depth   — core_layers in {2, 4, 6}  (4 = the deployed model, already in
                  ablate_cores.json as ssm_s1337; here we add 2 and 6).
  * Fusion      — gated (deployed) vs mean vs concat  (gated = ssm_s1337).

Each variant deep-copies the deploy config, overrides ONE model field, and reuses
the tested trainer. Results append to results/ablate_variants.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

from data.config import load_data_config
from utils.device import get_device
from utils.yaml_io import load_yaml, save_json
from experiments.bemamba_compare.train_compare import train_one

OUT = "experiments/edge_deploy/results/ablate_variants.json"

# name -> (core, model-cfg overrides)
VARIANTS = {
    "ssm_L2":          ("ssm", {"core_layers": 2}),
    "ssm_L6":          ("ssm", {"core_layers": 6}),
    "ssm_fuse_mean":   ("ssm", {"fusion_kind": "mean"}),
    "ssm_fuse_concat": ("ssm", {"fusion_kind": "concat"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--protocol", default="episode-random")
    ap.add_argument("--config", default="configs/experiments/deploy_w2.yaml")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    exp0 = load_yaml(args.config)
    cfg = load_data_config(exp0["data_config"])
    device = get_device(cfg)
    combo = ["gps", "camera"]

    results = {}
    if os.path.exists(args.out) and not args.force:
        results = json.load(open(args.out))

    for name in args.only:
        if name in results and not args.force:
            print(f"skip {name} (done)", flush=True); continue
        core, over = VARIANTS[name]
        exp = copy.deepcopy(exp0)
        exp["model"].update(over)
        print(f"\n=== {name}: core={core} overrides={over} | {args.protocol} | seed {args.seed} ===", flush=True)
        t0 = time.time()
        r = train_one(cfg, exp, core, combo, args.protocol, args.seed, device=device)
        t = r["test"]
        row = {"name": name, "core": core, "overrides": over, "seed": args.seed,
               "top1": float(t["top1"]), "top3": float(t["top3"]),
               "top5": float(t["top5"]), "dba": float(t["dba"]),
               "params": int(r["params"]), "gflops": r["efficiency"].get("gflops"),
               "train_time_s": round(r["train_time_s"], 1), "wall_s": round(time.time() - t0, 1)}
        results[name] = row
        save_json(results, args.out)
        print(f"  DONE {name}: dba={row['dba']:.4f} top1={row['top1']:.4f} "
              f"top3={row['top3']:.4f} | {row['params']:,} params", flush=True)

    print("\n==== ARCHITECTURE VARIANTS (episode-random, W=2, cam+GPS) ====")
    print(f"{'variant':18s} {'params':>10s} {'DBA':>7s} {'Top-1':>7s} {'Top-3':>7s}")
    for name in sorted(results):
        r = results[name]
        print(f"{name:18s} {r['params']:10,d} {r['dba']:.4f} {r['top1']:.4f} {r['top3']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())

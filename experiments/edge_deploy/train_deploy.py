"""Train the clean W=2 cam+gps SSM DEPLOYMENT model(s) and SAVE the checkpoint,
so the accuracy we report and the Pi latency we measure use the exact same
weights. Reuses the tested bemamba_compare trainer; one model per protocol.

  .venv/bin/python -m experiments.edge_deploy.train_deploy --config configs/experiments/deploy_w2.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.config import load_data_config
from utils.device import get_device
from utils.yaml_io import load_yaml, save_json
from experiments.bemamba_compare.train_compare import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/deploy_w2.yaml")
    args = ap.parse_args()

    exp = load_yaml(args.config)
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    out_dir = Path(exp["results_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    combo = ["gps", "camera"]
    seed = int(exp["sweep"]["seeds"][0])
    W = int(exp["window"])

    summary = {"window": W, "seed": seed, "modalities": combo, "protocols": {}}
    for protocol in exp["sweep"]["protocols"]:
        print(f"\n=== training W={W} deployment model | {protocol} | seed {seed} ===", flush=True)
        r = train_one(cfg, exp, "ssm", combo, protocol, seed, device=device)
        t = r["test"]
        print(f"  RESULT: top1={t['top1']:.4f} top3={t['top3']:.4f} top5={t['top5']:.4f} "
              f"dba={t['dba']:.4f} | {r['params']:,} params", flush=True)
        ckpt = {"model": r["best_state"],
                "val": {k: float(t[k]) for k in ("top1", "top3", "top5", "dba")},
                "modalities": combo, "core_kind": "ssm", "window": W,
                "seed": seed, "protocol": protocol}
        path = out_dir / f"deploy_ssm_w{W}_{protocol}_s{seed}.pt"
        torch.save(ckpt, path)
        print(f"  saved checkpoint -> {path}", flush=True)
        ps = t.get("per_scenario", {})
        summary["protocols"][protocol] = {
            "top1": float(t["top1"]), "top3": float(t["top3"]), "top5": float(t["top5"]),
            "dba": float(t["dba"]), "params": int(r["params"]),
            "per_scenario_top3": {str(k): float(v["top3"]) for k, v in ps.items()},
            "per_scenario_dba": {str(k): float(v["dba"]) for k, v in ps.items()}}
    save_json(summary, out_dir / f"deploy_w{W}_accuracy.json")
    print("\ndeploy training done.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

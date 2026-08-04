"""Reviewer C9: train the deployed model WITH GPS-noise augmentation and re-run the
robustness sweep, to test whether the positioning-limit (Fig. 6) can be mitigated.
Trains cam+GPS W=2 (episode-random, seed 1337) with model.gps_noise_std set, saves
the checkpoint so robustness_full.py can re-plot the GPS-error curve on it.

  .venv/bin/python -m experiments.edge_deploy.gps_noise_train --std 0.5
"""
from __future__ import annotations

import argparse
import copy

import torch

from data.config import load_data_config
from utils.device import get_device
from utils.yaml_io import load_yaml
from experiments.bemamba_compare.train_compare import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", type=float, default=0.5, help="GPS-noise std (standardized units)")
    args = ap.parse_args()
    exp = copy.deepcopy(load_yaml("configs/experiments/deploy_w2.yaml"))
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    exp["model"]["gps_noise_std"] = args.std
    print(f"=== training cam+GPS W=2 with gps_noise_std={args.std} (episode-random, seed 1337) ===", flush=True)
    r = train_one(cfg, exp, "ssm", ["gps", "camera"], "episode-random", 1337, device=device)
    out = f"experiments/edge_deploy/results/deploy_ssm_w2_gpsaug{args.std}_s1337.pt"
    torch.save({"model": r["best_state"], "gps_noise_std": args.std}, out)
    t = r["test"]
    print(f"CLEAN-test: dba={t['dba']:.4f} top3={t['top3']:.4f} top1={t['top1']:.4f}", flush=True)
    print(f"saved -> {out}\nNext: robustness_full.py --ckpt {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

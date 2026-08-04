"""Multi-seed rigor (reviewer C5) + seed-ensemble accuracy chase.

Two modes:
  --mode ci        : train each (core, seed) independently (seed drives split+init),
                     report per-core mean +/- std +/- 95% CI over >=5 seeds, and a
                     paired SSM-vs-Transformer bootstrap (same split per seed).
  --mode ensemble  : fix the split (seeds[0]) and vary only init_seed, then average
                     softmax logits over the seed models on the shared test set --
                     an honest accuracy point (N x compute, not the deployed model).

  .venv/bin/python -m experiments.edge_deploy.multiseed --mode ci \
      --cores ssm transformer --seeds 1337 2024 7 42 123 --protocol episode-random
  .venv/bin/python -m experiments.edge_deploy.multiseed --mode ensemble \
      --cores ssm --seeds 1337 2024 7 42 123 --protocol window-random
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from data.config import load_data_config
from utils.device import get_device
from utils.yaml_io import load_yaml, save_json
from models.beam_model import BeamModel
from experiments.bemamba_compare.train_compare import train_one, evaluate_full
from experiments.bemamba_compare.splits_compare import build_compare_loaders
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics


def _ci95(xs):
    x = np.array(xs, float); n = len(x)
    if n < 2:
        return float(x.mean()), 0.0, 0.0
    sd = x.std(ddof=1)
    return float(x.mean()), float(sd), float(1.96 * sd / math.sqrt(n))


def run_ci(cfg, exp, device, cores, seeds, protocol, out):
    combo = ["gps", "camera"]
    per = {c: {} for c in cores}   # core -> seed -> metrics
    for s in seeds:
        for c in cores:
            r = train_one(cfg, exp, c, combo, protocol, s, device=device)
            t = r["test"]
            per[c][s] = {"dba": float(t["dba"]), "top1": float(t["top1"]), "top3": float(t["top3"])}
            print(f"  {c} s{s}: dba={t['dba']:.4f} top3={t['top3']:.4f}", flush=True)
    res = {"protocol": protocol, "seeds": seeds, "per": per, "summary": {}}
    for c in cores:
        for m in ("dba", "top1", "top3"):
            mean, sd, ci = _ci95([per[c][s][m] for s in seeds])
            res["summary"].setdefault(c, {})[m] = {"mean": round(mean, 4), "std": round(sd, 4), "ci95": round(ci, 4)}
    # paired SSM-vs-Transformer (same split per seed)
    if "ssm" in cores and "transformer" in cores:
        diffs = [per["ssm"][s]["dba"] - per["transformer"][s]["dba"] for s in seeds]
        mean, sd, ci = _ci95(diffs)
        res["paired_ssm_minus_xf_dba"] = {"mean": round(mean, 4), "ci95": round(ci, 4),
                                          "wins": int(sum(d > 0 for d in diffs)), "n": len(diffs)}
        print(f"\npaired SSM-Transformer DBA: {mean:+.4f} +/- {ci:.4f} "
              f"(SSM wins {res['paired_ssm_minus_xf_dba']['wins']}/{len(diffs)})", flush=True)
    save_json(res, out)
    print("\n==== MULTI-SEED CI ({}), {} seeds ====".format(protocol, len(seeds)))
    for c in cores:
        d = res["summary"][c]
        print(f"  {c:12s} DBA {d['dba']['mean']:.4f}+/-{d['dba']['ci95']:.4f}  "
              f"Top-3 {d['top3']['mean']:.4f}+/-{d['top3']['ci95']:.4f}")


@torch.no_grad()
def _logits(model, loader, device):
    model.eval(); L, Y = [], []
    for b in loader:
        inp = {k: v.to(device) for k, v in b["inputs"].items()}
        o = model(inp); o = (o[0] if isinstance(o, tuple) else o).float()
        L.append(torch.softmax(o, -1).cpu()); Y.append(b["beam"])
    return torch.cat(L), torch.cat(Y)


def run_ensemble(cfg, exp, device, core, seeds, protocol, out):
    combo = ["gps", "camera"]
    split = seeds[0]
    states, singles = [], []
    for s in seeds:
        r = train_one(cfg, exp, core, combo, protocol, split, device=device, init_seed=s)
        states.append(r["best_state"]); singles.append(float(r["test"]["dba"]))
        print(f"  init s{s} (split {split}): dba={r['test']['dba']:.4f}", flush=True)
    _, te, _ = build_compare_loaders(cfg, exp, combo, protocol, split)
    nb = int(cfg.beam["num_beams"])
    probs = None; Y = None
    for st in states:
        m = BeamModel(combo, nb, core, exp["model"]).to(device); m.load_state_dict(st)
        p, Y = _logits(m, te, device)
        probs = p if probs is None else probs + p
    probs /= len(states)
    acc = new_accumulator(); add(acc, full_metrics(probs, Y)); e = finalize(acc)
    res = {"protocol": protocol, "core": core, "split_seed": split, "init_seeds": seeds,
           "single_mean_dba": round(float(np.mean(singles)), 4),
           "ensemble": {"dba": e["dba"], "top1": e["top1"], "top3": e["top3"]},
           "bemamba_camgps": {"dba": 0.877, "top3": 0.860}}
    save_json(res, out)
    print(f"\n==== {len(seeds)}-SEED ENSEMBLE ({protocol}) ====")
    print(f"  single-model mean DBA {res['single_mean_dba']:.4f}")
    print(f"  ENSEMBLE  DBA {e['dba']:.4f}  Top-3 {e['top3']:.4f}  (BeMamba cam+gps 0.877/0.860)")
    if e["dba"] > 0.877 and e["top3"] > 0.860:
        print("  --> ensemble beats BeMamba cam+gps on BOTH metrics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ci", "ensemble"], default="ci")
    ap.add_argument("--cores", nargs="+", default=["ssm", "transformer"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 2024, 7, 42, 123])
    ap.add_argument("--protocol", default="episode-random")
    ap.add_argument("--config", default="configs/experiments/deploy_w2.yaml")
    args = ap.parse_args()
    exp = load_yaml(args.config)
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    tag = f"{args.mode}_{args.protocol}"
    out = f"experiments/edge_deploy/results/multiseed_{tag}.json"
    if args.mode == "ci":
        run_ci(cfg, exp, device, args.cores, args.seeds, args.protocol, out)
    else:
        run_ensemble(cfg, exp, device, args.cores[0], args.seeds, args.protocol, out)


if __name__ == "__main__":
    raise SystemExit(main())

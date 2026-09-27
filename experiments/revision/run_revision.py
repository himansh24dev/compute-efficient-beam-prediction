"""Revision re-run (Scientific Reports, reviewer comments 2-3).

Every accuracy number in the paper is regenerated with checkpoint selection on a
pass-disjoint VALIDATION partition carved from the training passes (see
splits_compare.py); the 23 held-out test passes are the same as in the original
submission and are evaluated exactly once per model. Each job saves

  results/runs/<name>.json   metrics, validation-selected epoch, per-epoch history,
                             split description (train / val / test pass keys)
  results/runs/<name>.npz    per-sample test softmax, target, scenario, row, pass
  results/ckpt/<name>.pt     weights + the split's train-only GPS statistics

so the pass-level bootstrap, paired differences and seed ensembles need no
re-inference. Jobs are idempotent (finished ones are skipped), and `--shard i/n`
splits the registry across concurrent GPU workers.

  .venv/bin/python -m experiments.revision.run_revision --group all --shard 0/3
  .venv/bin/python -m experiments.revision.run_revision --list
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from data.config import load_data_config
from utils.device import get_device
from utils.yaml_io import load_yaml, save_json
from experiments.bemamba_compare.train_compare import train_one

ROOT = Path("experiments/revision/results")
BASE_CFG = "configs/experiments/deploy_w2.yaml"
PRETRAINED_CFG = "configs/experiments/pretrained_2m.yaml"
SPLIT_SEED = 1337                              # the original headline split (23 test passes)
INIT_SEEDS = [1337, 2024, 7, 42, 123]          # five initialisations on that split
LEAK_SEEDS = [1337, 2024, 7, 42, 123]          # leakage table: seed drives split + init
CAM_GPS = ["gps", "camera"]
MODS = {"2M": CAM_GPS, "gps": ["gps"], "cam": ["camera"],
        "3M": ["gps", "camera", "lidar"], "4M": ["gps", "camera", "lidar", "radar"]}


def job(name, group, *, core="ssm", mods="2M", protocol="episode-random", seed=SPLIT_SEED,
        init=None, window=2, model=None, exp=None, base=BASE_CFG, save_ckpt=True):
    return {"name": name, "group": group, "core": core, "mods": mods, "protocol": protocol,
            "seed": seed, "init": init if init is not None else seed, "window": window,
            "model": model or {}, "exp": exp or {}, "base": base, "save_ckpt": save_ckpt}


def registry() -> list[dict]:
    J = []
    # (A) headline split, W=2, five inits: deployed model + every ablation arm, so all
    #     ablations are paired on the same 23 held-out passes (reviewer comment 3).
    for s in INIT_SEEDS:
        J.append(job(f"A_ssm_2M_i{s}", "A", init=s))
        J.append(job(f"A_ssm_gps_i{s}", "A", mods="gps", init=s))
        J.append(job(f"A_ssm_cam_i{s}", "A", mods="cam", init=s))
        for core in ("transformer", "gru", "lstm", "mlp"):
            J.append(job(f"A_{core}_2M_i{s}", "A", core=core, init=s))
    for s in INIT_SEEDS:
        J.append(job(f"B_ssm_3M_i{s}", "B", mods="3M", init=s))
        J.append(job(f"B_ssm_4M_i{s}", "B", mods="4M", init=s))
        J.append(job(f"B_ssm_L2_i{s}", "B", init=s, model={"core_layers": 2}))
        J.append(job(f"B_ssm_L6_i{s}", "B", init=s, model={"core_layers": 6}))
        J.append(job(f"B_fuse_mean_i{s}", "B", init=s, model={"fusion_kind": "mean"}))
        J.append(job(f"B_fuse_concat_i{s}", "B", init=s, model={"fusion_kind": "concat"}))
        J.append(job(f"B_pretrained_i{s}", "B", init=s, base=PRETRAINED_CFG,
                     model={"camera": {"pretrained": True}},
                     exp={"train": {"batch_size": 48, "backbone_lr_scale": 0.1}}))
    # (C) split-protocol leakage at W=2 and W=5 (cam+GPS); seed drives split + init,
    #     as in the original Table 2, now over five seeds.
    for W in (2, 5):
        for p in ("window-random", "episode-random"):
            for s in LEAK_SEEDS:
                if W == 2 and p == "episode-random" and s == SPLIT_SEED:
                    continue                       # identical to A_ssm_2M_i1337
                J.append(job(f"C_w{W}_{p}_s{s}", "C", protocol=p, seed=s, window=W,
                             exp={"train": {"batch_size": 96 if W == 2 else 64}}))
    # (C') Table 1 rows (window-random, W=5, +LiDAR / +radar), five seeds
    for m in ("3M", "4M"):
        for s in LEAK_SEEDS:
            J.append(job(f"C_w5_window-random_{m}_s{s}", "C", mods=m, protocol="window-random",
                         seed=s, window=5, exp={"train": {"batch_size": 64}}))
    # (D) delay-aligned: one fixed anchor set (future_horizons [1,2] for every model)
    for h in (0, 1, 2):
        for s in INIT_SEEDS:
            J.append(job(f"D_h{h}_i{s}", "D", init=s,
                         exp={"future_horizons": [1, 2], "target_h": h}))
    # (E) window sweep on the headline split; anchor_window=16 keeps the sample set
    #     identical across W, so the curve is a controlled comparison.
    for W in (2, 4, 8, 16):
        for s in INIT_SEEDS[:3]:
            J.append(job(f"E_w{W}_i{s}", "E", init=s, window=W,
                         exp={"anchor_window": 16,
                              "train": {"batch_size": {2: 96, 4: 96, 8: 64, 16: 48}[W]}}))
    return J


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def run_job(j, device, epochs=None):
    out_json = ROOT / "runs" / f"{j['name']}.json"
    if out_json.is_file():
        print(f"skip {j['name']} (done)", flush=True); return
    exp = load_yaml(j["base"])
    exp.setdefault("split", {})["val_frac"] = 0.15
    exp["window"] = j["window"]
    exp.setdefault("model", {})
    _deep_update(exp["model"], copy.deepcopy(j["model"]))
    _deep_update(exp, copy.deepcopy(j["exp"]))
    if epochs:
        exp["train"]["epochs"] = epochs
    cfg = load_data_config(exp["data_config"])
    t0 = time.time()
    print(f"\n=== {j['name']} | {j['core']} {j['mods']} {j['protocol']} W={j['window']} "
          f"split={j['seed']} init={j['init']} ===", flush=True)
    r = train_one(cfg, exp, j["core"], MODS[j["mods"]], j["protocol"], j["seed"],
                  device=device, verbose=False, init_seed=j["init"])
    t = r["test"]; pr = t.pop("preds")
    tl = r["test_last"]; prl = tl.pop("preds")
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "runs" / f"{j['name']}.npz", prob=pr["prob"],
                        target=pr["target"], scenario_id=pr["scenario_id"], row=pr["row"],
                        passes=np.asarray(pr["pass"]))
    np.savez_compressed(ROOT / "runs" / f"{j['name']}__last.npz", prob=prl["prob"],
                        target=prl["target"], scenario_id=prl["scenario_id"], row=prl["row"],
                        passes=np.asarray(pr["pass"]))
    si = r["split_info"]
    rec = {"job": {k: v for k, v in j.items()}, "exp": exp,
           "test": {k: v for k, v in t.items() if k != "per_scenario"},
           "test_per_scenario": {str(k): v for k, v in t.get("per_scenario", {}).items()},
           "test_last_epoch": {k: v for k, v in tl.items() if k != "per_scenario"},
           "val_best": r["val_best"], "history": r["history"], "params": r["params"],
           "efficiency": r["efficiency"], "train_time_s": round(r["train_time_s"], 1),
           "wall_s": round(time.time() - t0, 1), "split_info": si,
           "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    if j["save_ckpt"]:
        (ROOT / "ckpt").mkdir(parents=True, exist_ok=True)
        torch.save({"model": r["best_state"], "modalities": MODS[j["mods"]], "core_kind": j["core"],
                    "window": j["window"], "model_cfg": exp["model"], "gps_stats": si.get("gps_stats"),
                    "split_seed": j["seed"], "init_seed": j["init"], "protocol": j["protocol"],
                    "val_best": r["val_best"]}, ROOT / "ckpt" / f"{j['name']}.pt")
    save_json(rec, out_json)
    print(f"  DONE {j['name']}: test dba={t['dba']:.4f} top1={t['top1']:.4f} top3={t['top3']:.4f} "
          f"| val-selected epoch {r['val_best'].get('epoch', -1) + 1} | last-epoch dba={tl['dba']:.4f} "
          f"| {rec['wall_s']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", help="A,B,C,D,E or all (comma-separated)")
    ap.add_argument("--only", nargs="*", default=None, help="explicit job names")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    groups = None if args.group == "all" else set(args.group.split(","))
    jobs = [j for j in registry() if (groups is None or j["group"] in groups)]
    if args.only:
        jobs = [j for j in jobs if j["name"] in set(args.only)]
    i, n = (int(x) for x in args.shard.split("/"))
    jobs = jobs[i::n]
    if args.list:
        for j in jobs:
            done = (ROOT / "runs" / f"{j['name']}.json").is_file()
            print(f"{'[x]' if done else '[ ]'} {j['name']}")
        print(f"{len(jobs)} jobs"); return
    cfg = load_data_config("configs/data.yaml")
    device = get_device(cfg)
    for j in jobs:
        try:
            run_job(j, device, epochs=args.epochs)
        except Exception:
            print(f"  FAIL {j['name']}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

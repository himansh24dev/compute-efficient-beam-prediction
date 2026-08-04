#!/usr/bin/env python3
"""Context-horizon sweep: val accuracy vs window length W, for SSM & Transformer.

Answers the linchpin question behind the KV-cache claim: does the task need long
temporal context? Reuses the Phase-1 trainer; varies cfg.snapshot.window per run.

    .venv/bin/python -m experiments.context_window.run_context
    .venv/bin/python -m experiments.context_window.run_context --smoke

Resumable (skips logged runs) and per-run robust. Frozen split (train/val S31+S32).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.config import load_data_config
from models.beam_model import BeamModel
from models.param_utils import core_parity
from utils.device import get_device
from utils.seed import seed_everything
from utils.yaml_io import load_yaml, save_json

from experiments.phase1.train import get_splits, train_core

FIELDS = ["date", "config_hash", "window", "core", "seed", "batch",
          "val_top1", "val_top5", "val_dba", "s33_top1",
          "core_params", "train_time_s", "epochs", "status"]


def _hash(exp):
    keep = {k: exp[k] for k in ("model", "combo", "data_config") if k in exp}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def safe_loader(cfg, modalities, window, batch, budget_gb=6.0, prefetch=2, max_workers=18):
    """Cap DataLoader workers so the prefetch buffer stays under a RAM budget.

    Each sample carries `window` frames; at large W a sample is many MB, and
    (workers * prefetch * batch * per_sample) can reach tens of GB in /dev/shm ->
    the OOM killer fires and takes down whatever has the highest oom_score (often
    the editor). Bounding the buffer here prevents that regardless of window size.
    """
    from data.cache import modality_shape
    per_frame = sum(math.prod(modality_shape(cfg, m)) * 4 for m in modalities)  # float32
    per_sample = max(window * per_frame, 1)
    workers = int(budget_gb * 1e9 / (prefetch * batch * per_sample))
    return max(2, min(max_workers, workers)), prefetch


def _done(csv_path, h):
    s = set()
    if csv_path.is_file():
        for r in csv.DictReader(open(csv_path)):
            if r.get("config_hash") == h and r.get("status") == "ok":
                s.add((int(r["window"]), r["core"], int(r["seed"])))
    return s


def _append(csv_path, row):
    ex = csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not ex:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/context_window.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    exp = load_yaml(args.config)
    smoke = args.smoke
    if smoke:
        exp["train"]["epochs"] = 1
        exp["windows"] = [4, 16]
        exp["cores"] = ["ssm", "transformer"]
        exp["seeds"] = [1337]

    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    num_beams = int(cfg.beam["num_beams"])
    combo = list(exp["combo"])
    h = _hash(exp)
    today = dt.date.today().isoformat()
    results_dir = Path(exp["results_dir"])
    log_csv = results_dir / ("context_log_smoke.csv" if smoke else "context_log.csv")

    windows = list(exp["windows"]); cores = list(exp["cores"]); seeds = list(exp["seeds"])
    batch_by_w = {int(k): int(v) for k, v in exp.get("batch_by_window", {}).items()}
    total = len(windows) * len(cores) * len(seeds)
    print(f"=== Context-horizon sweep | cfg {h} | device {device} | combo {'+'.join(combo)} ===")
    print(f"  windows={windows} cores={cores} seeds={seeds} -> {total} runs")

    splits = get_splits(cfg, smoke=False)  # frozen split (train/val S31+S32, test S33)
    print(f"  split: {splits.summary()}")
    # Controlled comparison: fix the anchor set across all window sizes so every
    # W sees identical samples (see config note). Must be >= max(windows).
    anchor_w = int(exp.get("anchor_window", max(windows)))
    if anchor_w < max(windows):
        raise ValueError(f"anchor_window {anchor_w} < max window {max(windows)}")
    cfg.snapshot["anchor_window"] = anchor_w
    print(f"  anchor_window={anchor_w} (sample set held FIXED across all W)")

    # Parity (window-independent).
    seed_everything(int(exp["seed"]))
    sp = BeamModel(combo, num_beams, "ssm", exp["model"]); xp = BeamModel(combo, num_beams, "transformer", exp["model"])
    par = core_parity(sp.core_module(), xp.core_module(), tol=float(exp["model"]["param_parity_tol"]))
    print(f"[parity] ssm={par['ssm_core_params']:,} xf={par['transformer_core_params']:,} "
          f"rel={par['relative_diff']*100:.1f}%")
    del sp, xp

    done = _done(log_csv, h)
    if done:
        print(f"[resume] skipping {len(done)} logged runs")

    results = {}
    i = 0
    for W in windows:
        cfg.snapshot["window"] = int(W)                         # <-- the swept variable
        exp["train"]["batch_size"] = batch_by_w.get(int(W), exp["train"]["batch_size"])
        # Bound DataLoader RAM per window so large W can't OOM the machine.
        nw, pf = safe_loader(cfg, combo, int(W), exp["train"]["batch_size"],
                             max_workers=int(exp["train"].get("num_workers", 18)))
        exp["train"]["num_workers"] = nw
        exp["train"]["prefetch_factor"] = pf
        print(f"  [W={W}] batch={exp['train']['batch_size']} workers={nw} prefetch={pf} "
              f"(RAM-bounded)")
        for core in cores:
            for seed in seeds:
                i += 1
                if (int(W), core, seed) in done:
                    print(f"[{i}/{total}] SKIP W={W} {core} s{seed}")
                    continue
                tag = f"{h}_W{W}_{core}_s{seed}"
                print(f"\n[{i}/{total}] W={W} batch={exp['train']['batch_size']} | {core} | seed {seed}")
                try:
                    res = train_core(cfg, exp, core, combo, splits, device=device, seed=seed,
                                     verbose=True, save_dir=results_dir, tag=tag)
                    bv = res["best_val"]; te = res.get("test_s33", {})
                    results[(int(W), core, seed)] = bv["top1"]
                    print(f"  => W={W} {core}: val_top1={bv['top1']:.4f}")
                    _append(log_csv, {"date": today, "config_hash": h, "window": W, "core": core,
                                      "seed": seed, "batch": exp["train"]["batch_size"],
                                      "val_top1": round(bv["top1"], 4), "val_top5": round(bv["top5"], 4),
                                      "val_dba": round(bv["dba"], 4),
                                      "s33_top1": round(te.get("top1", float("nan")), 4) if te else "",
                                      "core_params": res["core_params"],
                                      "train_time_s": round(res["train_time_s"], 1),
                                      "epochs": exp["train"]["epochs"], "status": "ok"})
                except Exception as e:
                    traceback.print_exc()
                    _append(log_csv, {"date": today, "config_hash": h, "window": W, "core": core,
                                      "seed": seed, "batch": exp["train"].get("batch_size", ""),
                                      "epochs": exp["train"]["epochs"], "status": f"FAIL:{type(e).__name__}"})
                    print(f"  !! FAILED: {e}")

    # Horizon curve.
    print("\n" + "=" * 64)
    print("CONTEXT HORIZON — val Top-1 vs window W (mean over seeds)")
    summary = {"config_hash": h, "parity": par, "curve": {}}
    for core in cores:
        print(f"  {core}:")
        summary["curve"][core] = {}
        for W in windows:
            vs = [results[(int(W), core, s)] for s in seeds if (int(W), core, s) in results]
            if vs:
                m = statistics.mean(vs)
                summary["curve"][core][str(W)] = m
                bar = "█" * int(m * 60)
                print(f"    W={W:>3}: {m:.4f} {bar}")
    print("=" * 64)
    save_json(summary, results_dir / f"context_summary{'_smoke' if smoke else ''}_{h}.json")
    print(f"log -> {log_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

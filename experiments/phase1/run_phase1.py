#!/usr/bin/env python3
"""Phase-1 sweep: full-multimodal parity + modality ablation + day->night (S33)
zero-shot, over {modality combos} x {ssm, transformer} x {seeds}.

    .venv/bin/python -m experiments.phase1.run_phase1                 # full sweep (frozen split)
    .venv/bin/python -m experiments.phase1.run_phase1 --smoke         # 1-epoch code check on S31 cache

RESUMABLE: runs already present in experiment_log_phase1.csv (same config hash)
are skipped, so re-running continues where it left off. ROBUST: each run is
wrapped in try/except; a failure is logged and the sweep continues.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
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

LOG_FIELDS = ["date", "config_hash", "combo", "seed", "core",
              "val_top1", "val_top5", "val_dba",
              "s33_top1", "s33_top5", "s33_dba",
              "core_params", "total_params", "train_time_s", "epochs", "status"]


def _base_hash(exp: dict) -> str:
    keep = {k: exp[k] for k in ("model", "train", "data_config") if k in exp}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def _load_done(csv_path: Path, cfg_hash: str) -> set:
    done = set()
    if csv_path.is_file():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r.get("config_hash") == cfg_hash and r.get("status") == "ok":
                    done.add((r["combo"], r["core"], int(r["seed"])))
    return done


def _append(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/phase1.yaml")
    ap.add_argument("--smoke", action="store_true", help="1-epoch code check on S31 cache")
    args = ap.parse_args()

    exp = load_yaml(args.config)
    smoke = args.smoke
    if smoke:
        exp["train"]["epochs"] = 1
        exp["sweep"] = {"modality_combos": [["gps", "camera", "lidar", "radar"]],
                        "cores": ["ssm", "transformer"], "seeds": [1337]}

    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    num_beams = int(cfg.beam["num_beams"])
    cfg_hash = _base_hash(exp)
    today = dt.date.today().isoformat()
    results_dir = Path(exp["results_dir"])
    log_csv = results_dir / ("experiment_log_phase1_smoke.csv" if smoke else "experiment_log_phase1.csv")

    combos = [list(c) for c in exp["sweep"]["modality_combos"]]
    cores = list(exp["sweep"]["cores"])
    seeds = list(exp["sweep"]["seeds"])
    total = len(combos) * len(cores) * len(seeds)

    print(f"=== Phase 1 sweep | cfg {cfg_hash} | device {device} | smoke={smoke} ===")
    print(f"  combos={[('+'.join(c)) for c in combos]}")
    print(f"  cores={cores}  seeds={seeds}  epochs={exp['train']['epochs']}  -> {total} runs")

    # Splits (frozen v1 for the real run; S31-only for smoke).
    splits = get_splits(cfg, smoke=smoke)
    print(f"  split: {splits.summary()}")

    # Core parameter parity (seed/modality independent).
    seed_everything(int(exp["seed"]))
    probe_mods = ["gps", "camera", "lidar", "radar"]
    ssm_probe = BeamModel(probe_mods, num_beams, "ssm", exp["model"])
    xf_probe = BeamModel(probe_mods, num_beams, "transformer", exp["model"])
    parity = core_parity(ssm_probe.core_module(), xf_probe.core_module(),
                         tol=float(exp["model"]["param_parity_tol"]))
    print(f"[parity] ssm_core={parity['ssm_core_params']:,} xf_core={parity['transformer_core_params']:,} "
          f"rel_diff={parity['relative_diff']*100:.1f}% within_tol={parity['within_tol']}")
    del ssm_probe, xf_probe

    done = _load_done(log_csv, cfg_hash)
    if done:
        print(f"[resume] {len(done)} runs already logged for this config -> skipping them")

    results = {}   # (combo_label, core, seed) -> result
    run_i = 0
    for combo in combos:
        label = "+".join(combo)
        for core in cores:
            for seed in seeds:
                run_i += 1
                key = (label, core, seed)
                if key in done:
                    print(f"[{run_i}/{total}] SKIP (done): {label} | {core} | seed {seed}")
                    continue
                tag = f"{cfg_hash}_{label}_{core}_s{seed}"
                print(f"\n[{run_i}/{total}] RUN: {label} | {core} | seed {seed}")
                try:
                    res = train_core(cfg, exp, core, combo, splits, device=device, seed=seed,
                                     verbose=True, save_dir=results_dir, tag=tag)
                    bv, te = res["best_val"], res.get("test_s33", {})
                    results[key] = res
                    _append(log_csv, {
                        "date": today, "config_hash": cfg_hash, "combo": label, "seed": seed, "core": core,
                        "val_top1": round(bv["top1"], 4), "val_top5": round(bv["top5"], 4),
                        "val_dba": round(bv["dba"], 4),
                        "s33_top1": round(te.get("top1", float("nan")), 4) if te else "",
                        "s33_top5": round(te.get("top5", float("nan")), 4) if te else "",
                        "s33_dba": round(te.get("dba", float("nan")), 4) if te else "",
                        "core_params": res["core_params"], "total_params": res["total_params"],
                        "train_time_s": round(res["train_time_s"], 1),
                        "epochs": exp["train"]["epochs"], "status": "ok",
                    })
                    print(f"  => val_top1={bv['top1']:.4f} | S33 zero-shot top1="
                          f"{te.get('top1', float('nan')):.4f} | {res['train_time_s']:.0f}s")
                except Exception as e:  # keep the overnight sweep alive
                    traceback.print_exc()
                    _append(log_csv, {"date": today, "config_hash": cfg_hash, "combo": label,
                                      "seed": seed, "core": core, "val_top1": "", "val_top5": "",
                                      "val_dba": "", "s33_top1": "", "s33_top5": "", "s33_dba": "",
                                      "core_params": "", "total_params": "", "train_time_s": "",
                                      "epochs": exp["train"]["epochs"], "status": f"FAIL:{type(e).__name__}"})
                    print(f"  !! FAILED: {e} (logged; continuing)")

    _summarize(results, combos, cores, seeds, cfg_hash, parity, results_dir, smoke)
    print(f"\nlog -> {log_csv}")
    return 0


def _summarize(results, combos, cores, seeds, cfg_hash, parity, results_dir, smoke):
    def agg(label, core, split, metric):
        vals = [results[(label, core, s)][split].get(metric)
                for s in seeds if (label, core, s) in results
                and results[(label, core, s)].get(split)]
        vals = [v for v in vals if v is not None]
        return (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals)) if vals else (None, None, 0)

    print("\n" + "=" * 78)
    print("PHASE 1 SUMMARY (val Top-1, and S33 night zero-shot Top-1; mean±std over seeds)")
    summary = {"config_hash": cfg_hash, "parity": parity, "combos": {}}
    for combo in combos:
        label = "+".join(combo)
        summary["combos"][label] = {}
        row = f"\n  {label}"
        print(row)
        for core in cores:
            vm, vs, n = agg(label, core, "best_val", "top1")
            sm, ss, _ = agg(label, core, "test_s33", "top1")
            summary["combos"][label][core] = {"val_top1_mean": vm, "val_top1_std": vs,
                                              "s33_top1_mean": sm, "s33_top1_std": ss, "n": n}
            if vm is not None:
                s33 = f"{sm:.4f}±{ss:.4f}" if sm is not None else "n/a"
                print(f"    {core:11s}  val {vm:.4f}±{vs:.4f}   S33 {s33}   (n={n})")
        if all((label, "ssm", s) in results and (label, "transformer", s) in results for s in seeds) and seeds:
            g = statistics.mean([results[(label, "transformer", s)]["best_val"]["top1"]
                                 - results[(label, "ssm", s)]["best_val"]["top1"] for s in seeds])
            summary["combos"][label]["mean_gap_xf_minus_ssm"] = g
            print(f"    -> mean gap (xf-ssm) = {g*100:+.2f}%")
    print("=" * 78)
    out = results_dir / (f"phase1_summary{'_smoke' if smoke else ''}_{cfg_hash}.json")
    save_json(summary, out)
    print(f"summary -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the Gate-0 spike: train the SSM and the matched Transformer under
identical encoders/data on Scenario 31, then apply the Gate-0 decision.

Single seed:
    .venv/bin/python -m experiments.phase0.run_gate0
Multi-seed confirmation (recommended before the paper):
    .venv/bin/python -m experiments.phase0.run_gate0 --seeds 1337 2024 7
Quick smoke:
    .venv/bin/python -m experiments.phase0.run_gate0 --seeds 1337 2024 --epochs 1

Writes an append-only experiment log (one row per core per seed), per-run
summaries, and an aggregate multi-seed summary with mean +/- std. Never touches
scenarios 33/34.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.config import load_data_config
from models.beam_model import BeamModel
from models.param_utils import core_parity
from utils.device import get_device
from utils.seed import seed_everything
from utils.yaml_io import load_yaml, save_json

from experiments.phase0.train import train_core


def _base_hash(exp: dict) -> str:
    """Config hash that is STABLE across seeds (groups a multi-seed sweep)."""
    e = {k: v for k, v in exp.items() if k != "seed"}
    return hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()[:12]


def gate0_decision(gap: float, g: dict) -> tuple[str, str]:
    green, amber = float(g["green_max_gap"]), float(g["amber_max_gap"])
    if gap <= green:
        return "GREEN", "SSM parity — proceed to Phase 1 as planned."
    if gap <= amber:
        return "AMBER", "Switch backbone to hybrid (Mamba + periodic light attention); re-spike."
    return "RED", "SSM story dies for this task — escalate to Pritam; invoke pivot options."


def append_log(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.is_file()
    fields = ["date", "config_hash", "seed", "core", "val_top1", "val_top5", "val_dba",
              "test_top1", "test_top5", "test_dba", "core_params", "total_params",
              "train_time_s", "epochs"]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _agg(vals: list[float]) -> dict:
    return {"mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "values": [round(v, 4) for v in vals]}


def train_seed(cfg, exp, cores, device, base_hash, today, results_dir, seed) -> tuple[dict, list]:
    """Train all cores for one seed. Returns ({core: result}, log_rows)."""
    exp = {**exp, "seed": seed}
    seed_everything(seed)
    results, log_rows = {}, []
    for core in cores:
        print(f"\n--- seed {seed} | core: {core} ---")
        res = train_core(cfg, exp, core, device=device, verbose=True,
                         save_dir=results_dir, tag=f"{base_hash}_s{seed}_{core}")
        results[core] = res
        bv, te = res["best_val"], res["test"]
        print(f"  => best val top1={bv['top1']:.4f} (ep {bv.get('epoch')}) | "
              f"test top1={te['top1']:.4f} top5={te['top5']:.4f} dba={te['dba']:.4f} | "
              f"core_params={res['core_params']:,} | {res['train_time_s']:.0f}s")
        log_rows.append({
            "date": today, "config_hash": base_hash, "seed": seed, "core": core,
            "val_top1": round(bv["top1"], 4), "val_top5": round(bv["top5"], 4),
            "val_dba": round(bv["dba"], 4), "test_top1": round(te["top1"], 4),
            "test_top5": round(te["top5"], 4), "test_dba": round(te["dba"], 4),
            "core_params": res["core_params"], "total_params": res["total_params"],
            "train_time_s": round(res["train_time_s"], 1), "epochs": exp["train"]["epochs"],
        })
    return results, log_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/phase0.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (smoke test)")
    ap.add_argument("--cores", nargs="+", default=["ssm", "transformer"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="seeds to average over (default: the single config seed)")
    args = ap.parse_args()

    exp = load_yaml(args.config)
    if args.epochs is not None:
        exp["train"]["epochs"] = args.epochs
    seeds = args.seeds if args.seeds else [int(exp["seed"])]
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    num_beams = int(cfg.beam["num_beams"])
    base_hash = _base_hash(exp)
    today = dt.date.today().isoformat()
    results_dir = Path(exp["results_dir"])

    print(f"=== Gate 0 spike | cfg {base_hash} | device {device} | "
          f"scenario {exp['scenario']} | modalities {exp['modalities']} | "
          f"seeds {seeds} | epochs {exp['train']['epochs']} ===")

    # --- parameter-parity check (seed-independent) ---
    ssm_probe = BeamModel(exp["modalities"], num_beams, "ssm", exp["model"])
    xf_probe = BeamModel(exp["modalities"], num_beams, "transformer", exp["model"])
    parity = core_parity(ssm_probe.core_module(), xf_probe.core_module(),
                         tol=float(exp["model"]["param_parity_tol"]))
    print(f"[parity] ssm_core={parity['ssm_core_params']:,}  "
          f"transformer_core={parity['transformer_core_params']:,}  "
          f"rel_diff={parity['relative_diff']*100:.1f}%  within_tol={parity['within_tol']}")
    if not parity["within_tol"]:
        print("  WARNING: core params outside +/-tol — tune transformer.dim_feedforward.")
    del ssm_probe, xf_probe

    # --- train across seeds ---
    per_seed = {}       # seed -> {core: result}
    all_log_rows = []
    for seed in seeds:
        res, rows = train_seed(cfg, exp, args.cores, device, base_hash, today, results_dir, seed)
        per_seed[seed] = res
        all_log_rows.extend(rows)
    append_log(results_dir / "experiment_log.csv", all_log_rows)

    # --- aggregate across seeds ---
    def collect(core, split, metric):
        return [per_seed[s][core][split][metric] for s in seeds]

    summary = {"date": today, "config_hash": base_hash, "seeds": seeds,
               "epochs": exp["train"]["epochs"], "parity": parity, "per_core": {}}
    for core in args.cores:
        summary["per_core"][core] = {
            "val_top1": _agg(collect(core, "best_val", "top1")),
            "val_top5": _agg(collect(core, "best_val", "top5")),
            "val_dba": _agg(collect(core, "best_val", "dba")),
            "test_top1": _agg(collect(core, "test", "top1")),
            "test_top5": _agg(collect(core, "test", "top5")),
            "test_dba": _agg(collect(core, "test", "dba")),
            "core_params": per_seed[seeds[0]][core]["core_params"],
        }

    # --- Gate 0 on the aggregate (mean val Top-1 gap), plus per-seed gaps ---
    print("\n" + "=" * 72)
    if "ssm" in args.cores and "transformer" in args.cores:
        ssm_v = collect("ssm", "best_val", "top1")
        xf_v = collect("transformer", "best_val", "top1")
        per_seed_gap = [round(x - s, 4) for s, x in zip(ssm_v, xf_v)]
        mean_gap = statistics.mean(xf_v) - statistics.mean(ssm_v)
        verdict, action = gate0_decision(mean_gap, exp["gate0"])
        sd = summary["per_core"]["ssm"]["val_top1"]
        xd = summary["per_core"]["transformer"]["val_top1"]
        st = summary["per_core"]["ssm"]["test_top1"]
        xt = summary["per_core"]["transformer"]["test_top1"]
        print(f"GATE 0  ({len(seeds)} seed{'s' if len(seeds) > 1 else ''}: {seeds})")
        print(f"  SSM         val_top1 {sd['mean']:.4f} ± {sd['std']:.4f}   "
              f"test_top1 {st['mean']:.4f} ± {st['std']:.4f}")
        print(f"  Transformer val_top1 {xd['mean']:.4f} ± {xd['std']:.4f}   "
              f"test_top1 {xt['mean']:.4f} ± {xt['std']:.4f}")
        print(f"  per-seed gap (xf-ssm): {per_seed_gap}")
        print(f"  mean gap = {mean_gap*100:+.2f}%   ->  VERDICT: {verdict}")
        print(f"  {action}")
        summary["gate0"] = {"mean_gap": mean_gap, "per_seed_gap": per_seed_gap,
                            "verdict": verdict, "action": action,
                            "ssm_val_top1": _agg(ssm_v), "transformer_val_top1": _agg(xf_v)}
    print("=" * 72)

    out = results_dir / (f"gate0_multiseed_{base_hash}.json" if len(seeds) > 1
                         else f"gate0_{base_hash}.json")
    save_json(summary, out)
    print(f"\nResults saved:\n  experiment log : {results_dir/'experiment_log.csv'}"
          f"\n  summary        : {out}"
          f"\n  per-run        : history_{base_hash}_s<seed>_<core>.csv, "
          f"ckpt_{base_hash}_s<seed>_<core>.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

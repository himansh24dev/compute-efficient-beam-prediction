#!/usr/bin/env python3
"""One run -> every result: fair accuracy+efficiency comparison vs BeMamba
(Part A) and the state-split (P2) analysis (Part B).

    .venv/bin/python -m experiments.bemamba_compare.run_compare
    .venv/bin/python -m experiments.bemamba_compare.run_compare --smoke

Part A matrix: {ssm, transformer} x {2M,3M,4M} x {window-random, episode-random} x seeds.
Part B: on the trained 4M SSM -> analytical bytes-to-resume + payload-quantization Pareto.
Resumable, per-run robust. Scenario 34 stays held out (BeMamba uses 31-34; we use 31-33).
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

import torch

from data.config import load_data_config
from models.beam_model import BeamModel
from utils.device import get_device
from utils.yaml_io import load_yaml, save_json

from .train_compare import train_one
from .split_inference import analytical_payloads, feature_quant_accuracy
from .splits_compare import build_compare_loaders
from experiments.phase0.train import _move
from utils.device import channels_last

# BeMamba (IEEE TWC 2026) reported numbers, for side-by-side reference.
BEMAMBA_REF = {
    "venue": "IEEE TWC 2026", "protocol": "random 80/20 over S31-34 (window-level)",
    "top3_overall_4M": 0.8828, "dba_4M": 0.9107,
    "top3_per_scenario_4M": {"S31": 1.000, "S32": 0.8811, "S33": 0.8494, "S34": 0.8564},
    "params_4M_M": 16.19, "gflops_4M": 23.4, "fps_4M_rtx3090": 28.83,
    "note": "snapshot (5 frames), random split incl. S34, 13.3M-param FC head",
}

FIELDS = ["date", "cfg_hash", "core", "combo", "protocol", "seed",
          "top1", "top2", "top3", "top5", "dba",
          "s31_top3", "s32_top3", "s33_top3", "s34_top3",   # per-scenario Top-3 (vs BeMamba table)
          "s31_dba", "s32_dba", "s33_dba", "s34_dba",
          "params", "gflops", "fps", "latency_ms", "peak_mem_mb", "n_test", "epochs", "status"]
COMBOS = {"2M": ["gps", "camera"], "3M": ["gps", "camera", "lidar"],
          "4M": ["gps", "camera", "lidar", "radar"]}


def _hash(exp):
    keep = {k: exp[k] for k in ("model", "train", "window", "split", "compare_scenarios") if k in exp}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def _done(csv_path, h):
    s = set()
    if csv_path.is_file():
        for r in csv.DictReader(open(csv_path)):
            if r.get("cfg_hash") == h and r.get("status") == "ok":
                s.add((r["core"], r["combo"], r["protocol"], int(r["seed"])))
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
    ap.add_argument("--config", default="configs/experiments/bemamba_compare.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    exp = load_yaml(args.config)
    smoke = args.smoke
    if smoke:
        exp["train"]["epochs"] = 1
        exp["sweep"] = {"cores": ["ssm"], "combos": ["2M"],
                        "protocols": ["episode-random"], "seeds": [1337]}

    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    num_beams = int(cfg.beam["num_beams"])
    h = _hash(exp)
    today = dt.date.today().isoformat()
    results_dir = Path(exp["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)
    log_csv = results_dir / ("compare_log_smoke.csv" if smoke else "compare_log.csv")

    cores = list(exp["sweep"]["cores"]); combos = list(exp["sweep"]["combos"])
    protocols = list(exp["sweep"]["protocols"]); seeds = list(exp["sweep"]["seeds"])
    total = len(cores) * len(combos) * len(protocols) * len(seeds)
    print(f"=== BeMamba comparison | cfg {h} | device {device} | {total} runs (Part A) ===")
    print(f"  cores={cores} combos={combos} protocols={protocols} seeds={seeds}")

    done = _done(log_csv, h)
    results = {}
    ckpt_4M_ssm = None
    i = 0
    for core in cores:
        for cname in combos:
            combo = COMBOS[cname]
            for protocol in protocols:
                for seed in seeds:
                    i += 1
                    key = (core, cname, protocol, seed)
                    if key in done:
                        print(f"[{i}/{total}] SKIP {key}"); continue
                    print(f"\n[{i}/{total}] {core} | {cname} | {protocol} | seed {seed}")
                    try:
                        r = train_one(cfg, exp, core, combo, protocol, seed, device=device)
                        results[key] = r; t = r["test"]; e = r["efficiency"]
                        if core == "ssm" and cname == "4M" and protocol == "episode-random":
                            ckpt_4M_ssm = (r["best_state"], combo, seed)
                        ps = t.get("per_scenario", {})
                        def pv(sid, key):
                            return round(ps[sid][key], 4) if sid in ps else ""
                        psline = " ".join(f"S{sid}={ps[sid]['top3']:.3f}" for sid in sorted(ps))
                        print(f"  => top1={t['top1']:.4f} top3={t['top3']:.4f} dba={t['dba']:.4f} "
                              f"| per-scenario top3: {psline} | {r['params']:,} params | {e['fps']:.1f} FPS")
                        _append(log_csv, {
                            "date": today, "cfg_hash": h, "core": core, "combo": cname,
                            "protocol": protocol, "seed": seed,
                            "top1": round(t["top1"], 4), "top2": round(t["top2"], 4),
                            "top3": round(t["top3"], 4), "top5": round(t["top5"], 4),
                            "dba": round(t["dba"], 4),
                            "s31_top3": pv(31, "top3"), "s32_top3": pv(32, "top3"),
                            "s33_top3": pv(33, "top3"), "s34_top3": pv(34, "top3"),
                            "s31_dba": pv(31, "dba"), "s32_dba": pv(32, "dba"),
                            "s33_dba": pv(33, "dba"), "s34_dba": pv(34, "dba"),
                            "params": r["params"],
                            "gflops": round(e["gflops"], 3) if e["gflops"] == e["gflops"] else "",
                            "fps": round(e["fps"], 1), "latency_ms": round(e["latency_ms_per_sample"], 3),
                            "peak_mem_mb": round(e["peak_mem_mb"], 1) if e["peak_mem_mb"] == e["peak_mem_mb"] else "",
                            "n_test": r["split_info"]["n_test"], "epochs": exp["train"]["epochs"],
                            "status": "ok"})
                    except Exception as ex:
                        traceback.print_exc()
                        _append(log_csv, {"date": today, "cfg_hash": h, "core": core, "combo": cname,
                                          "protocol": protocol, "seed": seed,
                                          "epochs": exp["train"]["epochs"], "status": f"FAIL:{type(ex).__name__}"})
                        print(f"  !! FAILED: {ex}")

    # -------- Part B: state-split (P2) analysis on the 4M SSM --------
    print("\n" + "=" * 72)
    print("PART B: state-split (P2) analysis")
    payloads = analytical_payloads(cfg, exp["model"], COMBOS["4M"])
    print(f"  per-frame raw sensors : {payloads['per_frame_raw_bytes']/1024:.1f} KB")
    print(f"  fused feature / step  : {payloads['feature_step_bytes']} B")
    print(f"  SSM state (constant)  : {payloads['ssm_state_bytes_const']/1024:.1f} KB")
    print(f"  crossover (state<featurehistory) at L = {payloads['crossover_length_state_vs_features']} frames")
    quant = None
    if ckpt_4M_ssm is not None and not smoke:
        state, combo, seed = ckpt_4M_ssm
        model = BeamModel(combo, num_beams, "ssm", exp["model"]).to(device)
        model.load_state_dict(state)
        if channels_last(cfg):
            model = model.to(memory_format=torch.channels_last)
        _, _, te, _ = build_compare_loaders(cfg, exp, combo, "episode-random", seed)
        quant = feature_quant_accuracy(model, te, cfg, device)
        print("  payload-quantization Pareto (fused feature):")
        for bits, m in quant.items():
            print(f"    {bits:>6}: {m['feature_bytes_per_step']:6.0f} B/step  "
                  f"top1={m['top1']:.4f} top3={m['top3']:.4f} dba={m['dba']:.4f}")
    print("=" * 72)

    _summarize(results, exp, h, results_dir, payloads, quant, smoke)
    print(f"\nlog -> {log_csv}")
    return 0


def _summarize(results, exp, h, results_dir, payloads, quant, smoke):
    summary = {"cfg_hash": h, "bemamba_reference": BEMAMBA_REF,
               "part_b_split_inference": {"analytical_payloads": payloads,
                                          "feature_quant_pareto": quant},
               "part_a": {}}
    print("\nPART A summary (mean over seeds) vs BeMamba 4M [Top-3 0.8828 / DBA 0.9107]:")
    for core in exp["sweep"]["cores"]:
        for cname in exp["sweep"]["combos"]:
            for protocol in exp["sweep"]["protocols"]:
                rs = [v for k, v in results.items() if k[0] == core and k[1] == cname and k[2] == protocol]
                if not rs:
                    continue
                t1 = statistics.mean(r["test"]["top1"] for r in rs)
                t3 = statistics.mean(r["test"]["top3"] for r in rs)
                db = statistics.mean(r["test"]["dba"] for r in rs)
                p = rs[0]["params"]
                summary["part_a"][f"{core}|{cname}|{protocol}"] = {
                    "top1": t1, "top3": t3, "dba": db, "params": p, "n": len(rs)}
                # per-scenario mean top3 across seeds
                ps_top3 = {}
                for sid in (31, 32, 33, 34):
                    vals = [r["test"]["per_scenario"][sid]["top3"] for r in rs
                            if sid in r["test"].get("per_scenario", {})]
                    if vals:
                        ps_top3[sid] = statistics.mean(vals)
                summary["part_a"][f"{core}|{cname}|{protocol}"]["per_scenario_top3"] = ps_top3
                psline = " ".join(f"S{sid}={v:.3f}" for sid, v in ps_top3.items())
                print(f"  {core:11s} {cname:3s} {protocol:15s}: "
                      f"top1={t1:.4f} top3={t3:.4f} dba={db:.4f} ({p:,} params) | S: {psline}")
    print("\n  BeMamba 4M (TWC 2026, window-random incl. S34): "
          f"S31=1.000 S32=0.8811 S33=0.8494 S34=0.8564 | overall top3=0.8828 dba=0.9107 | 16.19M params")
    out = results_dir / f"compare_summary{'_smoke' if smoke else ''}_{h}.json"
    save_json(summary, out)
    print(f"summary -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())

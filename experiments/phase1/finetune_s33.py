#!/usr/bin/env python3
"""S33 fine-tune: the single documented day->night adaptation run.

For each (combo, core, seed): load the day-trained Phase-1 checkpoint, measure
ZERO-SHOT accuracy on S33-test, fine-tune on S33-train (early-stop on S33-val),
and re-measure on the SAME S33-test. Reports zero-shot vs fine-tuned.

    .venv/bin/python -m experiments.phase1.finetune_s33
    .venv/bin/python -m experiments.phase1.finetune_s33 --smoke      # 1-epoch check

Leakage-free (S33 split by episode). GPS uses the frozen DAY stats throughout.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import math
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.config import load_data_config
from data.dataset import make_snapshot_dataset
from data.preprocess.stats import load_gps_stats
from data.splits import single_scenario_split
from models.beam_model import BeamModel
from utils.device import autocast_context, channels_last, dataloader_kwargs, get_device
from utils.seed import seed_everything
from utils.yaml_io import load_yaml, save_json

from experiments.phase0.train import _apply_modality_subset, _move, _progress, evaluate
from experiments.phase1.run_phase1 import _base_hash

LOG_FIELDS = ["date", "day_hash", "combo", "core", "seed", "split_seed",
              "zshot_top1", "zshot_top5", "zshot_dba",
              "ft_top1", "ft_top5", "ft_dba", "ft_val_top1",
              "delta_top1", "epochs", "status"]


def build_s33_loaders(cfg, ft_exp, model_cfg, modalities, splits):
    _apply_modality_subset(cfg, modalities)
    cfg.snapshot["future_horizons"] = list(model_cfg.get("future_horizons", []))
    gps_stats = load_gps_stats(cfg) if "gps" in modalities else None   # DAY stats, no refit
    ds_tr = make_snapshot_dataset(cfg, splits.train, gps_stats)
    ds_va = make_snapshot_dataset(cfg, splits.val, gps_stats)
    ds_te = make_snapshot_dataset(cfg, splits.test, gps_stats)
    cfg.loader["num_workers"] = int(ft_exp["train"].get("num_workers", 12))
    cfg.loader["batch_size"] = int(ft_exp["train"]["batch_size"])
    cfg.loader["prefetch_factor"] = int(ft_exp["train"].get("prefetch_factor", 3))
    tr_kw, ev_kw = dataloader_kwargs(cfg, train=True), dataloader_kwargs(cfg, train=False)
    return DataLoader(ds_tr, **tr_kw), DataLoader(ds_va, **ev_kw), DataLoader(ds_te, **ev_kw)


def finetune_core(cfg, ft_exp, model_cfg, core, modalities, day_ckpt, splits,
                  device, seed, save_dir, tag, verbose=True):
    seed_everything(seed)
    num_beams = int(cfg.beam["num_beams"])
    tr, va, te = build_s33_loaders(cfg, ft_exp, model_cfg, modalities, splits)

    model = BeamModel(modalities, num_beams, core, model_cfg).to(device)
    if channels_last(cfg):
        model = model.to(memory_format=torch.channels_last)
    sd = torch.load(day_ckpt, map_location=device)
    model.load_state_dict(sd["model"])

    # ZERO-SHOT (day model, no adaptation) on S33-test.
    zshot = evaluate(model, te, device, cfg, num_beams)
    if verbose:
        print(f"  [{core}] zero-shot S33-test Top-1 = {zshot['top1']:.4f}")

    # Fine-tune.
    tcfg = ft_exp["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))
    epochs = int(tcfg["epochs"]); warmup = int(tcfg.get("warmup_epochs", 0))
    steps_per_epoch = max(1, len(tr)); base_lr = float(tcfg["lr"])

    def lr_at(step):
        e = step / steps_per_epoch
        if e < warmup:
            return e / max(warmup, 1e-9)
        prog = (e - warmup) / max(epochs - warmup, 1e-9)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and cfg.runtime.get("amp", True)))
    crit = nn.CrossEntropyLoss(label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
    chlast = channels_last(cfg); grad_clip = float(tcfg.get("grad_clip", 0.0))

    best = {"top1": -1.0}; best_state = None; gstep = 0
    for epoch in range(epochs):
        model.train()
        run_loss = 0.0; ep_t0 = time.time(); nb = len(tr)
        for i, batch in enumerate(tr, 1):
            for g in opt.param_groups:
                g["lr"] = base_lr * lr_at(gstep)
            inputs = _move(batch["inputs"], device, chlast)
            target = batch["beam"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_context(cfg, device):
                logits = model(inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = crit(logits.float(), target)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
            run_loss += loss.item(); gstep += 1
            if verbose and (i % 5 == 0 or i == nb):
                _progress(core, epoch, epochs, i, nb, run_loss / i, base_lr * lr_at(gstep), ep_t0)
        val = evaluate(model, va, device, cfg, num_beams)
        if val["top1"] > best["top1"]:
            best = {**val, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            if sys.stdout.isatty():
                sys.stdout.write("\r" + " " * 110 + "\r")
            print(f"  [{core}] ft ep {epoch+1:02d}/{epochs} loss={run_loss/steps_per_epoch:.3f} "
                  f"S33-val_top1={val['top1']:.4f}{' *best' if val['top1']==best['top1'] else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)
    ft_test = evaluate(model, te, device, cfg, num_beams)
    if save_dir is not None and best_state is not None:
        torch.save({"model": best_state, "core": core, "modalities": modalities,
                    "seed": seed, "zero_shot_test": zshot, "ft_test": ft_test},
                   Path(save_dir) / f"ckpt_ft_{tag}.pt")
    return {"core": core, "modalities": modalities, "seed": seed,
            "zero_shot_test": zshot, "ft_test": ft_test, "ft_best_val": best,
            "splits": splits.summary()}


def _find_day_ckpt(day_dir: Path, day_hash: str, label: str, core: str, seed: int) -> Path:
    p = day_dir / f"ckpt_{day_hash}_{label}_{core}_s{seed}.pt"
    if p.is_file():
        return p
    matches = list(day_dir.glob(f"ckpt_*_{label}_{core}_s{seed}.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"no day checkpoint for {label}/{core}/seed{seed} in {day_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/finetune_s33.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ft_exp = load_yaml(args.config)
    day_exp = load_yaml(ft_exp["day_config"])
    model_cfg = day_exp["model"]                       # architecture MUST match the day ckpt
    day_hash = _base_hash(day_exp)
    cfg = load_data_config(day_exp["data_config"])
    device = get_device(cfg)
    scenario = int(ft_exp["finetune_scenario"])
    day_dir = Path(ft_exp["day_results_dir"])
    results_dir = Path(ft_exp["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)

    combos = [list(c) for c in ft_exp["sweep"]["modality_combos"]]
    cores = list(ft_exp["sweep"]["cores"])
    seeds = list(ft_exp["sweep"]["seeds"])
    if args.smoke:
        ft_exp["train"]["epochs"] = 1
        combos, cores, seeds = [["gps", "camera"]], ["ssm"], [1337]

    # Frozen leakage-free S33 episode split.
    fr = tuple(ft_exp["split"]["fracs"]); ssd = int(ft_exp["split"]["split_seed"])
    splits = single_scenario_split(cfg, scenario, fracs=fr, seed=ssd)
    print(f"=== S33 fine-tune | day_hash {day_hash} | device {device} ===")
    print(f"  S33 split (episodes): {splits.summary()}")

    log_csv = results_dir / ("ft_log_smoke.csv" if args.smoke else "finetune_s33_log.csv")
    today = dt.date.today().isoformat()
    results = []
    for combo in combos:
        label = "+".join(combo)
        for core in cores:
            for seed in seeds:
                tag = f"{day_hash}_{label}_{core}_s{seed}"
                print(f"\n--- fine-tune {label} | {core} | day-seed {seed} ---")
                try:
                    day_ckpt = _find_day_ckpt(day_dir, day_hash, label, core, seed)
                    res = finetune_core(cfg, ft_exp, model_cfg, core, combo, day_ckpt, splits,
                                        device, seed, results_dir, tag, verbose=True)
                    z, f = res["zero_shot_test"], res["ft_test"]
                    d = f["top1"] - z["top1"]
                    print(f"  => ZERO-SHOT {z['top1']:.4f} -> FINE-TUNED {f['top1']:.4f}  "
                          f"(Δ {d*100:+.1f} pts) | S33-test DBA {f['dba']:.4f}")
                    results.append(res)
                    _log(log_csv, {"date": today, "day_hash": day_hash, "combo": label, "core": core,
                                   "seed": seed, "split_seed": ssd,
                                   "zshot_top1": round(z["top1"], 4), "zshot_top5": round(z["top5"], 4),
                                   "zshot_dba": round(z["dba"], 4), "ft_top1": round(f["top1"], 4),
                                   "ft_top5": round(f["top5"], 4), "ft_dba": round(f["dba"], 4),
                                   "ft_val_top1": round(res["ft_best_val"]["top1"], 4),
                                   "delta_top1": round(d, 4), "epochs": ft_exp["train"]["epochs"],
                                   "status": "ok"})
                except Exception as e:
                    traceback.print_exc()
                    _log(log_csv, {"date": today, "day_hash": day_hash, "combo": label, "core": core,
                                   "seed": seed, "split_seed": ssd, "status": f"FAIL:{type(e).__name__}"})
                    print(f"  !! FAILED: {e}")

    _summarize(results, results_dir, day_hash, args.smoke)
    print(f"\nlog -> {log_csv}")
    return 0


def _log(csv_path, row):
    exists = csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def _summarize(results, results_dir, day_hash, smoke):
    if not results:
        return
    import statistics
    print("\n" + "=" * 70)
    print("S33 FINE-TUNE: zero-shot -> fine-tuned (S33-test Top-1, mean over seeds)")
    summary = {"day_hash": day_hash, "by_core": {}}
    cores = sorted({r["core"] for r in results})
    for core in cores:
        rs = [r for r in results if r["core"] == core]
        z = [r["zero_shot_test"]["top1"] for r in rs]
        f = [r["ft_test"]["top1"] for r in rs]
        zm, fm = statistics.mean(z), statistics.mean(f)
        summary["by_core"][core] = {"zero_shot_mean": zm, "ft_mean": fm,
                                    "ft_std": statistics.pstdev(f) if len(f) > 1 else 0.0,
                                    "delta": fm - zm, "n": len(rs)}
        print(f"  {core:11s}: {zm:.4f} -> {fm:.4f}   (Δ {(fm-zm)*100:+.1f} pts, n={len(rs)})")
    print("=" * 70)
    save_json(summary, results_dir / f"finetune_s33_summary{'_smoke' if smoke else ''}_{day_hash}.json")


if __name__ == "__main__":
    raise SystemExit(main())

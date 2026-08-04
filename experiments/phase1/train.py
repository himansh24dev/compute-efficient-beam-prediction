"""Phase-1 trainer: full-multimodal snapshot current-beam task on the PROPER
frozen split (train/val = S31+S32 day; test = S33 night, zero-shot / cross-condition).

Any modality subset is selectable (for the modality-ablation sweep). Encoders/
fusion/head are identical across the SSM and Transformer cores; only the core
differs (params matched, checked by the sweep runner). Reuses the Phase-0
train/eval helpers so the two phases are directly comparable.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.config import DataConfig
from data.dataset import make_snapshot_dataset
from data.preprocess.stats import fit_gps_stats
from data.splits import Splits, resolve_splits, single_scenario_split
from models.beam_model import BeamModel
from models.param_utils import count_parameters
from utils.device import autocast_context, channels_last, dataloader_kwargs, get_device
from utils.seed import seed_everything

from experiments.phase0.train import _apply_modality_subset, _move, _progress, evaluate


def get_splits(cfg: DataConfig, smoke: bool = False) -> Splits:
    """Real: frozen v1 (train/val S31+S32, test S33). Smoke: S31-only (uses the
    already-built S31 cache; never needs S32/S33)."""
    if smoke:
        return single_scenario_split(cfg, 31, fracs=(0.70, 0.15, 0.15), seed=1337)
    return resolve_splits(cfg)


def build_loaders(cfg: DataConfig, exp: dict, modalities: list[str], splits: Splits):
    _apply_modality_subset(cfg, modalities)
    cfg.snapshot["future_horizons"] = list(exp["model"].get("future_horizons", []))
    gps_stats = fit_gps_stats(cfg, splits.train, refit=True) if "gps" in modalities else None

    ds_train = make_snapshot_dataset(cfg, splits.train, gps_stats)
    ds_val = make_snapshot_dataset(cfg, splits.val, gps_stats)
    ds_test = make_snapshot_dataset(cfg, splits.test, gps_stats) if splits.test else None

    cfg.loader["num_workers"] = int(exp["train"].get("num_workers", cfg.loader.get("num_workers", 8)))
    cfg.loader["batch_size"] = int(exp["train"]["batch_size"])
    cfg.loader["prefetch_factor"] = int(exp["train"].get("prefetch_factor", cfg.loader.get("prefetch_factor", 4)))
    tr_kw, ev_kw = dataloader_kwargs(cfg, train=True), dataloader_kwargs(cfg, train=False)
    return (
        DataLoader(ds_train, **tr_kw),
        DataLoader(ds_val, **ev_kw),
        DataLoader(ds_test, **ev_kw) if ds_test is not None else None,
    )


def train_core(cfg: DataConfig, exp: dict, core_kind: str, modalities: list[str],
               splits: Splits, device=None, seed: int | None = None, verbose: bool = True,
               save_dir: Path | None = None, tag: str | None = None) -> dict:
    seed = int(seed if seed is not None else exp["seed"])
    seed_everything(seed)
    device = device or get_device(cfg)
    num_beams = int(cfg.beam["num_beams"])
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"{core_kind}"
    ckpt_path = (save_dir / f"ckpt_{tag}.pt") if save_dir else None
    history_path = (save_dir / f"history_{tag}.csv") if save_dir else None

    tr, va, te = build_loaders(cfg, exp, modalities, splits)
    model = BeamModel(modalities, num_beams, core_kind, exp["model"]).to(device)
    if channels_last(cfg):
        model = model.to(memory_format=torch.channels_last)

    tcfg = exp["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))
    epochs = int(tcfg["epochs"])
    warmup = int(tcfg.get("warmup_epochs", 0))
    steps_per_epoch = max(1, len(tr))
    base_lr = float(tcfg["lr"])

    def lr_at(step):
        e = step / steps_per_epoch
        if e < warmup:
            return e / max(warmup, 1e-9)
        prog = (e - warmup) / max(epochs - warmup, 1e-9)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and cfg.runtime.get("amp", True)))
    crit = nn.CrossEntropyLoss(label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
    chlast = channels_last(cfg)
    grad_clip = float(tcfg.get("grad_clip", 0.0))

    best = {"top1": -1.0}
    history = []
    gstep = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        ep_t0 = time.time()
        n_batches = len(tr)
        for i, batch in enumerate(tr, 1):
            lr_now = base_lr * lr_at(gstep)
            for g in opt.param_groups:
                g["lr"] = lr_now
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
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            gstep += 1
            if verbose and (i % 5 == 0 or i == n_batches):
                _progress(core_kind, epoch, epochs, i, n_batches, run_loss / i, lr_now, ep_t0)
        val = evaluate(model, va, device, cfg, num_beams)
        improved = val["top1"] > best["top1"]
        if improved:
            best = {**val, "epoch": epoch}
            if ckpt_path is not None:
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": val,
                            "core_kind": core_kind, "modalities": modalities, "seed": seed}, ckpt_path)
        history.append({"epoch": epoch + 1, "train_loss": run_loss / steps_per_epoch,
                        "val_top1": val["top1"], "val_top5": val["top5"], "val_dba": val["dba"]})
        if verbose:
            if sys.stdout.isatty():
                sys.stdout.write("\r" + " " * 110 + "\r")
            print(f"  [{core_kind:11s}|{'+'.join(modalities):22s}] ep {epoch+1:02d}/{epochs} "
                  f"loss={run_loss/steps_per_epoch:.3f} val_top1={val['top1']:.4f} "
                  f"top5={val['top5']:.4f} dba={val['dba']:.4f}{' *best' if improved else ''}")
    train_time = time.time() - t0

    if history_path is not None:
        with open(history_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader(); w.writerows(history)

    # Cross-condition zero-shot test (S33 night) if available.
    test = evaluate(model, te, device, cfg, num_beams) if te is not None else {}
    return {
        "core": core_kind, "modalities": modalities, "seed": seed,
        "best_val": best, "test_s33": test,
        "core_params": count_parameters(model.core_module()),
        "total_params": count_parameters(model),
        "train_time_s": train_time, "history": history,
        "splits": splits.summary(),
    }

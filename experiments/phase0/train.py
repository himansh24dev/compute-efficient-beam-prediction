"""Phase-0 trainer: train ONE temporal core (ssm|transformer) on the Scenario-31
snapshot current-beam task and return val/test metrics.

Reused by run_gate0.py to train both cores under identical data/seed/encoders.
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
from data.index import build_sample_index
from data.preprocess.stats import fit_gps_stats
from data.splits import single_scenario_split
from models.beam_model import BeamModel
from models.param_utils import core_parity, count_parameters
from utils.device import autocast_context, channels_last, dataloader_kwargs, get_device
from utils.seed import seed_everything

from .metrics import dba_correct, topk_correct


def _apply_modality_subset(cfg: DataConfig, modalities: list[str]) -> None:
    """Enable only the Phase-0 modalities (camera+gps); disable the rest.

    Reuses the existing v1 cache (camera+gps tensors already built for S31).
    """
    for m in ("camera", "lidar", "radar", "gps"):
        cfg.modalities[m]["enabled"] = (m in modalities)


def build_loaders(cfg: DataConfig, exp: dict):
    modalities = list(exp["modalities"])
    _apply_modality_subset(cfg, modalities)
    # Current-beam-only spike: no future-horizon lookahead (maximizes usable
    # windows; the model has no future heads here).
    cfg.snapshot["future_horizons"] = list(exp["model"].get("future_horizons", []))
    scenario = int(exp["scenario"])
    fracs = tuple(exp["split"]["fracs"])
    seed = int(exp["seed"])

    splits = single_scenario_split(cfg, scenario, fracs=fracs, seed=seed)
    gps_stats = fit_gps_stats(cfg, splits.train, refit=True) if "gps" in modalities else None

    ds_train = make_snapshot_dataset(cfg, splits.train, gps_stats)
    ds_val = make_snapshot_dataset(cfg, splits.val, gps_stats)
    ds_test = make_snapshot_dataset(cfg, splits.test, gps_stats)

    nw = int(exp["train"].get("num_workers", cfg.loader.get("num_workers", 8)))
    cfg.loader["num_workers"] = nw
    cfg.loader["batch_size"] = int(exp["train"]["batch_size"])
    tr_kw = dataloader_kwargs(cfg, train=True)
    ev_kw = dataloader_kwargs(cfg, train=False)

    return (
        DataLoader(ds_train, **tr_kw),
        DataLoader(ds_val, **ev_kw),
        DataLoader(ds_test, **ev_kw),
        splits, modalities,
    )


def _move(inputs, device, chlast):
    # channels_last is applied inside the camera encoder (on the 4D per-frame
    # tensor); here we just move to device.
    return {m: v.to(device, non_blocking=True) for m, v in inputs.items()}


def _progress(core_kind, epoch, epochs, i, n, loss, lr, t0):
    """Live in-epoch progress.

    On a TTY: a redrawing one-line bar. When piped (e.g. through `tee` into a log,
    which is how the overnight scripts run), `\\r` bars are useless and would spam
    the file -- so we emit throttled newline-terminated lines instead, roughly four
    per epoch. Either way progress is visible while the job runs.
    """
    frac = i / max(n, 1)
    elapsed = time.time() - t0
    ips = i / max(elapsed, 1e-9)
    eta = (n - i) / max(ips, 1e-9)
    if sys.stdout.isatty():
        fill = int(28 * frac)
        bar = "█" * fill + "░" * (28 - fill)
        sys.stdout.write(
            f"\r  [{core_kind:11s}] ep {epoch+1:02d}/{epochs} |{bar}| "
            f"{i:>4}/{n} loss={loss:.3f} lr={lr:.1e} {ips:4.0f} it/s eta {eta:4.0f}s   "
        )
        sys.stdout.flush()
    else:
        every = max(1, n // 4)
        if i % every == 0 or i == n:
            print(f"  [{core_kind:11s}] ep {epoch+1:02d}/{epochs} {i:>5}/{n} "
                  f"({frac*100:3.0f}%) loss={loss:.3f} {ips:4.0f} it/s eta {eta:4.0f}s",
                  flush=True)


@torch.no_grad()
def evaluate(model, loader, device, cfg, num_beams: int) -> dict:
    model.eval()
    ks = (1, 3, 5)
    ts = (1, 2, 3)
    tot = 0
    topk = {k: 0 for k in ks}
    dba = {t: 0 for t in ts}
    chlast = channels_last(cfg)
    for batch in loader:
        inputs = _move(batch["inputs"], device, chlast)
        target = batch["beam"].to(device, non_blocking=True)
        with autocast_context(cfg, device):
            logits = model(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.float()
        for k, v in topk_correct(logits, target, ks).items():
            topk[k] += v
        for t, v in dba_correct(logits, target, ts).items():
            dba[t] += v
        tot += target.numel()
    acc = {f"top{k}": topk[k] / max(tot, 1) for k in ks}
    dba_score = sum(dba[t] / max(tot, 1) for t in ts) / len(ts)
    acc["dba"] = dba_score
    acc["n"] = tot
    return acc


def train_core(cfg: DataConfig, exp: dict, core_kind: str, device=None, verbose: bool = True,
               save_dir: Path | None = None, tag: str | None = None) -> dict:
    seed_everything(int(exp["seed"]))
    device = device or get_device(cfg)
    num_beams = int(cfg.beam["num_beams"])
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or core_kind
    ckpt_path = (save_dir / f"ckpt_{tag}.pt") if save_dir else None
    history_path = (save_dir / f"history_{tag}.csv") if save_dir else None

    tr, va, te, splits, modalities = build_loaders(cfg, exp)
    model = BeamModel(modalities, num_beams, core_kind, exp["model"]).to(device)
    if channels_last(cfg):
        model = model.to(memory_format=torch.channels_last)

    tcfg = exp["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))
    epochs = int(tcfg["epochs"])
    warmup = int(tcfg.get("warmup_epochs", 0))
    steps_per_epoch = max(1, len(tr))

    def lr_at(step):
        e = step / steps_per_epoch
        if e < warmup:
            return e / max(warmup, 1e-9)
        prog = (e - warmup) / max(epochs - warmup, 1e-9)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    base_lr = float(tcfg["lr"])
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
                            "core_kind": core_kind, "exp": exp}, ckpt_path)
        history.append({"epoch": epoch + 1, "train_loss": run_loss / steps_per_epoch,
                        "val_top1": val["top1"], "val_top3": val["top3"],
                        "val_top5": val["top5"], "val_dba": val["dba"]})
        if verbose:
            if sys.stdout.isatty():
                sys.stdout.write("\r" + " " * 100 + "\r")  # clear progress line
            star = " *best" if improved else ""
            print(f"  [{core_kind:11s}] epoch {epoch+1:02d}/{epochs} "
                  f"loss={run_loss/steps_per_epoch:.3f} "
                  f"val_top1={val['top1']:.4f} top5={val['top5']:.4f} dba={val['dba']:.4f}"
                  f"{star}")
    train_time = time.time() - t0

    if history_path is not None:
        with open(history_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    test = evaluate(model, te, device, cfg, num_beams)
    core_params = count_parameters(model.core_module())
    total_params = count_parameters(model)
    return {
        "core": core_kind,
        "best_val": best,
        "test": test,
        "core_params": core_params,
        "total_params": total_params,
        "train_time_s": train_time,
        "splits": splits.summary(),
        "modalities": modalities,
        "history": history,
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "history_path": str(history_path) if history_path else None,
    }

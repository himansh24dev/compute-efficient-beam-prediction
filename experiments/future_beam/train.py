"""Future-beam trainer: one model with a current-beam head + auxiliary t+h heads,
supervised jointly, evaluated per horizon. Reuses the Phase-1 loader builder
(which already wires the dataset's future targets from model.future_horizons).
"""
from __future__ import annotations

import copy
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from data.config import DataConfig
from data.splits import Splits
from models.beam_model import BeamModel
from models.param_utils import count_parameters
from utils.device import autocast_context, channels_last, get_device
from utils.seed import seed_everything

from experiments.phase0.train import _move, _progress
from experiments.phase1.train import build_loaders


@torch.no_grad()
def evaluate_future(model, loader, device, cfg, horizons) -> dict:
    model.eval()
    tot = 0
    cur = 0
    hcorr = [0] * len(horizons)
    chlast = channels_last(cfg)
    for batch in loader:
        inp = _move(batch["inputs"], device, chlast)
        beam = batch["beam"].to(device, non_blocking=True)
        fut = batch["future"].to(device, non_blocking=True)          # (B, H)
        with autocast_context(cfg, device):
            out = model(inp)
        logits, flog = out                                            # (B,64), (B,H,64)
        cur += (logits.float().argmax(-1) == beam).sum().item()
        flog = flog.float()
        for i in range(len(horizons)):
            hcorr[i] += (flog[:, i].argmax(-1) == fut[:, i]).sum().item()
        tot += beam.numel()
    res = {"current": cur / max(tot, 1)}
    for i, h in enumerate(horizons):
        res[f"t+{h}"] = hcorr[i] / max(tot, 1)
    res["n"] = tot
    return res


def train_core_future(cfg: DataConfig, exp: dict, core_kind: str, modalities: list[str],
                      splits: Splits, device=None, seed: int | None = None,
                      verbose: bool = True, save_dir: Path | None = None, tag: str | None = None) -> dict:
    seed = int(seed if seed is not None else exp["seed"])
    seed_everything(seed)
    device = device or get_device(cfg)
    num_beams = int(cfg.beam["num_beams"])
    horizons = list(exp["model"]["future_horizons"])
    fut_w = float(exp.get("future_loss_weight", 1.0))
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or core_kind

    tr, va, te = build_loaders(cfg, exp, modalities, splits)
    model = BeamModel(modalities, num_beams, core_kind, exp["model"]).to(device)
    if model.future_heads is None:
        raise RuntimeError("future_horizons not set on the model — check config")
    if channels_last(cfg):
        model = model.to(memory_format=torch.channels_last)

    tcfg = exp["train"]
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

    def sel(m):  # scalar to pick the best checkpoint: mean of current + all horizons
        return (m["current"] + sum(m[f"t+{h}"] for h in horizons)) / (1 + len(horizons))

    best = {"score": -1.0}; best_state = None; gstep = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train(); run = 0.0; ep_t0 = time.time(); nb = len(tr)
        for i, batch in enumerate(tr, 1):
            for g in opt.param_groups:
                g["lr"] = base_lr * lr_at(gstep)
            inp = _move(batch["inputs"], device, chlast)
            beam = batch["beam"].to(device, non_blocking=True)
            fut = batch["future"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_context(cfg, device):
                logits, flog = model(inp)
                loss = crit(logits.float(), beam)
                floss = sum(crit(flog[:, k].float(), fut[:, k]) for k in range(len(horizons))) / len(horizons)
                loss = loss + fut_w * floss
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
            run += loss.item(); gstep += 1
            if verbose and (i % 5 == 0 or i == nb):
                _progress(core_kind, epoch, epochs, i, nb, run / i, base_lr * lr_at(gstep), ep_t0)
        val = evaluate_future(model, va, device, cfg, horizons)
        score = sel(val)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, **val}
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            if sys.stdout.isatty():
                sys.stdout.write("\r" + " " * 110 + "\r")
            hs = " ".join(f"t+{h}={val[f't+{h}']:.3f}" for h in horizons)
            print(f"  [{core_kind:11s}] ep {epoch+1:02d}/{epochs} loss={run/steps_per_epoch:.3f} "
                  f"cur={val['current']:.3f} {hs}", flush=True)
    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_dir is not None:
            torch.save({"model": best_state, "core": core_kind, "modalities": modalities,
                        "seed": seed, "horizons": horizons, "best_val": best},
                       save_dir / f"ckpt_{tag}.pt")
    test = evaluate_future(model, te, device, cfg, horizons) if te is not None else {}
    return {"core": core_kind, "modalities": modalities, "seed": seed, "horizons": horizons,
            "best_val": best, "test_s33": test,
            "core_params": count_parameters(model.core_module()),
            "total_params": count_parameters(model), "train_time_s": train_time,
            "splits": splits.summary()}

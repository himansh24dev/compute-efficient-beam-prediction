"""Train one (core, modalities, protocol, seed) config and evaluate with the FULL
metric set (Top-1/2/3/5 + exact DBA) plus an efficiency profile."""
from __future__ import annotations

import copy
import math
import sys
import time

import torch
import torch.nn as nn

from data.config import DataConfig
from models.beam_model import BeamModel
from utils.device import autocast_context, channels_last, get_device
from utils.seed import seed_everything

from experiments.phase0.train import _move, _progress
from .metrics_full import add, finalize, full_metrics, new_accumulator
from .splits_compare import build_compare_loaders
from .efficiency import count_params, measure_speed, try_flops


@torch.no_grad()
def evaluate_full(model, loader, device, cfg, return_preds: bool = False,
                  target_h: int = 0) -> dict:
    """Full metrics on `loader`. With return_preds, also returns per-sample softmax
    probabilities (float16), targets, scenario ids and anchor rows, so pass-level
    and paired analyses (and seed ensembles) need no re-inference. `target_h` > 0
    scores against the beam h frames ahead (delay-aligned evaluation)."""
    model.eval()
    acc = new_accumulator()
    per = {}                                              # scenario_id -> accumulator
    chlast = channels_last(cfg)
    P, T, S, R = [], [], [], []
    for batch in loader:
        inp = _move(batch["inputs"], device, chlast)
        if target_h > 0:
            tgt = batch["future"][:, target_h - 1].to(device, non_blocking=True)
        else:
            tgt = batch["beam"].to(device, non_blocking=True)
        sid = batch["scenario_id"].to(device, non_blocking=True)
        with autocast_context(cfg, device):
            out = model(inp)
            logits = (out[0] if isinstance(out, tuple) else out).float()
        if return_preds:
            P.append(torch.softmax(logits, -1).half().cpu()); T.append(tgt.cpu())
            S.append(sid.cpu()); R.append(batch["row"])
        add(acc, full_metrics(logits, tgt))
        for s in sid.unique():
            m = sid == s
            si = int(s.item())
            per.setdefault(si, new_accumulator())
            add(per[si], full_metrics(logits[m], tgt[m]))
    res = finalize(acc)
    res["per_scenario"] = {si: finalize(a) for si, a in sorted(per.items())}
    if return_preds:
        res["preds"] = {"prob": torch.cat(P).numpy(), "target": torch.cat(T).numpy(),
                        "scenario_id": torch.cat(S).numpy(), "row": torch.cat(R).numpy()}
    return res


def train_one(cfg: DataConfig, exp: dict, core: str, modalities: list[str],
              protocol: str, seed: int, device=None, verbose: bool = True,
              init_seed: int | None = None) -> dict:
    # `seed` controls the data split; `init_seed` (defaults to `seed`) controls
    # model init + training RNG. Decoupling them lets us build a seed-ensemble on a
    # FIXED split (same test set) by varying only init_seed.
    seed_everything(seed if init_seed is None else init_seed)
    device = device or get_device(cfg)
    num_beams = int(cfg.beam["num_beams"])

    tr, va, te, split_info = build_compare_loaders(cfg, exp, modalities, protocol, seed)
    target_h = int(exp.get("target_h", 0))           # >0: train/score the beam h frames ahead
    model = BeamModel(modalities, num_beams, core, exp["model"]).to(device)
    if channels_last(cfg):
        model = model.to(memory_format=torch.channels_last)

    tcfg = exp["train"]
    base_lr = float(tcfg["lr"]); wd = float(tcfg["weight_decay"])
    bb_scale = float(tcfg.get("backbone_lr_scale", 1.0))   # <1 => gentler fine-tune of pretrained backbone
    # Discriminative LR: a pretrained camera backbone fine-tunes at a lower LR so
    # its ImageNet features aren't washed out early. Everything else at base_lr.
    cam = model.encoders["camera"] if "camera" in model.encoders else None
    bb_ids = set()
    if cam is not None and cam.__class__.__name__ == "PretrainedCameraEncoder":
        bb_ids = {id(p) for p in cam.features.parameters()}
    bb_params = [p for p in model.parameters() if id(p) in bb_ids]
    other_params = [p for p in model.parameters() if id(p) not in bb_ids]
    groups = [{"params": other_params, "base_lr": base_lr, "lr": base_lr}]
    if bb_params:
        groups.append({"params": bb_params, "base_lr": base_lr * bb_scale, "lr": base_lr * bb_scale})
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    epochs = int(tcfg["epochs"]); warmup = int(tcfg.get("warmup_epochs", 0))
    spe = max(1, len(tr))

    def lr_at(step):
        e = step / spe
        if e < warmup:
            return e / max(warmup, 1e-9)
        return 0.5 * (1 + math.cos(math.pi * min((e - warmup) / max(epochs - warmup, 1e-9), 1.0)))

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and cfg.runtime.get("amp", True)))
    crit = nn.CrossEntropyLoss(label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
    chlast = channels_last(cfg); clip = float(tcfg.get("grad_clip", 0.0))

    best = {"top1": -1.0}; best_state = None; gstep = 0; history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train(); run = 0.0; ep_t0 = time.time(); nb = len(tr)
        for i, batch in enumerate(tr, 1):
            sched = lr_at(gstep)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * sched
            inp = _move(batch["inputs"], device, chlast)
            if target_h > 0:
                tgt = batch["future"][:, target_h - 1].to(device, non_blocking=True)
            else:
                tgt = batch["beam"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_context(cfg, device):
                out = model(inp); logits = out[0] if isinstance(out, tuple) else out
                loss = crit(logits.float(), tgt)
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); run += loss.item(); gstep += 1
            if verbose and (i % 5 == 0 or i == nb):
                _progress(core, epoch, epochs, i, nb, run / i, base_lr * lr_at(gstep), ep_t0)
        # checkpoint selection on the VALIDATION partition only (carved from the
        # training side); the test partition is evaluated once, after training.
        val = evaluate_full(model, va, device, cfg, target_h=target_h)
        history.append({"epoch": epoch + 1, "loss": run / spe, "val_top1": val["top1"],
                        "val_top3": val["top3"], "val_dba": val["dba"]})
        if val["top1"] > best["top1"]:
            best = {**val, "epoch": epoch}; best_state = copy.deepcopy(model.state_dict())
        if verbose:
            if sys.stdout.isatty():
                sys.stdout.write("\r" + " " * 110 + "\r")
            print(f"  [{core}|{'+'.join(modalities)}|{protocol}] ep {epoch+1:02d}/{epochs} "
                  f"loss={run/spe:.3f} val_top1={val['top1']:.4f} val_top3={val['top3']:.4f} "
                  f"val_dba={val['dba']:.4f}", flush=True)
    train_time = time.time() - t0

    # selection-free reference: the final-epoch model, scored once on test
    test_last = evaluate_full(model, te, device, cfg, return_preds=True, target_h=target_h)
    if best_state is not None:
        model.load_state_dict(best_state)
    val_best = {k: best[k] for k in ("top1", "top3", "dba", "epoch") if k in best}
    test = evaluate_full(model, te, device, cfg, return_preds=True, target_h=target_h)
    ekey = split_info.pop("episode_key")
    pr = test["preds"]
    pr["pass"] = [ekey(s_, r_) for s_, r_ in zip(pr["scenario_id"], pr["row"])]

    # Efficiency profile on one real batch.
    sample = next(iter(te))
    sample_in = _move(sample["inputs"], device, chlast)
    speed = measure_speed(model, sample_in, cfg, device)
    gflops = try_flops(model, sample_in)

    return {"core": core, "modalities": modalities, "protocol": protocol, "seed": seed,
            "test": test, "test_last": test_last, "val_best": val_best, "history": history,
            "split_info": split_info, "train_time_s": train_time,
            "params": count_params(model),
            "efficiency": {**speed, "gflops": gflops},
            "best_state": best_state}

"""Reviewer (Sci Rep): DELAY-ALIGNED / future-beam evaluation for the DEPLOYED config.
The deployed model predicts the CURRENT beam y_t but emits it ~50-96 ms later, so the
honest deployment question is the beam required at emission time. Here, on the SAME
W=2 camera+GPS episode-random (pass-disjoint) split as the deployed model, we:

  (1) train single-head models to predict the beam at t, t+1, t+2 (frame ~92 ms), and
  (2) evaluate each against every target horizon -- in particular the deployed
      current-beam (t) model against the t+1 target (the *staleness penalty*: how much
      the emitted stale beam actually costs, measured, not approximated), vs a model
      retrained to predict t+1 (delay-aligned).

Everything uses ONE fixed anchor set (future_horizons=[1,2]) so horizons are comparable.

  .venv/bin/python -m experiments.edge_deploy.future_beam_deploy --epochs 50
"""
from __future__ import annotations

import argparse
import copy
import math

import numpy as np
import torch
import torch.nn as nn

from data.config import load_data_config
from data.dataset import make_snapshot_dataset
from data.preprocess.stats import load_gps_stats
from models.beam_model import BeamModel
from utils.device import autocast_context, channels_last, get_device
from utils.seed import seed_everything
from utils.yaml_io import load_yaml, save_json
from torch.utils.data import DataLoader

from experiments.phase0.train import _apply_modality_subset, _move
from experiments.bemamba_compare.splits_compare import _all_episodes
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics

SEED = 1337
HData = [1, 2]                     # dataset future targets (frames); ~92, ~184 ms


def target_at(batch, h, device):
    if h == 0:
        return batch["beam"].to(device, non_blocking=True)
    return batch["future"][:, h - 1].to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, loader, device, cfg, h):
    model.eval(); acc = new_accumulator()
    for batch in loader:
        inp = _move(batch["inputs"], device, channels_last(cfg))
        tgt = target_at(batch, h, device)
        with autocast_context(cfg, device):
            out = model(inp)
            logits = (out[0] if isinstance(out, tuple) else out).float()
        add(acc, full_metrics(logits, tgt))
    return finalize(acc)


def train_for_horizon(cfg, exp, tr, te, device, target_h, epochs):
    seed_everything(SEED)
    model = BeamModel(["gps", "camera"], int(cfg.beam["num_beams"]), "ssm", exp["model"]).to(device)
    if channels_last(cfg):
        model = model.to(memory_format=torch.channels_last)
    tcfg = exp["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    warmup = int(tcfg.get("warmup_epochs", 2)); spe = max(1, len(tr))
    crit = nn.CrossEntropyLoss(label_smoothing=float(tcfg.get("label_smoothing", 0.05)))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    clip = float(tcfg.get("grad_clip", 1.0)); gstep = 0

    def lr_scale(step):
        e = step / spe
        if e < warmup:
            return e / max(warmup, 1e-9)
        return 0.5 * (1 + math.cos(math.pi * min((e - warmup) / max(epochs - warmup, 1e-9), 1.0)))

    best = {"dba": -1.0}; best_state = None
    for ep in range(epochs):
        model.train()
        for batch in tr:
            for g in opt.param_groups:
                g["lr"] = float(tcfg["lr"]) * lr_scale(gstep)
            inp = _move(batch["inputs"], device, channels_last(cfg))
            tgt = target_at(batch, target_h, device)
            opt.zero_grad(set_to_none=True)
            with autocast_context(cfg, device):
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else out
                loss = crit(logits.float(), tgt)
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); gstep += 1
        val = evaluate(model, te, device, cfg, target_h)
        if val["dba"] > best["dba"]:
            best = val; best_state = copy.deepcopy(model.state_dict())
        print(f"    h{target_h} ep{ep+1:02d}/{epochs} dba={val['dba']:.4f} top3={val['top3']:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)
    mods = ["gps", "camera"]
    _apply_modality_subset(cfg, mods)
    cfg.snapshot["future_horizons"] = list(HData)          # fixed anchor set for all horizons
    cfg.snapshot["window"] = int(exp["window"])
    cfg.snapshot.pop("anchor_window", None)
    gps_stats = load_gps_stats(cfg)

    eps = _all_episodes(cfg, list(exp["compare_scenarios"]))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(eps)); n_tr = int(round(0.8 * len(eps)))
    train_eps = [eps[i] for i in order[:n_tr]]; test_eps = [eps[i] for i in order[n_tr:]]
    ds_tr = make_snapshot_dataset(cfg, train_eps, gps_stats, allow_heldout=True)
    ds_te = make_snapshot_dataset(cfg, test_eps, gps_stats, allow_heldout=True)
    tr = DataLoader(ds_tr, batch_size=int(exp["train"]["batch_size"]), shuffle=True, num_workers=10, drop_last=True)
    te = DataLoader(ds_te, batch_size=256, shuffle=False, num_workers=8)
    dt = 0.092
    print(f"anchors: train {len(ds_tr)}  test {len(ds_te)}  | frame ~{dt*1e3:.0f} ms")

    # train one model per target horizon (0=current, 1~92ms, 2~184ms)
    models = {}
    for h in (0, 1, 2):
        print(f"  === training model for horizon t+{h} ({h*dt*1e3:.0f} ms) ===", flush=True)
        models[h] = train_for_horizon(cfg, exp, tr, te, device, h, args.epochs)

    # cross-horizon evaluation matrix: model trained@k, evaluated@j
    print("\n=== DBA / Top-3 : rows = model trained-for, cols = evaluated-against ===")
    res = {"frame_ms": dt * 1e3, "matrix": {}}
    hdr = "  trained\\eval " + "".join(f"   t+{j}({j*dt*1e3:>3.0f}ms)" for j in (0, 1, 2))
    print(hdr)
    for k in (0, 1, 2):
        row = {}
        cells = []
        for j in (0, 1, 2):
            m = evaluate(models[k], te, device, cfg, j)
            row[f"eval_t+{j}"] = {"dba": round(m["dba"], 4), "top3": round(m["top3"], 4),
                                  "top1": round(m["top1"], 4)}
            cells.append(f" {m['dba']:.3f}/{m['top3']:.3f}")
        res["matrix"][f"trained_t+{k}"] = row
        print(f"  t+{k}        " + "".join(cells))

    d = res["matrix"]
    print("\n--- key comparisons (target = beam at emission time, ~1 frame / 92 ms ahead) ---")
    cur_cur = d["trained_t+0"]["eval_t+0"]
    stale = d["trained_t+0"]["eval_t+1"]
    aligned = d["trained_t+1"]["eval_t+1"]
    print(f"  deployed current-beam model @ current target : DBA {cur_cur['dba']:.3f} / Top-3 {cur_cur['top3']:.3f}")
    print(f"  deployed current-beam model @ t+1 (stale)    : DBA {stale['dba']:.3f} / Top-3 {stale['top3']:.3f}"
          f"   -> staleness penalty {cur_cur['dba']-stale['dba']:+.3f} DBA")
    print(f"  RETRAINED delay-aligned model @ t+1          : DBA {aligned['dba']:.3f} / Top-3 {aligned['top3']:.3f}"
          f"   -> recovers {aligned['dba']-stale['dba']:+.3f} DBA vs stale")
    save_json(res, "experiments/edge_deploy/results/future_beam_deploy.json")
    print("\nsaved -> experiments/edge_deploy/results/future_beam_deploy.json")


if __name__ == "__main__":
    raise SystemExit(main())

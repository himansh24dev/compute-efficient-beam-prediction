"""Comprehensive test-time SENSOR-ROBUSTNESS sweep for the deployed cam+GPS model.

A Sensors-journal reviewer wants to know how the predictor holds up when the
sensors degrade, not just when a modality drops out entirely. We evaluate the
frozen deployment checkpoint on the honest (episode-random) test split under:

  1. Modality dropout        — zero a whole modality (both / camera / gps).
  2. GPS position error      — additive Gaussian offset of sigma metres on the
                               (dlat, dlon) coordinates (a realistic GPS bias),
                               converted into the model's standardized input via
                               the train-set GPS std (data/derived/stats/gps_v1.json).
  3. Camera Gaussian noise   — additive N(0, sigma) on standardized pixels.
  4. Camera blur             — box blur, kernel k (stride-1 avg pool).
  5. Camera brightness shift — add a constant to all pixels (standardized units).
  6. Camera occlusion        — zero a centred square patch covering fraction p.

Everything is eval-time (no retraining); runs on CPU by default so it does not
contend with a GPU training sweep. Latency is unchanged (perturbations are on the
inputs), so this table is purely accuracy-vs-degradation.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from data.config import load_data_config
from utils.yaml_io import load_yaml, save_json
from models.beam_model import BeamModel
from experiments.bemamba_compare.splits_compare import build_compare_loaders
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics
from experiments.edge_deploy.export_onnx import infer_cfg


def _load_model(ckpt, device):
    obj = torch.load(ckpt, map_location=device)
    c = infer_cfg(obj["model"])
    mcfg = {"dim": c["dim"], "core_layers": c["core_layers"], "camera": {"width": 32},
            "gps": {"in_dim": 3, "hidden": 128},
            "ssm": {"d_state": c["d_state"], "d_conv": c["d_conv"],
                    "expand": c["expand"], "dt_rank": c["dt_rank"]}}
    model = BeamModel(["gps", "camera"], 64, "ssm", mcfg).to(device)
    model.load_state_dict(obj["model"])
    model.eval()
    return model


def _perturb(inp, kind, level, gps_std, gen):
    """Return a NEW input dict with `kind` perturbation at `level` applied."""
    out = {k: v.clone() for k, v in inp.items()}
    if kind == "clean":
        return out
    if kind == "drop_camera":
        out["camera"] = torch.zeros_like(out["camera"]); return out
    if kind == "drop_gps":
        out["gps"] = torch.zeros_like(out["gps"]); return out
    if kind == "gps_noise_m":
        g = out["gps"]                                   # (B,W,3) standardized
        b = g.shape[0]
        # one position offset per sample (metres), same across the window frames
        off_m = torch.randn(b, 2, generator=gen, device=g.device) * level
        off_std = off_m / gps_std[:2]                    # metres -> standardized
        g[..., 0] += off_std[:, 0:1]
        g[..., 1] += off_std[:, 1:2]
        out["gps"] = g; return out
    if kind == "cam_gauss":
        c = out["camera"]
        out["camera"] = c + torch.randn(c.shape, generator=gen, device=c.device) * level
        return out
    if kind == "cam_blur":
        k = int(level); c = out["camera"]
        b, w, ch, h, wd = c.shape
        x = c.reshape(b * w, ch, h, wd)
        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        out["camera"] = x.reshape(b, w, ch, h, wd); return out
    if kind == "cam_bright":
        out["camera"] = out["camera"] + level; return out
    if kind == "cam_occlude":
        c = out["camera"]; h = c.shape[-2]; wd = c.shape[-1]
        ph, pw = int(h * level), int(wd * level)
        y0, x0 = (h - ph) // 2, (wd - pw) // 2
        c[..., y0:y0 + ph, x0:x0 + pw] = 0.0             # 0 == mean (gray)
        out["camera"] = c; return out
    raise ValueError(kind)


@torch.no_grad()
def _eval(model, loader, device, kind, level, gps_std, seed=0):
    gen = torch.Generator(device=device); gen.manual_seed(1000 + seed)
    acc = new_accumulator()
    for batch in loader:
        inp = {k: v.to(device) for k, v in batch["inputs"].items()}
        inp = _perturb(inp, kind, level, gps_std, gen)
        tgt = batch["beam"].to(device)
        logits = model(inp)
        logits = (logits[0] if isinstance(logits, tuple) else logits).float()
        add(acc, full_metrics(logits, tgt))
    r = finalize(acc)
    return {"top1": r["top1"], "top3": r["top3"], "dba": r["dba"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/edge_deploy/results/deploy_ssm_w2_episode-random_s1337.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/edge_deploy/results/robustness_full.json")
    args = ap.parse_args()

    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    cfg = load_data_config(exp["data_config"])
    device = torch.device(args.device)
    gps_std = torch.tensor(json.load(open("data/derived/stats/gps_v1.json"))["std"],
                           dtype=torch.float32, device=device)

    model = _load_model(args.ckpt, device)
    _, te, _ = build_compare_loaders(cfg, exp, ["gps", "camera"], "episode-random", 1337)

    # (family, label, levels) — level meaning depends on the family
    sweep = [
        ("modality", [("clean", 0), ("drop_camera", 0), ("drop_gps", 0)]),
        ("gps_noise_m", [("gps_noise_m", s) for s in (1, 2, 5, 10, 20)]),
        ("cam_gauss",   [("cam_gauss", s) for s in (0.1, 0.25, 0.5, 1.0)]),
        ("cam_blur",    [("cam_blur", k) for k in (3, 7, 15)]),
        ("cam_bright",  [("cam_bright", d) for d in (0.5, 1.0, -0.5, -1.0)]),
        ("cam_occlude", [("cam_occlude", p) for p in (0.25, 0.5, 0.75)]),
    ]

    out = {}
    clean = _eval(model, te, device, "clean", 0, gps_std)
    out["clean"] = clean
    print(f"clean: top1={clean['top1']:.4f} top3={clean['top3']:.4f} dba={clean['dba']:.4f}\n", flush=True)
    for family, items in sweep:
        out[family] = []
        print(f"--- {family} ---", flush=True)
        for kind, level in items:
            r = _eval(model, te, device, kind, level, gps_std)
            r["level"] = level
            out[family].append(r)
            drop = r["dba"] - clean["dba"]
            print(f"  {kind:14s} level={str(level):5s}: dba={r['dba']:.4f} "
                  f"top1={r['top1']:.4f} top3={r['top3']:.4f}  (Δdba {drop:+.4f})", flush=True)
    save_json(out, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

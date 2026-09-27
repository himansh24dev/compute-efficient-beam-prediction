"""Test-time sensor robustness, fully specified (reviewer comment 4).

Checkpoints: the five validation-selected camera+GPS W=2 models on the headline
split (A_ssm_2M_i*), evaluated on the 23 held-out test passes, no retraining.
Every stochastic perturbation is drawn R=10 times per checkpoint (fixed seeds), and
we report the mean and the 2.5/97.5 percentiles over the 5 x 10 (checkpoint x
realization) evaluations.

GPS position error, in metres, in the base-station-relative local tangent plane
(north, east); speed is left unperturbed. Offsets are added to the raw metres and
then standardized with the split's train-only statistics, exactly like clean inputs.
Three error processes, each with per-axis standard deviation sigma (so the RMS
horizontal error is sigma*sqrt(2)):
  iid_window : one isotropic Gaussian offset per test window, shared by the W=2
               frames of that window, independent across windows (the original
               submission's model; the most favourable to averaging).
  pass_bias  : one isotropic Gaussian offset per vehicle pass, shared by every frame
               of that pass (a fully correlated bias, the least favourable).
  gauss_markov: first-order Gauss-Markov process per pass and axis,
               e_k = a e_{k-1} + sqrt(1-a^2) sigma n_k, a = exp(-dt/tau), dt = 92 ms
               (the DeepSense frame interval), tau = 10 s, stationary std sigma --
               the standard first-order model of slowly varying GNSS error.
Camera perturbations act on the ImageNet-standardized 224x224 input (one standardized
unit ~ 0.226 of the [0,1] pixel range, i.e. ~58/255):
  cam_gauss  : i.i.d. N(0, sigma^2) per pixel and channel, sigma in {0.1,0.25,0.5,1.0}
  cam_blur   : box blur, k x k mean filter (stride 1, zero padding), k in {3,7,15}
  cam_bright : additive constant d in {+-0.5, +-1.0} standardized units
  cam_occlude: centred rectangle set to 0 (= the ImageNet mean colour) covering a
               fraction p in {0.25, 0.5, 0.75} of the height and of the width
               (so an area fraction p^2)
Missing-sensor tests replace one modality's standardized input by zeros, i.e. the
training-mean position (GPS) or a mean-colour frame (camera).

  .venv/bin/python -m experiments.revision.robustness
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.config import load_data_config
from models.beam_model import BeamModel
from utils.yaml_io import load_yaml, save_json
from experiments.bemamba_compare.splits_compare import build_compare_loaders
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics

CKPTS = [f"experiments/revision/results/ckpt/A_ssm_2M_i{s}.pt" for s in (1337, 2024, 7, 42, 123)]
OUT = "experiments/revision/results/robustness.json"
R = 10
DT, TAU = 0.092, 10.0


def load(ck, device):
    o = torch.load(ck, map_location=device)
    m = BeamModel(o["modalities"], 64, o["core_kind"], o["model_cfg"]).to(device)
    m.load_state_dict(o["model"]); m.eval()
    return m, o


def gps_offsets(kind, sigma, batch, rows_by_pass, rng_state):
    """Per-sample, per-frame offsets in metres, shape (B, W, 2)."""
    sid, row = batch["scenario_id"].numpy(), batch["row"].numpy()
    b, w = batch["inputs"]["gps"].shape[:2]
    out = np.zeros((b, w, 2), np.float32)
    if kind == "iid_window":
        out[:] = rng_state["rng"].standard_normal((b, 1, 2)) * sigma
        return out
    table = rng_state["table"]                       # (sid,row) -> offset (2,)
    for i in range(b):
        for j in range(w):
            out[i, j] = table[(int(sid[i]), int(row[i]) - (w - 1 - j))]
    return out


def build_table(kind, sigma, passes, rng):
    """Per-frame offsets for pass-correlated processes: {(sid,row): (2,)}."""
    table = {}
    a = np.exp(-DT / TAU)
    for sid, r0, r1 in passes:
        n = r1 - r0
        if kind == "pass_bias":
            e = np.repeat(rng.standard_normal((1, 2)) * sigma, n, axis=0)
        else:  # gauss_markov, stationary start
            e = np.zeros((n, 2)); e[0] = rng.standard_normal(2) * sigma
            for k in range(1, n):
                e[k] = a * e[k - 1] + np.sqrt(1 - a * a) * sigma * rng.standard_normal(2)
        for k in range(n):
            table[(sid, r0 + k)] = e[k].astype(np.float32)
    return table


@torch.no_grad()
def evaluate(model, te, device, gps_std, kind="clean", level=0.0, seed=0, passes=None):
    rng = np.random.default_rng(10_000 + seed)
    gen = torch.Generator(device=device); gen.manual_seed(10_000 + seed)
    state = {"rng": rng}
    if kind.startswith("gps_") and kind != "gps_iid_window":
        state["table"] = build_table(kind[4:], level, passes, rng)
    acc = new_accumulator()
    for batch in te:
        inp = {k: v.to(device) for k, v in batch["inputs"].items()}
        if kind == "drop_camera":
            inp["camera"] = torch.zeros_like(inp["camera"])
        elif kind == "drop_gps":
            inp["gps"] = torch.zeros_like(inp["gps"])
        elif kind.startswith("gps_"):
            off = torch.from_numpy(gps_offsets(kind[4:], level, batch, None, state)).to(device)
            inp["gps"][..., :2] += off / gps_std[:2]
        elif kind == "cam_gauss":
            c = inp["camera"]
            inp["camera"] = c + torch.randn(c.shape, generator=gen, device=device) * level
        elif kind == "cam_blur":
            k = int(level); c = inp["camera"]; b, w, ch, h, wd = c.shape
            x = F.avg_pool2d(c.reshape(b * w, ch, h, wd), kernel_size=k, stride=1, padding=k // 2,
                             count_include_pad=True)
            inp["camera"] = x.reshape(b, w, ch, h, wd)
        elif kind == "cam_bright":
            inp["camera"] = inp["camera"] + level
        elif kind == "cam_occlude":
            c = inp["camera"]; h, wd = c.shape[-2:]
            ph, pw = int(h * level), int(wd * level); y0, x0 = (h - ph) // 2, (wd - pw) // 2
            c[..., y0:y0 + ph, x0:x0 + pw] = 0.0
        out = model(inp)
        logits = (out[0] if isinstance(out, tuple) else out).float()
        add(acc, full_metrics(logits, batch["beam"].to(device)))
    r = finalize(acc)
    return {"dba": r["dba"], "top1": r["top1"], "top3": r["top3"]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    exp["split"]["val_frac"] = 0.15
    sweep = [("clean", [0.0], 1), ("drop_camera", [0.0], 1), ("drop_gps", [0.0], 1),
             ("gps_iid_window", [1, 2, 3, 5, 10, 20], R),
             ("gps_pass_bias", [1, 2, 3, 5, 10, 20], R),
             ("gps_gauss_markov", [1, 2, 3, 5, 10, 20], R),
             ("cam_gauss", [0.1, 0.25, 0.5, 1.0], R),
             ("cam_blur", [3, 7, 15], 1), ("cam_bright", [0.5, 1.0, -0.5, -1.0], 1),
             ("cam_occlude", [0.25, 0.5, 0.75], 1)]
    raw = defaultdict(list)                       # (kind, level) -> list of dicts
    for ck in CKPTS:
        model, o = load(ck, device)
        cfg = load_data_config(exp["data_config"])
        _, _, te, info = build_compare_loaders(cfg, exp, o["modalities"], "episode-random", o["split_seed"])
        assert np.allclose(info["gps_stats"]["std"], o["gps_stats"]["std"])
        gps_std = torch.tensor(o["gps_stats"]["std"], dtype=torch.float32, device=device)
        from experiments.bemamba_compare.splits_compare import split_episodes
        _, _, test_eps = split_episodes(cfg, exp, o["split_seed"])
        passes = [(e.scenario_id, e.row_start, e.row_end) for e in test_eps]
        te = list(te)                              # materialise the test set once (~4.7 GB)
        for kind, levels, reps in sweep:
            for lv in levels:
                for rr in range(reps):
                    raw[(kind, lv)].append(evaluate(model, te, device, gps_std, kind, lv, rr, passes))
        print(f"done {ck}", flush=True)
    res = {}
    for (kind, lv), rs in raw.items():
        d = np.array([x["dba"] for x in rs]); t3 = np.array([x["top3"] for x in rs])
        res.setdefault(kind, []).append({
            "level": lv, "n_evals": len(rs), "dba": round(float(d.mean()), 4),
            "dba_lo": round(float(np.percentile(d, 2.5)), 4), "dba_hi": round(float(np.percentile(d, 97.5)), 4),
            "top3": round(float(t3.mean()), 4)})
        print(f"{kind:18s} {str(lv):6s} DBA {d.mean():.3f} [{np.percentile(d,2.5):.3f},{np.percentile(d,97.5):.3f}]")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    save_json(res, OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())

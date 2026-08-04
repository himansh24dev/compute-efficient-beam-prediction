"""Reviewer N8: the 5-seed CIs vary the training seed against a FIXED pass-level test
set, so they estimate seed variance while holding constant the dominant uncertainty --
WHICH ~21 passes land in test. Frames within a pass are strongly correlated (the
paper's own Sec.VI-B thesis), so the effective sample size for generalisation is the
number of test PASSES (~21), not the number of frames (~3900).

This computes a pass-level bootstrap CI for the deployed W=2 model's flagship numbers
by resampling whole test EPISODES (=passes) with replacement -- NO retraining. It
reproduces the exact seed-1337 episode-random split the deployed checkpoint was
trained on, so the pooled point estimate matches the paper's 0.865 DBA.

  .venv/bin/python -m experiments.edge_deploy.passlevel_ci
"""
from __future__ import annotations

import numpy as np
import torch

from data.config import load_data_config
from data.dataset import make_snapshot_dataset
from data.preprocess.stats import load_gps_stats
from models.beam_model import BeamModel
from utils.device import autocast_context, get_device
from utils.yaml_io import load_yaml, save_json
from torch.utils.data import DataLoader

from experiments.phase0.train import _apply_modality_subset, _move
from experiments.bemamba_compare.splits_compare import _all_episodes

CKPT = "experiments/edge_deploy/results/deploy_ssm_w2_episode-random_s1337.pt"
SEED = 1337
DELTA = 5.0


def metrics(preds, tgt):
    """preds: (N,5) best-first predicted beam indices; tgt: (N,). Exact DBA (K=3,
    Delta=5) + Top-1/2/3/5, matching metrics_full.full_metrics."""
    tgt = tgt[:, None]
    hit = preds == tgt
    top = {k: float(hit[:, :k].any(1).mean()) for k in (1, 2, 3, 5)}
    dist = np.clip(np.abs(preds[:, :3] - tgt) / DELTA, None, 1.0)     # (N,3)
    cmin = np.minimum.accumulate(dist, axis=1)                        # cumulative min over top-1..j
    etas = [1.0 - cmin[:, k].mean() for k in range(3)]
    return float(np.mean(etas)), top


def main():
    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    cfg = load_data_config(exp["data_config"])
    device = get_device(cfg)

    # --- replicate build_compare_loaders' cfg mutations + episode-random split ---
    mods = ["gps", "camera"]
    _apply_modality_subset(cfg, mods)
    cfg.snapshot["future_horizons"] = []
    cfg.snapshot["window"] = int(exp["window"])
    cfg.snapshot.pop("anchor_window", None)
    gps_stats = load_gps_stats(cfg)
    scenarios = list(exp["compare_scenarios"])
    frac = float(exp["split"]["train_frac"])

    eps = _all_episodes(cfg, scenarios)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(eps))
    n_tr = int(round(frac * len(eps)))
    test_eps = [eps[i] for i in order[n_tr:]]
    print(f"episodes total {len(eps)}  train {n_tr}  test(passes) {len(test_eps)}")

    # --- load deployed checkpoint ---
    ck = torch.load(CKPT, map_location=device)
    model = BeamModel(mods, int(cfg.beam["num_beams"]), "ssm", exp["model"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {CKPT}  (reported val dba={ck['val']['dba']:.4f} top3={ck['val']['top3']:.4f})")

    # --- per-pass predictions ---
    per_pass = []   # list of (preds(Ni,5), tgt(Ni,))
    for ep in test_eps:
        ds = make_snapshot_dataset(cfg, [ep], gps_stats, allow_heldout=True)
        if len(ds) == 0:
            continue
        dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=6)
        P, T = [], []
        with torch.no_grad():
            for batch in dl:
                inp = _move(batch["inputs"], device, False)
                with autocast_context(cfg, device):
                    out = model(inp)
                    logits = (out[0] if isinstance(out, tuple) else out).float()
                P.append(logits.topk(5, dim=-1).indices.cpu().numpy())
                T.append(batch["beam"].numpy())
        per_pass.append((np.concatenate(P), np.concatenate(T)))

    n_frames = sum(len(t) for _, t in per_pass)
    all_p = np.concatenate([p for p, _ in per_pass])
    all_t = np.concatenate([t for _, t in per_pass])
    dba0, top0 = metrics(all_p, all_t)
    print(f"\npooled point estimate ({len(per_pass)} passes, {n_frames} frames):")
    print(f"  DBA {dba0:.4f}  Top-1 {top0[1]:.4f}  Top-2 {top0[2]:.4f}  "
          f"Top-3 {top0[3]:.4f}  Top-5 {top0[5]:.4f}")

    # --- bootstrap over PASSES (with replacement) ---
    B = 5000
    boot = np.random.default_rng(0)
    idx = np.arange(len(per_pass))
    D, T1, T2, T3, T5 = [], [], [], [], []
    for _ in range(B):
        pick = boot.choice(idx, size=len(idx), replace=True)
        p = np.concatenate([per_pass[i][0] for i in pick])
        t = np.concatenate([per_pass[i][1] for i in pick])
        d, tk = metrics(p, t)
        D.append(d); T1.append(tk[1]); T2.append(tk[2]); T3.append(tk[3]); T5.append(tk[5])

    def ci(name, xs, pt):
        xs = np.array(xs)
        lo, hi = np.percentile(xs, [2.5, 97.5])
        half = (hi - lo) / 2
        print(f"  {name:6s} {pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (+/-{half:.3f}, sd {xs.std():.3f})")
        return {"point": pt, "lo": float(lo), "hi": float(hi), "half": float(half), "sd": float(xs.std())}

    print(f"\npass-level bootstrap ({B} resamples of {len(per_pass)} passes):")
    res = {"n_passes": len(per_pass), "n_frames": int(n_frames), "B": B,
           "dba": ci("DBA", D, dba0), "top1": ci("Top-1", T1, top0[1]),
           "top2": ci("Top-2", T2, top0[2]), "top3": ci("Top-3", T3, top0[3]),
           "top5": ci("Top-5", T5, top0[5])}
    save_json(res, "experiments/edge_deploy/results/passlevel_ci.json")
    print("\nsaved -> experiments/edge_deploy/results/passlevel_ci.json")


if __name__ == "__main__":
    raise SystemExit(main())

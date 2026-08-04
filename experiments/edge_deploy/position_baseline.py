"""Cheap non-deep POSITION-ONLY baselines (reviewer C3): does a k-NN / lookup over
GPS already match the deployed model, and how fast is it? Uses the SAME
episode-random (leakage-free) split and the SAME DBA/Top-k metrics as the model,
so the numbers are directly comparable to Table IV.

Baselines:
  * majority   — always predict the most frequent training beam (a prior; the
                 reference for "graceful failure", reviewer C7).
  * knn(k)     — k-nearest-neighbour over the (standardized) GPS window; each of
                 the 64 beams is scored by inverse-distance-weighted votes of the
                 k neighbours, giving a full ranking for Top-k + DBA.
Reports accuracy AND per-query latency (brute-force numpy) so we can put position
lookup on the Pareto/latency picture.

  .venv/bin/python -m experiments.edge_deploy.position_baseline
"""
from __future__ import annotations

import json
import time

import numpy as np

from data.config import load_data_config
from utils.yaml_io import load_yaml, save_json
from experiments.bemamba_compare.splits_compare import build_compare_loaders
from experiments.bemamba_compare.metrics_full import new_accumulator, add, finalize, full_metrics

import torch


def _collect(loader):
    G, Y = [], []
    for b in loader:
        G.append(b["inputs"]["gps"].reshape(b["inputs"]["gps"].shape[0], -1).numpy())
        Y.append(b["beam"].numpy())
    return np.concatenate(G).astype(np.float32), np.concatenate(Y).astype(np.int64)


def _metrics_from_scores(scores, targets):
    acc = new_accumulator()
    add(acc, full_metrics(torch.from_numpy(scores).float(), torch.from_numpy(targets).long()))
    r = finalize(acc)
    return {"top1": r["top1"], "top3": r["top3"], "dba": r["dba"]}


def main():
    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    cfg = load_data_config(exp["data_config"])
    tr, te, info = build_compare_loaders(cfg, exp, ["gps"], "episode-random", 1337)
    Xtr, Ytr = _collect(tr)
    Xte, Yte = _collect(te)
    nb = int(cfg.beam["num_beams"])
    print(f"train {Xtr.shape} test {Xte.shape}  ({nb} beams)", flush=True)

    out = {"n_train": len(Xtr), "n_test": len(Xte)}

    # --- majority prior ---
    maj = np.bincount(Ytr, minlength=nb).argmax()
    sc = np.zeros((len(Xte), nb), np.float32); sc[:, maj] = 1.0
    out["majority"] = _metrics_from_scores(sc, Yte)
    print(f"majority (beam {maj}): {out['majority']}", flush=True)

    # --- k-NN over the GPS window (inverse-distance weighted beam votes) ---
    # brute-force squared distances; measure per-query latency on a sample.
    for k in (1, 5, 20):
        d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)      # (Nte, Ntr)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]           # k nearest
        scores = np.zeros((len(Xte), nb), np.float32)
        for i in range(len(Xte)):
            nn = idx[i]
            w = 1.0 / (np.sqrt(d2[i, nn]) + 1e-6)
            np.add.at(scores[i], Ytr[nn], w)
        out[f"knn{k}"] = _metrics_from_scores(scores, Yte)
        print(f"knn(k={k}): {out[f'knn{k}']}", flush=True)

    # --- latency of a single k=20 query (brute force over the train set) ---
    q = Xte[0]
    t0 = time.perf_counter()
    for _ in range(200):
        dd = ((Xtr - q) ** 2).sum(-1)
        np.argpartition(dd, 20)[:20]
    lat_ms = (time.perf_counter() - t0) / 200 * 1000
    out["knn_query_ms"] = round(lat_ms, 3)
    print(f"\nk-NN brute-force query latency (train={len(Xtr)}): {lat_ms:.3f} ms/query", flush=True)

    out["deployed_model_ref"] = {"dba": 0.865, "top3": 0.809, "top1": 0.434, "latency_ms": 23.6}
    save_json(out, "experiments/edge_deploy/results/position_baseline.json")
    print("\nsaved -> results/position_baseline.json")
    print(f"\nREFERENCE deployed cam+GPS model: DBA 0.865 / Top-3 0.809 @ 23.6 ms")
    print("If knn DBA ~ 0.85, the camera branch must justify its ~19 ms elsewhere (C3).")


if __name__ == "__main__":
    raise SystemExit(main())

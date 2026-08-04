"""Two comparison protocols over scenarios 31-33 (34 stays held out):

  window-random  : random 80/20 split at the WINDOW level (BeMamba's protocol;
                   temporally-adjacent frames leak across train/test -> optimistic).
  episode-random : random 80/20 split at the EPISODE level (leakage-free; ours).

Same scenarios and same ~80/20 ratio, so the gap between the two isolates exactly
how much the random-window split inflates accuracy.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Subset

from data.config import DataConfig
from data.dataset import make_snapshot_dataset
from data.index import build_episodes
from data.preprocess.stats import load_gps_stats
from utils.device import dataloader_kwargs

from experiments.phase0.train import _apply_modality_subset


def _all_episodes(cfg, scenarios):
    eps = []
    for sid in scenarios:
        # allow_heldout=True so scenario 34 can be included to MATCH BeMamba's
        # random 80/20 over all 31-34. This comparison deliberately un-holds-out 34.
        _, e = build_episodes(cfg, sid, allow_heldout=True)
        eps.extend(e)
    return eps


def build_compare_loaders(cfg: DataConfig, exp: dict, modalities: list[str],
                          protocol: str, seed: int):
    _apply_modality_subset(cfg, modalities)
    cfg.snapshot["future_horizons"] = []
    cfg.snapshot["window"] = int(exp.get("window", cfg.snapshot["window"]))
    cfg.snapshot.pop("anchor_window", None)
    scenarios = list(exp["compare_scenarios"])            # e.g. [31,32,33]
    frac = float(exp["split"]["train_frac"])
    gps_stats = load_gps_stats(cfg) if "gps" in modalities else None

    cfg.loader["num_workers"] = int(exp["train"].get("num_workers", 8))
    cfg.loader["batch_size"] = int(exp["train"]["batch_size"])
    cfg.loader["prefetch_factor"] = int(exp["train"].get("prefetch_factor", 2))
    tr_kw = dataloader_kwargs(cfg, train=True)
    ev_kw = dataloader_kwargs(cfg, train=False)

    rng = np.random.default_rng(seed)
    eps = _all_episodes(cfg, scenarios)

    if protocol == "episode-random":
        order = rng.permutation(len(eps))
        n_tr = int(round(frac * len(eps)))
        train_eps = [eps[i] for i in order[:n_tr]]
        test_eps = [eps[i] for i in order[n_tr:]]
        ds_tr = make_snapshot_dataset(cfg, train_eps, gps_stats, allow_heldout=True)
        ds_te = make_snapshot_dataset(cfg, test_eps, gps_stats, allow_heldout=True)

    elif protocol == "window-random":
        ds_all = make_snapshot_dataset(cfg, eps, gps_stats, allow_heldout=True)   # all windows over 31-34
        n = len(ds_all)
        order = rng.permutation(n)
        n_tr = int(round(frac * n))
        ds_tr = Subset(ds_all, order[:n_tr].tolist())
        ds_te = Subset(ds_all, order[n_tr:].tolist())

    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    return (DataLoader(ds_tr, **tr_kw), DataLoader(ds_te, **ev_kw),
            {"protocol": protocol, "scenarios": scenarios, "n_train": len(ds_tr),
             "n_test": len(ds_te), "window": cfg.snapshot["window"]})

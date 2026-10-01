"""Two comparison protocols over scenarios 31-34:

  window-random  : random 80/20 split at the WINDOW level (BeMamba's protocol;
                   temporally-adjacent frames leak across train/test -> optimistic).
  episode-random : random 80/20 split at the EPISODE (vehicle-pass) level
                   (leakage-free; ours).

Same scenarios and same ~80/20 ratio, so the gap between the two isolates exactly
how much the random-window split inflates accuracy.

Validation (revision). A validation partition is carved from the TRAINING side
only, after the test partition has been fixed, and is used for checkpoint
selection; the test partition is never seen during training or selection.
  - episode-random: `val_frac` of the training PASSES are held out as validation,
    so train / val / test are mutually pass-disjoint. Windows are generated inside
    each pass after the partition, so no window spans two partitions and no frame
    of a validation or test pass is ever used for training.
  - window-random : `val_frac` of the training WINDOWS are held out (sample-level,
    consistent with that protocol's own sample-level construction).
The test partition is drawn exactly as in the original release (same RNG stream),
so the held-out test passes / windows are unchanged.

GPS standardization statistics are fit on the frames of the TRAINING partition of
this split only (never validation or test).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader, Subset

from data.config import DataConfig
from data.dataset import make_snapshot_dataset
from data.index import build_episodes
from data.preprocess.stats import fit_gps_stats_frames
from utils.device import dataloader_kwargs

from experiments.phase0.train import _apply_modality_subset

VAL_SEED_OFFSET = 100_003          # independent RNG stream for the validation draw


def _all_episodes(cfg, scenarios):
    eps = []
    for sid in scenarios:
        # allow_heldout=True so scenario 34 can be included to MATCH BeMamba's
        # random 80/20 over all 31-34. This comparison deliberately un-holds-out 34.
        _, e = build_episodes(cfg, sid, allow_heldout=True)
        eps.extend(e)
    return eps


def split_episodes(cfg, exp: dict, seed: int):
    """Episode-random partition -> (train_eps, val_eps, test_eps). Test identical to
    the original release; validation drawn from the training passes only."""
    eps = _all_episodes(cfg, list(exp["compare_scenarios"]))
    frac = float(exp["split"]["train_frac"])
    val_frac = float(exp["split"].get("val_frac", 0.15))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eps))
    n_tr = int(round(frac * len(eps)))
    pool = [eps[i] for i in order[:n_tr]]
    test_eps = [eps[i] for i in order[n_tr:]]
    vrng = np.random.default_rng(seed + VAL_SEED_OFFSET)
    vorder = vrng.permutation(len(pool))
    n_val = int(round(val_frac * len(pool)))
    val_eps = [pool[i] for i in vorder[:n_val]]
    train_eps = [pool[i] for i in vorder[n_val:]]
    return train_eps, val_eps, test_eps


def episode_frames(eps) -> dict[int, np.ndarray]:
    fr = defaultdict(list)
    for e in eps:
        fr[e.scenario_id].extend(range(e.row_start, e.row_end))
    return {k: np.asarray(v) for k, v in fr.items()}


def episode_lookup(eps):
    """(scenario_id, row) -> episode key, for mapping predictions back to passes."""
    by_sid = defaultdict(list)
    for e in eps:
        by_sid[e.scenario_id].append((e.row_start, e.row_end, e.key))
    def key(sid: int, row: int) -> str:
        for a, b, k in by_sid[int(sid)]:
            if a <= int(row) < b:
                return k
        raise KeyError((sid, row))
    return key


def build_compare_loaders(cfg: DataConfig, exp: dict, modalities: list[str],
                          protocol: str, seed: int):
    """Returns (train_loader, val_loader, test_loader, info)."""
    _apply_modality_subset(cfg, modalities)
    cfg.snapshot["future_horizons"] = list(exp.get("future_horizons", []))
    cfg.snapshot["window"] = int(exp.get("window", cfg.snapshot["window"]))
    cfg.snapshot.pop("anchor_window", None)
    if exp.get("anchor_window"):
        cfg.snapshot["anchor_window"] = int(exp["anchor_window"])
    frac = float(exp["split"]["train_frac"])
    val_frac = float(exp["split"].get("val_frac", 0.15))
    W = cfg.snapshot["window"]

    cfg.loader["num_workers"] = int(exp["train"].get("num_workers", 8))
    cfg.loader["batch_size"] = int(exp["train"]["batch_size"])
    cfg.loader["prefetch_factor"] = int(exp["train"].get("prefetch_factor", 2))
    tr_kw = dataloader_kwargs(cfg, train=True)
    ev_kw = dataloader_kwargs(cfg, train=False)
    eps_all = _all_episodes(cfg, list(exp["compare_scenarios"]))

    if protocol == "episode-random":
        train_eps, val_eps, test_eps = split_episodes(cfg, exp, seed)
        gps_stats = fit_gps_stats_frames(cfg, episode_frames(train_eps)) if "gps" in modalities else None
        ds_tr = make_snapshot_dataset(cfg, train_eps, gps_stats, allow_heldout=True)
        ds_va = make_snapshot_dataset(cfg, val_eps, gps_stats, allow_heldout=True)
        ds_te = make_snapshot_dataset(cfg, test_eps, gps_stats, allow_heldout=True)
        parts = {"train_passes": [e.key for e in train_eps],
                 "val_passes": [e.key for e in val_eps],
                 "test_passes": [e.key for e in test_eps]}

    elif protocol == "window-random":
        # all windows over 31-34; windows carry their anchor row, so the training
        # frames (for GPS stats) are exactly the frames the training windows read.
        ds_probe = make_snapshot_dataset(cfg, eps_all, None, allow_heldout=True)
        n = len(ds_probe)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        n_tr = int(round(frac * n))
        pool, test_idx = order[:n_tr], order[n_tr:]
        vrng = np.random.default_rng(seed + VAL_SEED_OFFSET)
        vperm = vrng.permutation(len(pool))
        n_val = int(round(val_frac * len(pool)))
        val_idx, train_idx = pool[vperm[:n_val]], pool[vperm[n_val:]]
        gps_stats = None
        if "gps" in modalities:
            anchors = ds_probe.anchors
            fr = defaultdict(set)
            for i in train_idx:
                sid, t = anchors[int(i)]
                fr[sid].update(range(t - W + 1, t + 1))
            gps_stats = fit_gps_stats_frames(cfg, {k: np.fromiter(v, dtype=np.int64) for k, v in fr.items()})
        ds_all = make_snapshot_dataset(cfg, eps_all, gps_stats, allow_heldout=True)
        ds_tr = Subset(ds_all, train_idx.tolist())
        ds_va = Subset(ds_all, val_idx.tolist())
        ds_te = Subset(ds_all, test_idx.tolist())
        parts = {}

    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    info = {"protocol": protocol, "scenarios": list(exp["compare_scenarios"]),
            "n_train": len(ds_tr), "n_val": len(ds_va), "n_test": len(ds_te),
            "window": W, "val_frac": val_frac, "gps_stats": gps_stats,
            "episode_key": episode_lookup(eps_all), **parts}
    return (DataLoader(ds_tr, **tr_kw), DataLoader(ds_va, **ev_kw),
            DataLoader(ds_te, **ev_kw), info)

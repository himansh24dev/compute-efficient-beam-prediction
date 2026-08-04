"""Frozen, leakage-free, EPISODE-LEVEL train/val/test splits.

Rules (from CLAUDE.md, enforced here):
  - train/val come ONLY from scenarios 31+32 (day).
  - split is by episode (seq_index), never by sample -> no leakage.
  - val episodes are chosen deterministically from (episode_key, split_seed).
  - test = scenario 33 (night), zero-shot, never tuned on.
  - scenario 34 is NOT part of any split produced here (final-eval only).

On first call the resolved assignment is written to
`<derived_root>/splits/<version>.json` and reused verbatim thereafter, so the
protocol is frozen and versioned. Delete that file to regenerate (a new commit
should accompany any regeneration).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .config import DataConfig
from .index import Episode, build_episodes
from .paths import assert_loadable
from utils.yaml_io import load_json, save_json


@dataclass(frozen=True)
class Splits:
    version: str
    split_seed: int
    val_fraction: float
    train: list[Episode]
    val: list[Episode]
    test: list[Episode]

    def summary(self) -> dict:
        def stat(eps: list[Episode]) -> dict:
            return {
                "episodes": len(eps),
                "samples": sum(e.length for e in eps),
                "scenarios": sorted({e.scenario_id for e in eps}),
            }
        return {"train": stat(self.train), "val": stat(self.val), "test": stat(self.test)}


def single_scenario_split(cfg: DataConfig, scenario_id: int,
                          fracs: tuple[float, float, float] = (0.7, 0.15, 0.15),
                          seed: int = 1337) -> Splits:
    """Deterministic, leakage-free train/val/test split of ONE scenario's episodes.

    Used by the Phase-0 spike (Scenario 31 only). Assignment is by a stable hash
    of the episode key, so adding data never reshuffles existing episodes. This
    does NOT touch the frozen streaming-task manifest; it's a spike-local split.
    """
    assert_loadable(cfg, scenario_id, allow_heldout=False)
    ftr, fva, fte = fracs
    if abs((ftr + fva + fte) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1.0, got {fracs}")
    _, eps = build_episodes(cfg, scenario_id, allow_heldout=False)
    train, val, test = [], [], []
    for e in eps:
        r = _val_score(e.key, seed)
        (train if r < ftr else val if r < ftr + fva else test).append(e)
    for lst in (train, val, test):
        lst.sort(key=lambda e: e.key)
    splits = Splits(version=f"{cfg.version}-phase0-s{scenario_id}", split_seed=seed,
                    val_fraction=fva, train=train, val=val, test=test)
    # Disjointness (reuse the generic key check, skip the night-only assertion).
    tr = {e.key for e in train}; va = {e.key for e in val}; te = {e.key for e in test}
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), "phase0 split leakage"
    return splits


def _val_score(episode_key: str, split_seed: int) -> float:
    """Deterministic uniform-ish score in [0,1) for an episode from a stable hash.

    Using a hash (not RNG state) makes the assignment independent of iteration
    order and of how many episodes exist -> adding data doesn't reshuffle others.
    """
    h = hashlib.sha256(f"{split_seed}:{episode_key}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def _manifest_path(cfg: DataConfig):
    return cfg.derived_root / "splits" / f"{cfg.version}.json"


def resolve_splits(cfg: DataConfig, freeze: bool = True) -> Splits:
    """Compute (or load frozen) splits. Idempotent."""
    path = _manifest_path(cfg)
    if path.is_file():
        return _load_frozen(cfg, path)

    split_cfg = cfg.split
    tv_scenarios = list(split_cfg["train_val_scenarios"])
    test_scenarios = list(split_cfg["test_scenarios"])
    val_fraction = float(split_cfg["val_fraction"])
    split_seed = int(split_cfg["split_seed"])

    # Safety: never let 33/34 into train/val, never auto-load 34 anywhere.
    for sid in tv_scenarios:
        assert_loadable(cfg, sid, allow_heldout=False)
        if sid in (33, 34):
            raise ValueError(f"SAFETY: scenario {sid} must not be in train_val_scenarios")
    for sid in test_scenarios:
        assert_loadable(cfg, sid, allow_heldout=False)

    # Collect train/val episodes and split by deterministic per-episode score.
    train: list[Episode] = []
    val: list[Episode] = []
    for sid in tv_scenarios:
        _, eps = build_episodes(cfg, sid, allow_heldout=False)
        for e in eps:
            (val if _val_score(e.key, split_seed) < val_fraction else train).append(e)

    test: list[Episode] = []
    for sid in test_scenarios:
        _, eps = build_episodes(cfg, sid, allow_heldout=False)
        test.extend(eps)

    train.sort(key=lambda e: e.key)
    val.sort(key=lambda e: e.key)
    test.sort(key=lambda e: e.key)

    splits = Splits(
        version=cfg.version,
        split_seed=split_seed,
        val_fraction=val_fraction,
        train=train,
        val=val,
        test=test,
    )
    _assert_no_leakage(splits)
    if freeze:
        _write_frozen(cfg, path, splits)
    return splits


# --------------------------------------------------------------------------- #
# Freeze / load / integrity
# --------------------------------------------------------------------------- #
def _ep_to_dict(e: Episode) -> dict:
    return {
        "scenario_id": e.scenario_id,
        "episode_id": e.episode_id,
        "row_start": e.row_start,
        "row_end": e.row_end,
        "length": e.length,
    }


def _ep_from_dict(d: dict) -> Episode:
    return Episode(
        scenario_id=int(d["scenario_id"]),
        episode_id=int(d["episode_id"]),
        row_start=int(d["row_start"]),
        row_end=int(d["row_end"]),
        length=int(d["length"]),
    )


def _write_frozen(cfg: DataConfig, path, splits: Splits) -> None:
    manifest = {
        "version": splits.version,
        "split_seed": splits.split_seed,
        "val_fraction": splits.val_fraction,
        "train": [_ep_to_dict(e) for e in splits.train],
        "val": [_ep_to_dict(e) for e in splits.val],
        "test": [_ep_to_dict(e) for e in splits.test],
    }
    save_json(manifest, path)


def _load_frozen(cfg: DataConfig, path) -> Splits:
    m = load_json(path)
    if m["version"] != cfg.version:
        raise ValueError(
            f"frozen split manifest version {m['version']!r} != config version "
            f"{cfg.version!r} ({path}). Bump config version or delete the manifest."
        )
    splits = Splits(
        version=m["version"],
        split_seed=int(m["split_seed"]),
        val_fraction=float(m["val_fraction"]),
        train=[_ep_from_dict(d) for d in m["train"]],
        val=[_ep_from_dict(d) for d in m["val"]],
        test=[_ep_from_dict(d) for d in m["test"]],
    )
    _assert_no_leakage(splits)
    return splits


def _assert_no_leakage(splits: Splits) -> None:
    """Episode keys must be disjoint across train/val/test."""
    tr = {e.key for e in splits.train}
    va = {e.key for e in splits.val}
    te = {e.key for e in splits.test}
    for a_name, a, b_name, b in (
        ("train", tr, "val", va),
        ("train", tr, "test", te),
        ("val", va, "test", te),
    ):
        overlap = a & b
        if overlap:
            raise ValueError(f"LEAKAGE: {len(overlap)} episodes shared between "
                             f"{a_name} and {b_name}: {sorted(overlap)[:5]}...")
    # test must be night-only scenarios (33) here.
    bad_test = {e.scenario_id for e in splits.test} - {33}
    if bad_test:
        raise ValueError(f"unexpected scenarios in test split: {sorted(bad_test)}")

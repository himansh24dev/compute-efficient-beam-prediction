"""Fit + cache normalization statistics.

Currently: GPS standardization stats (mean/std of [dlat_m, dlon_m, speed]).
Fit on TRAIN episodes ONLY (never val/test/heldout) and cached to
`<derived_root>/stats/gps_<version>.json`. This prevents test-set leakage
through normalization -- a subtle but real failure mode.
"""
from __future__ import annotations

import numpy as np

from ..config import DataConfig
from ..index import Episode, build_sample_index
from . import gps
from utils.yaml_io import load_json, save_json


def _gps_stats_path(cfg: DataConfig):
    return cfg.derived_root / "stats" / f"gps_{cfg.version}.json"


def fit_gps_stats(cfg: DataConfig, train_episodes: list[Episode],
                  refit: bool = False) -> dict:
    """Compute (or load cached) mean/std for GPS features over train episodes."""
    path = _gps_stats_path(cfg)
    if path.is_file() and not refit:
        return load_json(path)

    cols = cfg.columns
    # Cache per-scenario sample tables so we read each CSV once.
    tables: dict[int, "object"] = {}
    feats = []
    for ep in train_episodes:
        if ep.scenario_id not in tables:
            tables[ep.scenario_id] = build_sample_index(cfg, ep.scenario_id)
        df = tables[ep.scenario_id]
        rows = df.iloc[ep.row_start:ep.row_end]
        for _, r in rows.iterrows():
            from ..paths import resolve_relative
            bs_lat, bs_lon = gps.read_latlon(resolve_relative(cfg, ep.scenario_id, r[cols["bs_loc"]]))
            ue_lat, ue_lon = gps.read_latlon(resolve_relative(cfg, ep.scenario_id, r[cols["ue_loc"]]))
            speed = r.get(cols["ue_speed"], 0.0)
            feats.append(gps.raw_features(ue_lat, ue_lon, bs_lat, bs_lon, speed))

    arr = np.asarray(feats, dtype=np.float64)
    stats = {
        "feature_names": ["dlat_m", "dlon_m", "speed_kmph"],
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "n_samples": int(arr.shape[0]),
        "n_episodes": len(train_episodes),
        "version": cfg.version,
    }
    save_json(stats, path)
    return stats


def load_gps_stats(cfg: DataConfig) -> dict:
    path = _gps_stats_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"GPS stats not fit yet ({path}). Call fit_gps_stats(cfg, splits.train) first."
        )
    return load_json(path)

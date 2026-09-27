"""Torch Datasets for streaming multimodal beam tracking.

Two views over the same episodes:

  SnapshotDataset
      Sliding windows of `window` past frames -> current beam + auxiliary future
      beams (t+1..t+H). This is the Phase-0/E1 snapshot task and the training
      view for the streaming reformulation.

  StreamingEpisodeDataset
      One item = one full episode as a time-ordered sequence (variable length).
      Used for the streaming latency/memory characterization (E2) where the
      model consumes the whole pass step by step.

torch is imported lazily so the rest of the package (and all preprocessing)
works without it. A single `FrameLoader` reads + preprocesses one frame's
enabled modalities and is shared by both datasets.

NOTE: this reference loader reads from disk per frame (no cache). For training
throughput, add a derived-tensor cache under `derived_root/` later; the raw
files are never mutated (CLAUDE.md rule).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DataConfig
from .index import Episode, build_sample_index
from .paths import resolve_relative
from .preprocess import beam as beam_pp
from .preprocess import camera as cam_pp
from .preprocess import gps as gps_pp
from .preprocess import lidar as lidar_pp
from .preprocess import radar as radar_pp


# --------------------------------------------------------------------------- #
# Frame loading / preprocessing (numpy; torch-free)
# --------------------------------------------------------------------------- #
class FrameLoader:
    """Loads + preprocesses the enabled modalities for a single sample row.

    Uses the precomputed memmap cache (data/cache.py) when available for a
    scenario, else preprocesses from the raw files. GPS is stored raw and
    standardized here, so both paths behave identically and refitting GPS stats
    never requires a cache rebuild.
    """

    def __init__(self, cfg: DataConfig, gps_stats: dict | None = None,
                 use_cache: bool | None = None):
        self.cfg = cfg
        self.cols = cfg.columns
        self.mods = cfg.modalities
        self.num_beams = int(cfg.beam["num_beams"])
        if gps_stats is not None:
            self._gps_mean = np.asarray(gps_stats["mean"], dtype=np.float32)
            self._gps_std = np.asarray(gps_stats["std"], dtype=np.float32)
        else:
            self._gps_mean = self._gps_std = None
        self._use_cache = cfg.cache.get("enabled", False) if use_cache is None else use_cache
        self._readers: dict[int, object] = {}  # lazily-opened CachedReader per scenario

    @property
    def enabled_modalities(self) -> list[str]:
        return [m for m in ("camera", "lidar", "radar", "gps") if self.mods[m]["enabled"]]

    def _reader(self, scenario_id: int):
        """Return a CachedReader for the scenario, or None if no cache exists."""
        if not self._use_cache:
            return None
        if scenario_id not in self._readers:
            from .cache import CachedReader, is_cached
            self._readers[scenario_id] = (
                CachedReader(self.cfg, scenario_id) if is_cached(self.cfg, scenario_id) else None
            )
        return self._readers[scenario_id]

    def _apply_gps_stats(self, out: dict) -> None:
        if "gps" in out and self._gps_mean is not None:
            out["gps"] = gps_pp.standardize(out["gps"], self._gps_mean, self._gps_std)

    def load_frame(self, scenario_id: int, row_pos: int, row) -> dict[str, np.ndarray]:
        """Return {modality: array, 'beam_label': int}. Cache-first, else compute.

        `row_pos` is the absolute index into build_sample_index (the cache key);
        `row` is that DataFrame row (used only on the compute path).
        """
        reader = self._reader(scenario_id)
        if reader is not None:
            out = reader.read(row_pos)          # raw gps + tensors from memmap
            self._apply_gps_stats(out)
            return out
        return self._compute_frame(scenario_id, row)

    def _compute_frame(self, scenario_id: int, row) -> dict[str, np.ndarray]:
        cols = self.cols
        out: dict[str, np.ndarray] = {}
        if self.mods["camera"]["enabled"]:
            p = resolve_relative(self.cfg, scenario_id, row[cols["rgb"]])
            out["camera"] = cam_pp.load_rgb(
                p, tuple(self.mods["camera"]["resize_hw"]),
                self.mods["camera"]["mean"], self.mods["camera"]["std"],
            )
        if self.mods["lidar"]["enabled"]:
            p = resolve_relative(self.cfg, scenario_id, row[cols["lidar"]])
            out["lidar"] = lidar_pp.load_bev(p, self.mods["lidar"])
        if self.mods["radar"]["enabled"]:
            p = resolve_relative(self.cfg, scenario_id, row[cols["radar"]])
            out["radar"] = radar_pp.load_maps(p, self.mods["radar"])
        if self.mods["gps"]["enabled"]:
            bs_lat, bs_lon = gps_pp.read_latlon(resolve_relative(self.cfg, scenario_id, row[cols["bs_loc"]]))
            ue_lat, ue_lon = gps_pp.read_latlon(resolve_relative(self.cfg, scenario_id, row[cols["ue_loc"]]))
            out["gps"] = gps_pp.raw_features(ue_lat, ue_lon, bs_lat, bs_lon, row.get(cols["ue_speed"], 0.0))
        out["beam_label"] = np.int64(int(row["beam_label"]))
        self._apply_gps_stats(out)
        return out

    def load_power(self, scenario_id: int, row) -> np.ndarray:
        p = resolve_relative(self.cfg, scenario_id, row[self.cols["pwr"]])
        return beam_pp.load_power_vector(p, self.num_beams)


# --------------------------------------------------------------------------- #
# Torch datasets (lazy import so the package works without torch)
# --------------------------------------------------------------------------- #
def _require_torch():
    try:
        import torch  # noqa: F401
        from torch.utils.data import Dataset
        return torch, Dataset
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "torch is required to instantiate Datasets (pip install torch). "
            "Preprocessing in data.preprocess.* works without torch."
        ) from e


def _to_tensor_frame(torch, frame: dict[str, np.ndarray]) -> dict:
    t = {}
    for k, v in frame.items():
        if k == "beam_label":
            t[k] = torch.as_tensor(int(v), dtype=torch.long)
        else:
            t[k] = torch.from_numpy(np.ascontiguousarray(v)).float()
    return t


@dataclass
class _EpisodeTable:
    """Caches the per-scenario sample DataFrame so we don't re-read CSVs."""
    cfg: DataConfig
    allow_heldout: bool = False
    _tables: dict = None

    def df(self, scenario_id: int):
        if self._tables is None:
            self._tables = {}
        if scenario_id not in self._tables:
            self._tables[scenario_id] = build_sample_index(
                self.cfg, scenario_id, allow_heldout=self.allow_heldout)
        return self._tables[scenario_id]


def make_snapshot_dataset(cfg: DataConfig, episodes: list[Episode], gps_stats: dict | None,
                          allow_heldout: bool = False):
    torch, Dataset = _require_torch()
    loader = FrameLoader(cfg, gps_stats)
    tables = _EpisodeTable(cfg, allow_heldout=allow_heldout)
    snap = cfg.snapshot
    window = int(snap["window"])
    horizons = list(snap["future_horizons"])
    stride = int(snap.get("stride", 1))
    max_h = max(horizons) if horizons else 0
    # `anchor_window` decouples WHICH samples exist from HOW MUCH context each
    # carries. Setting it to the largest window in a sweep keeps the sample set
    # IDENTICAL across window sizes, so an accuracy-vs-window curve is a
    # controlled comparison instead of confounding context with dataset size.
    anchor_window = int(snap.get("anchor_window", window))
    if anchor_window < window:
        raise ValueError(f"anchor_window ({anchor_window}) must be >= window ({window})")

    # Precompute valid window anchors within each episode (no cross-episode span).
    # anchor t means input frames [t-window+1 .. t], target current = t,
    # future targets = t+h. Anchors require t-anchor_window+1 >= ep_start.
    windows: list[tuple[int, int]] = []  # (scenario_id, absolute_row_of_t)
    for ep in episodes:
        first_t = ep.row_start + anchor_window - 1
        last_t = ep.row_end - 1 - max_h
        for t in range(first_t, last_t + 1, stride):
            windows.append((ep.scenario_id, t))

    class SnapshotDataset(Dataset):
        anchors = windows                     # [(scenario_id, row_of_t)], in index order

        def __len__(self):
            return len(windows)

        def __getitem__(self, i):
            scenario_id, t = windows[i]
            df = tables.df(scenario_id)
            frames = []
            for r in range(t - window + 1, t + 1):
                frames.append(_to_tensor_frame(torch, loader.load_frame(scenario_id, r, df.iloc[r])))
            # Stack per modality along a new time axis -> (T, ...).
            seq = {}
            for m in loader.enabled_modalities:
                seq[m] = torch.stack([f[m] for f in frames], dim=0)
            item = {
                "inputs": seq,                                    # dict[modality] -> (T, ...)
                "beam": frames[-1]["beam_label"],                 # current-beam label
                "future": torch.stack(
                    [torch.as_tensor(int(df.iloc[t + h]["beam_label"]), dtype=torch.long)
                     for h in horizons]
                ) if horizons else torch.empty(0, dtype=torch.long),
                "scenario_id": torch.as_tensor(scenario_id, dtype=torch.long),
                # absolute row of the anchor frame t -> maps each sample back to its
                # pass (episode) for pass-level / paired bootstrap analyses
                "row": torch.as_tensor(t, dtype=torch.long),
            }
            return item

    return SnapshotDataset()


def make_streaming_dataset(cfg: DataConfig, episodes: list[Episode], gps_stats: dict | None):
    """One item per episode = the full time-ordered sequence. Use batch_size=1
    (variable length) or a length-bucketing collate for E2 characterization."""
    torch, Dataset = _require_torch()
    loader = FrameLoader(cfg, gps_stats)
    tables = _EpisodeTable(cfg)
    eps = list(episodes)

    class StreamingEpisodeDataset(Dataset):
        def __len__(self):
            return len(eps)

        def __getitem__(self, i):
            ep = eps[i]
            df = tables.df(ep.scenario_id)
            frames = [
                _to_tensor_frame(torch, loader.load_frame(ep.scenario_id, r, df.iloc[r]))
                for r in range(ep.row_start, ep.row_end)
            ]
            seq = {}
            for m in loader.enabled_modalities:
                seq[m] = torch.stack([f[m] for f in frames], dim=0)  # (T, ...)
            beams = torch.stack([f["beam_label"] for f in frames])   # (T,)
            return {
                "inputs": seq,
                "beams": beams,
                "length": torch.as_tensor(ep.length, dtype=torch.long),
                "episode_key": ep.key,
                "scenario_id": torch.as_tensor(ep.scenario_id, dtype=torch.long),
            }

    return StreamingEpisodeDataset()

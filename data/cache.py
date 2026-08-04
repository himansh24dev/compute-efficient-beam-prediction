"""Parallel preprocessing cache — preprocess every frame ONCE, across all cores.

Each modality for a scenario is stored as a single memmapped `.npy` of shape
(N, *modality_shape), so training reads a contiguous tensor slice instead of
recomputing camera resize / LiDAR BEV / radar FFTs on every step. Building the
cache saturates the CPU: N frames are split into chunks and preprocessed by a
process pool, with each worker writing its rows directly into the memmap (no big
IPC). Raw scenario data is never touched.

Cache layout:
    <derived_root>/cache/<version>/s<sid>/
        camera.npy   (N, 3, H, W)     float32   [if enabled]
        lidar.npy    (N, C, H, W)     float32   [if enabled]
        radar.npy    (N, 2, H, W)     float32   [if enabled]
        gps.npy      (N, 3)           float32   [if enabled]   (raw, un-standardized)
        beam.npy     (N,)             int64
        meta.json    shapes / dtype / build order / done flag

`gps.npy` stores RAW features; standardization with train-only stats happens at
read time so refitting stats never requires a rebuild.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .config import DataConfig, load_data_config
from .index import build_sample_index
from .paths import resolve_relative
from utils.yaml_io import load_json, save_json

MODALITY_KEYS = ("camera", "lidar", "radar", "gps")


# --------------------------------------------------------------------------- #
# Shapes / paths
# --------------------------------------------------------------------------- #
def modality_shape(cfg: DataConfig, m: str) -> tuple[int, ...]:
    mod = cfg.modalities[m]
    if m == "camera":
        h, w = mod["resize_hw"]
        return (3, int(h), int(w))
    if m == "lidar":
        h, w = mod["grid_hw"]
        return (len(mod["channels"]), int(h), int(w))
    if m == "radar":
        h, w = mod["out_hw"]
        return (2, int(h), int(w))
    if m == "gps":
        return (len(mod["features"]),)
    raise ValueError(f"unknown modality {m!r}")


def enabled_cached_modalities(cfg: DataConfig) -> list[str]:
    return [m for m in MODALITY_KEYS if cfg.modalities[m]["enabled"]]


def scenario_cache_dir(cfg: DataConfig, scenario_id: int) -> Path:
    return cfg.derived_root / "cache" / cfg.version / f"s{scenario_id}"


def _meta_path(cfg: DataConfig, scenario_id: int) -> Path:
    return scenario_cache_dir(cfg, scenario_id) / "meta.json"


def is_cached(cfg: DataConfig, scenario_id: int) -> bool:
    mp = _meta_path(cfg, scenario_id)
    if not mp.is_file():
        return False
    try:
        meta = load_json(mp)
    except Exception:
        return False
    if not meta.get("done"):
        return False
    # every currently-enabled modality must be present with the expected shape
    for m in enabled_cached_modalities(cfg):
        if m not in meta.get("modalities", {}):
            return False
        if tuple(meta["modalities"][m]) != (meta["n_samples"], *modality_shape(cfg, m)):
            return False
    return True


# --------------------------------------------------------------------------- #
# Building (parallel)
# --------------------------------------------------------------------------- #
def _init_worker() -> None:
    """Keep each worker single-threaded so 24 procs don't oversubscribe the CPU."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


# Globals populated per-worker (cheap: built once per process, reused per chunk).
_W: dict = {}


def _worker_setup(config_path: str, scenario_id: int, mods: list[str], allow_heldout: bool):
    from . import cache as _self  # for the preprocess fns via load
    cfg = load_data_config(config_path)
    df = build_sample_index(cfg, scenario_id, allow_heldout=allow_heldout)
    cdir = scenario_cache_dir(cfg, scenario_id)
    mmaps = {m: open_memmap(cdir / f"{m}.npy", mode="r+") for m in mods}
    _W.update(cfg=cfg, df=df, mmaps=mmaps, scenario_id=scenario_id)


def _worker_chunk(args) -> tuple[int, int]:
    config_path, scenario_id, mods, allow_heldout, start, end = args
    if not _W or _W.get("scenario_id") != scenario_id:
        _init_worker()
        _worker_setup(config_path, scenario_id, mods, allow_heldout)
    cfg = _W["cfg"]; df = _W["df"]; mmaps = _W["mmaps"]
    cols = cfg.columns

    from .preprocess import camera as cam, lidar, radar, gps
    for r in range(start, end):
        row = df.iloc[r]
        if "camera" in mmaps:
            mmaps["camera"][r] = cam.load_rgb(
                resolve_relative(cfg, scenario_id, row[cols["rgb"]]),
                tuple(cfg.modalities["camera"]["resize_hw"]),
                cfg.modalities["camera"]["mean"], cfg.modalities["camera"]["std"])
        if "lidar" in mmaps:
            mmaps["lidar"][r] = lidar.load_bev(
                resolve_relative(cfg, scenario_id, row[cols["lidar"]]), cfg.modalities["lidar"])
        if "radar" in mmaps:
            mmaps["radar"][r] = radar.load_maps(
                resolve_relative(cfg, scenario_id, row[cols["radar"]]), cfg.modalities["radar"])
        if "gps" in mmaps:
            bs = gps.read_latlon(resolve_relative(cfg, scenario_id, row[cols["bs_loc"]]))
            ue = gps.read_latlon(resolve_relative(cfg, scenario_id, row[cols["ue_loc"]]))
            mmaps["gps"][r] = gps.raw_features(ue[0], ue[1], bs[0], bs[1], row.get(cols["ue_speed"], 0.0))
    for m in mmaps.values():
        m.flush()
    return start, end


def build_scenario_cache(cfg: DataConfig, scenario_id: int, workers: int | None = None,
                         allow_heldout: bool = False, force: bool = False,
                         progress: bool = True) -> Path:
    """Preprocess all frames of a scenario into memmaps, in parallel. Idempotent."""
    import multiprocessing as mp

    if is_cached(cfg, scenario_id) and not force:
        if progress:
            print(f"[cache] s{scenario_id} already built -> {scenario_cache_dir(cfg, scenario_id)}")
        return scenario_cache_dir(cfg, scenario_id)

    df = build_sample_index(cfg, scenario_id, allow_heldout=allow_heldout)
    n = len(df)
    mods = enabled_cached_modalities(cfg)
    cdir = scenario_cache_dir(cfg, scenario_id)
    cdir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate memmaps (w+ creates + zero-fills lazily/sparsely).
    dtype = np.dtype(cfg.cache.get("dtype", "float32"))
    for m in mods:
        open_memmap(cdir / f"{m}.npy", mode="w+", dtype=dtype, shape=(n, *modality_shape(cfg, m)))
    # Beam labels (small; write directly).
    np.save(cdir / "beam.npy", df["beam_label"].to_numpy(dtype=np.int64))

    # Chunking.
    if workers is None:
        cw = int(cfg.cache.get("num_workers", 0) or 0)
        workers = cw if cw > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, n))
    n_chunks = max(workers, workers * int(cfg.cache.get("chunk_per_worker", 4)))
    bounds = np.linspace(0, n, n_chunks + 1, dtype=int)
    tasks = [(str(cfg.config_path), scenario_id, mods, allow_heldout, int(bounds[i]), int(bounds[i + 1]))
             for i in range(n_chunks) if bounds[i + 1] > bounds[i]]

    if progress:
        print(f"[cache] building s{scenario_id}: {n} frames x {mods} "
              f"on {workers} workers ({len(tasks)} chunks)...")

    done = 0
    if workers == 1:
        for t in tasks:
            s, e = _worker_chunk(t); done += e - s
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
            for s, e in pool.imap_unordered(_worker_chunk, tasks):
                done += e - s
                if progress:
                    print(f"\r[cache] s{scenario_id}: {done}/{n} frames", end="", flush=True)
        if progress:
            print()

    meta = {
        "version": cfg.version,
        "scenario_id": int(scenario_id),
        "n_samples": int(n),
        "dtype": str(dtype),
        "order": "build_sample_index (episode, sort_key)",
        "modalities": {m: [n, *modality_shape(cfg, m)] for m in mods},
        "done": True,
    }
    save_json(meta, _meta_path(cfg, scenario_id))
    return cdir


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
class CachedReader:
    """Memmap reader for a scenario's cached tensors, keyed by row position
    (absolute index into build_sample_index for that scenario)."""

    def __init__(self, cfg: DataConfig, scenario_id: int):
        if not is_cached(cfg, scenario_id):
            raise FileNotFoundError(
                f"no cache for scenario {scenario_id}. Run scripts/build_cache.py first."
            )
        self.cfg = cfg
        self.scenario_id = scenario_id
        cdir = scenario_cache_dir(cfg, scenario_id)
        self.mods = enabled_cached_modalities(cfg)
        # mmap_mode='r' -> lazy, shared page cache across DataLoader workers.
        self._m = {m: np.load(cdir / f"{m}.npy", mmap_mode="r") for m in self.mods}
        self._beam = np.load(cdir / "beam.npy", mmap_mode="r")

    def has(self, modality: str) -> bool:
        return modality in self._m

    def read(self, row_pos: int) -> dict[str, np.ndarray]:
        out = {m: np.array(self._m[m][row_pos]) for m in self.mods}  # copy out of mmap
        out["beam_label"] = np.int64(int(self._beam[row_pos]))
        return out

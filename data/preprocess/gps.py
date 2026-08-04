"""GPS preprocessing.

Deployment fact: the base-station sensors are local; only GPS comes from the
vehicle (the single uplink modality). We express the vehicle position RELATIVE
to the (fixed) base station and standardize with stats fit on TRAIN episodes
only (see preprocess/stats.py) to avoid test/heldout leakage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Approx metres-per-degree at the DeepSense V2I latitude (~33.4 N). Converting
# lat/lon deltas to metres keeps features on a physically meaningful, roughly
# isotropic scale before standardization.
_M_PER_DEG_LAT = 111_000.0
_M_PER_DEG_LON = 92_800.0  # cos(33.4 deg) * 111_000


def read_latlon(path: str | Path) -> tuple[float, float]:
    """Read a 2-line GPS file (lat then lon)."""
    arr = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError(f"{path}: expected lat/lon, got {arr.shape[0]} value(s)")
    return float(arr[0]), float(arr[1])


def relative_metres(ue_lat: float, ue_lon: float,
                    bs_lat: float, bs_lon: float) -> tuple[float, float]:
    """Vehicle position relative to the base station, in metres (north, east)."""
    dlat_m = (ue_lat - bs_lat) * _M_PER_DEG_LAT
    dlon_m = (ue_lon - bs_lon) * _M_PER_DEG_LON
    return dlat_m, dlon_m


def raw_features(ue_lat: float, ue_lon: float, bs_lat: float, bs_lon: float,
                 speed_kmph: float) -> np.ndarray:
    """Un-standardized GPS feature vector [dlat_m, dlon_m, speed_kmph].

    Standardization is applied separately using cached train-only stats so the
    same raw features can be re-standardized if stats are refit.
    """
    dlat_m, dlon_m = relative_metres(ue_lat, ue_lon, bs_lat, bs_lon)
    speed = 0.0 if speed_kmph is None or np.isnan(speed_kmph) else float(speed_kmph)
    return np.array([dlat_m, dlon_m, speed], dtype=np.float32)


def standardize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((raw - mean) / np.maximum(std, 1e-6)).astype(np.float32)

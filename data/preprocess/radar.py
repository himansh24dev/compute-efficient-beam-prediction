"""Radar preprocessing: raw FMCW I/Q cube -> range-angle + range-velocity maps.

Follows the Demirhan & Alkhateeb radar-aided beam-prediction recipe: FFT the raw
cube into interpretable range-angle (RA) and range-velocity/Doppler (RV) maps,
take log-magnitude, normalize, and hand a small 2-channel image to a CNN.

Input cube shape: (antennas=4, samples_per_chirp=256, chirps=250), complex64.
Output: (2, H, W) float32 = [RA, RV], each min-max normalized to [0,1].
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_cube(path: str | Path) -> np.ndarray:
    cube = np.load(path)
    if cube.ndim != 3:
        raise ValueError(f"{path}: expected 3D radar cube, got shape {cube.shape}")
    return cube  # (A, S, C) complex


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _log_scale(mag: np.ndarray, log: bool) -> np.ndarray:
    """Log-scale an already-integrated 2D magnitude map (small array)."""
    if log:
        mag = 20.0 * np.log10(mag + 1e-6)
    return _norm01(mag)


def _range_fft(cube: np.ndarray, n_range_fft: int, range_bins_keep: int) -> np.ndarray:
    """Shared range FFT along the samples/chirp axis -> (A, R, C) complex."""
    rng = np.fft.fft(cube, n=n_range_fft, axis=1)
    return rng[:, :range_bins_keep, :]


def _ra_from_rng(rng: np.ndarray, n_angle_fft: int, log_magnitude: bool) -> np.ndarray:
    """Range-angle map via non-coherent integration over chirps, then log-scale.

    Averaging the |angle-FFT| over chirps BEFORE log-scaling keeps the expensive
    log() on the small (R, Ang) map rather than the full (Ang, R, C) cube.
    """
    ang = np.fft.fftshift(np.fft.fft(rng, n=n_angle_fft, axis=0), axes=0)  # (Ang, R, C)
    mag = np.abs(ang).mean(axis=2).T                                       # (R, Ang)
    return _log_scale(mag, log_magnitude)


def _rv_from_rng(rng: np.ndarray, n_doppler_fft: int, log_magnitude: bool,
                 remove_static: bool) -> np.ndarray:
    """Range-velocity (Doppler) map: MTI, Doppler FFT, integrate over antennas."""
    if remove_static:
        # MTI clutter removal: drop the zero-Doppler (static) component.
        rng = rng - rng.mean(axis=2, keepdims=True)
    dop = np.fft.fftshift(np.fft.fft(rng, n=n_doppler_fft, axis=2), axes=2)  # (A, R, Dop)
    mag = np.abs(dop).mean(axis=0)                                          # (R, Dop)
    return _log_scale(mag, log_magnitude)


def range_angle_map(cube: np.ndarray, n_range_fft: int, n_angle_fft: int,
                    range_bins_keep: int, log_magnitude: bool) -> np.ndarray:
    """(A,S,C) -> (range_bins_keep, n_angle_fft) range-angle map in [0,1]."""
    rng = _range_fft(cube, n_range_fft, range_bins_keep)
    return _ra_from_rng(rng, n_angle_fft, log_magnitude)


def range_velocity_map(cube: np.ndarray, n_range_fft: int, n_doppler_fft: int,
                       range_bins_keep: int, log_magnitude: bool,
                       remove_static: bool) -> np.ndarray:
    """(A,S,C) -> (range_bins_keep, n_doppler_fft) range-velocity map in [0,1]."""
    rng = _range_fft(cube, n_range_fft, range_bins_keep)
    return _rv_from_rng(rng, n_doppler_fft, log_magnitude, remove_static)


def load_maps(path: str | Path, mod_cfg: dict) -> np.ndarray:
    """Return the 2-channel (RA, RV) radar image resized to out_hw.

    Computes the shared range FFT once, then derives both maps from it.
    """
    cube = load_cube(path)
    rng = _range_fft(cube, int(mod_cfg["n_range_fft"]), int(mod_cfg["range_bins_keep"]))
    ra = _ra_from_rng(rng, int(mod_cfg["n_angle_fft"]), bool(mod_cfg["log_magnitude"]))
    rv = _rv_from_rng(rng, int(mod_cfg["n_doppler_fft"]), bool(mod_cfg["log_magnitude"]),
                      bool(mod_cfg.get("remove_static", True)))
    H, W = tuple(mod_cfg["out_hw"])
    ra = cv2.resize(ra, (W, H), interpolation=cv2.INTER_AREA)
    rv = cv2.resize(rv, (W, H), interpolation=cv2.INTER_AREA)
    return np.stack([ra, rv], axis=0).astype(np.float32)        # (2, H, W)

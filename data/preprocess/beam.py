"""Beam label + received-power vector loading.

The label is the argmax of the 64-dim power vector; the CSV already stores it
1-indexed in `unit1_beam`. We keep the full power vector because the paper's
comms metric is received-power loss vs. the genie best beam.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_power_vector(path: str | Path, num_beams: int = 64) -> np.ndarray:
    """Load the per-beam 60 GHz power vector (one value per line)."""
    v = np.loadtxt(path, dtype=np.float32).reshape(-1)
    if v.shape[0] != num_beams:
        raise ValueError(f"{path}: expected {num_beams} power values, got {v.shape[0]}")
    return v


def genie_beam(power_vector: np.ndarray) -> int:
    """0-indexed best beam = argmax of the power vector."""
    return int(np.argmax(power_vector))


def received_power_loss_db(power_vector: np.ndarray, chosen_beam: int) -> float:
    """Loss (dB) of the chosen beam vs. the genie-best beam. 0 dB = optimal.

    Used as a comms-value metric downstream; defined here so training/eval share
    one implementation.
    """
    best = float(np.max(power_vector))
    chosen = float(power_vector[chosen_beam])
    eps = 1e-12
    return float(10.0 * np.log10((best + eps) / (chosen + eps)))

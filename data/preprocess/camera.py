"""Camera preprocessing: load -> resize -> to CHW float, ImageNet-normalized.

Kept edge-friendly and ONNX-export-safe (no exotic ops): plain resize + scale +
normalize. Returns a numpy CHW float32 array; the Dataset wraps it as a tensor.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: str | Path, resize_hw: tuple[int, int],
             mean: list[float], std: list[float]) -> np.ndarray:
    """Return a normalized CHW float32 image of shape (3, H, W)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    h, w = resize_hw
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean_a = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_a = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean_a) / std_a
    return np.transpose(img, (2, 0, 1)).copy()  # HWC -> CHW

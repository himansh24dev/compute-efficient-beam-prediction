"""Deterministic seeding. Every run logs its seed (see CLAUDE.md conventions)."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed python/numpy (and torch if available). Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed

"""Parameter counting + the SSM-vs-Transformer core parity check.

CLAUDE.md: "Parameter counts between SSM and transformer cores must match within
+/-10% for any comparison." This module makes that check explicit and loggable.
"""
from __future__ import annotations

import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def core_parity(ssm_core: nn.Module, xf_core: nn.Module, tol: float = 0.10) -> dict:
    a = count_parameters(ssm_core)
    b = count_parameters(xf_core)
    rel = abs(a - b) / max(a, b)
    return {
        "ssm_core_params": a,
        "transformer_core_params": b,
        "relative_diff": rel,
        "within_tol": rel <= tol,
        "tol": tol,
    }

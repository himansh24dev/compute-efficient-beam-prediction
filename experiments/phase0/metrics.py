"""Beam-prediction metrics: Top-k accuracy and DBA (distance-based accuracy)."""
from __future__ import annotations

import torch


@torch.no_grad()
def topk_correct(logits: torch.Tensor, target: torch.Tensor, ks=(1, 3, 5)) -> dict[int, int]:
    """Count top-k hits in a batch. Returns {k: n_correct}."""
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=-1)                     # (B, maxk)
    hit = pred.eq(target.unsqueeze(-1))                     # (B, maxk)
    return {k: int(hit[:, :k].any(dim=-1).sum().item()) for k in ks}


@torch.no_grad()
def dba_correct(logits: torch.Tensor, target: torch.Tensor, thresholds=(1, 2, 3)) -> dict[int, int]:
    """DBA numerators: top-1 prediction within `t` beam indices of the target.

    DBA score (dataset-recommended) = mean over thresholds of these accuracies.
    Beams are indexed along a 1-D array, so beam-distance = |pred - target|.
    """
    pred = logits.argmax(dim=-1)                            # (B,)
    dist = (pred - target).abs()
    return {t: int((dist <= t).sum().item()) for t in thresholds}

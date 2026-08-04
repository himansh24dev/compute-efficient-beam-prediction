"""Gated multimodal fusion with train-time modality dropout.

Each modality's per-frame embedding is scored by a small gate; a softmax over
modalities produces per-modality weights and the fused embedding is their
weighted sum. Modality dropout randomly removes a modality during training
(masking its gate before the softmax) -> robustness to missing sensors and a
regularizer, exactly as specified in the architecture plan.

Inputs/outputs are time-major aware: feats[m] has shape (B, T, D); output (B, T, D).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, modalities: list[str], dim: int, modality_dropout_p: float = 0.0):
        super().__init__()
        self.modalities = list(modalities)
        self.dim = dim
        self.p = float(modality_dropout_p)
        # Per-modality scalar gate logit from its own embedding.
        self.gate = nn.ModuleDict({m: nn.Linear(dim, 1) for m in self.modalities})

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        mods = self.modalities
        stack = torch.stack([feats[m] for m in mods], dim=0)          # (M, B, T, D)
        logits = torch.stack([self.gate[m](feats[m]).squeeze(-1) for m in mods], dim=0)  # (M,B,T)

        if self.training and self.p > 0 and len(mods) > 1:
            # Drop each modality independently with prob p, but never drop all.
            keep = (torch.rand(len(mods), *logits.shape[1:], device=logits.device) >= self.p)
            all_dropped = ~keep.any(dim=0)                            # (B,T)
            if all_dropped.any():
                # Force-keep a random modality where everything was dropped.
                rand_idx = torch.randint(0, len(mods), all_dropped.shape, device=logits.device)
                onehot = torch.nn.functional.one_hot(rand_idx, len(mods)).permute(-1, *range(all_dropped.dim())).bool()
                keep = keep | (onehot & all_dropped.unsqueeze(0))
            logits = logits.masked_fill(~keep, float("-inf"))

        weights = torch.softmax(logits, dim=0).unsqueeze(-1)          # (M,B,T,1)
        fused = (weights * stack).sum(dim=0)                          # (B,T,D)
        return fused


def _dropout_mask(mods, shape, p, device):
    """Per-modality keep mask (M,B,T) with prob (1-p), never dropping all."""
    keep = (torch.rand(len(mods), *shape, device=device) >= p)
    all_dropped = ~keep.any(dim=0)
    if all_dropped.any():
        idx = torch.randint(0, len(mods), all_dropped.shape, device=device)
        onehot = torch.nn.functional.one_hot(idx, len(mods)).permute(-1, *range(all_dropped.dim())).bool()
        keep = keep | (onehot & all_dropped.unsqueeze(0))
    return keep


class MeanFusion(nn.Module):
    """Parameter-free average of per-modality embeddings (renormalised over the
    modalities kept after dropout). A fusion-mechanism ablation baseline."""

    def __init__(self, modalities: list[str], dim: int, modality_dropout_p: float = 0.0):
        super().__init__()
        self.modalities = list(modalities); self.dim = dim; self.p = float(modality_dropout_p)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        mods = self.modalities
        stack = torch.stack([feats[m] for m in mods], dim=0)          # (M,B,T,D)
        if self.training and self.p > 0 and len(mods) > 1:
            keep = _dropout_mask(mods, stack.shape[1:-1], self.p, stack.device)  # (M,B,T)
            k = keep.unsqueeze(-1).float()
            return (stack * k).sum(0) / k.sum(0).clamp_min(1.0)
        return stack.mean(0)


class ConcatFusion(nn.Module):
    """Concatenate per-modality embeddings and project back to `dim` (learned
    linear fusion). A fusion-mechanism ablation baseline."""

    def __init__(self, modalities: list[str], dim: int, modality_dropout_p: float = 0.0):
        super().__init__()
        self.modalities = list(modalities); self.dim = dim; self.p = float(modality_dropout_p)
        self.proj = nn.Linear(len(self.modalities) * dim, dim)

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        mods = self.modalities
        stack = torch.stack([feats[m] for m in mods], dim=0)          # (M,B,T,D)
        if self.training and self.p > 0 and len(mods) > 1:
            keep = _dropout_mask(mods, stack.shape[1:-1], self.p, stack.device)
            stack = stack * keep.unsqueeze(-1).float()
        cat = torch.cat([stack[i] for i in range(len(mods))], dim=-1)  # (B,T,M*D)
        return self.proj(cat)


def build_fusion(kind: str, modalities: list[str], dim: int, modality_dropout_p: float = 0.0):
    kind = (kind or "gated").lower()
    if kind == "gated":
        return GatedFusion(modalities, dim, modality_dropout_p)
    if kind == "mean":
        return MeanFusion(modalities, dim, modality_dropout_p)
    if kind == "concat":
        return ConcatFusion(modalities, dim, modality_dropout_p)
    raise ValueError(f"unknown fusion kind: {kind!r}")

"""Swappable temporal-core factory. Every core shares the (B,T,D)->(B,T,D)
causal contract so the rest of the network is identical across variants. Besides
the SSM and Transformer, we provide GRU/LSTM (classic recurrent baselines) and an
MLP (a 'no-sequence-model' / bag-of-frames baseline) so a reviewer's "why a state
space model and not a GRU / a plain MLP / a distilled student?" is answered with a
parameter-matched, same-protocol comparison."""
from __future__ import annotations

import torch.nn as nn

from .ssm import SSMCore
from .transformer import TransformerCore


class GRUCore(nn.Module):
    """Multi-layer GRU. Inherently causal (left-to-right); returns the full
    hidden-state sequence so the head can read the last frame like the others."""

    def __init__(self, dim: int, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(dim, dim, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, x):
        y, _ = self.rnn(x)
        return y


class LSTMCore(nn.Module):
    """Multi-layer LSTM (classic recurrent baseline)."""

    def __init__(self, dim: int, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(dim, dim, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)

    def forward(self, x):
        y, _ = self.rnn(x)
        return y


class MLPCore(nn.Module):
    """Per-frame residual MLP: NO temporal mixing (bag-of-frames). With last-frame
    readout this predicts the beam from the current fused feature alone — the
    honest "do you even need a sequence model at W=2?" baseline. `hidden` is sized
    so its parameter count matches the SSM/Transformer cores."""

    def __init__(self, dim: int, n_layers: int = 4, hidden: int = 456, dropout: float = 0.0):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden, dim))
            for _ in range(n_layers)])

    def forward(self, x):
        for nrm, blk in zip(self.norms, self.blocks):
            x = x + blk(nrm(x))
        return x


def build_temporal_core(kind: str, dim: int, n_layers: int, model_cfg: dict) -> nn.Module:
    kind = kind.lower()
    if kind in ("ssm", "mamba", "s6"):
        s = model_cfg.get("ssm", {})
        return SSMCore(dim, n_layers=n_layers, d_state=int(s.get("d_state", 16)),
                       d_conv=int(s.get("d_conv", 4)), expand=int(s.get("expand", 2)),
                       dt_rank=s.get("dt_rank"))
    if kind in ("transformer", "attn", "xf"):
        t = model_cfg.get("transformer", {})
        return TransformerCore(dim, n_layers=n_layers, nhead=int(t.get("nhead", 4)),
                               dim_feedforward=int(t.get("dim_feedforward", 192)),
                               dropout=float(t.get("dropout", 0.0)),
                               max_len=int(t.get("max_len", 128)))
    if kind == "gru":
        g = model_cfg.get("gru", {})
        return GRUCore(dim, n_layers=n_layers, dropout=float(g.get("dropout", 0.0)))
    if kind == "lstm":
        l = model_cfg.get("lstm", {})
        return LSTMCore(dim, n_layers=n_layers, dropout=float(l.get("dropout", 0.0)))
    if kind == "mlp":
        m = model_cfg.get("mlp", {})
        return MLPCore(dim, n_layers=n_layers, hidden=int(m.get("hidden", 456)),
                       dropout=float(m.get("dropout", 0.0)))
    raise ValueError(f"unknown temporal core kind: {kind!r}")

"""Streaming causal Transformer core with an explicit KV cache.

Same pre-norm MHA+FFN architecture as a standard causal Transformer, but exposes:
  - forward(x)                : full-sequence causal pass (parity with training)
  - step(x_t, caches, window) : one timestep, appending K,V to a growing cache
                                (window=None) or a fixed sliding window (window=W)

The KV cache is what grows O(L) with stream length -> the exact quantity the C1
claim contrasts against the SSM's fixed recurrent state. The sliding-window
variant bounds it at O(W) but discards history beyond W (the accuracy tradeoff).

Positional encoding is intentionally omitted here: this core exists to measure
the systems behaviour (KV-cache memory / per-step attention cost), which is
identical with or without positions. The Phase-2 training core adds positions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MHA(nn.Module):
    def __init__(self, dim: int, nhead: int):
        super().__init__()
        assert dim % nhead == 0
        self.nhead = nhead
        self.hd = dim // nhead
        self.dim = dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def _split(self, t, B, L):
        return t.reshape(B, L, self.nhead, self.hd).transpose(1, 2)  # (B, nh, L, hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q, B, L), self._split(k, B, L), self._split(v, B, L)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (B,nh,L,hd)
        o = o.transpose(1, 2).reshape(B, L, self.dim)
        return self.out(o)

    def step(self, x_t: torch.Tensor, cache: dict, window: int | None):
        """x_t: (B,1,D). cache: {'k':(B,nh,T,hd)|None,'v':...}. Grows/slides in place."""
        B = x_t.shape[0]
        q, k, v = self.qkv(x_t).chunk(3, dim=-1)
        q, k, v = self._split(q, B, 1), self._split(k, B, 1), self._split(v, B, 1)
        if cache["k"] is None:
            K, V = k, v
        else:
            K = torch.cat([cache["k"], k], dim=2)
            V = torch.cat([cache["v"], v], dim=2)
        if window is not None and K.shape[2] > window:
            K, V = K[:, :, -window:], V[:, :, -window:]
        cache["k"], cache["v"] = K, V
        o = F.scaled_dot_product_attention(q, K, V)                   # single query, all keys
        o = o.transpose(1, 2).reshape(B, 1, self.dim)
        return self.out(o)


class _Layer(nn.Module):
    def __init__(self, dim: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MHA(dim, nhead)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim_feedforward), nn.GELU(),
                                 nn.Linear(dim_feedforward, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def step(self, x_t, cache, window):
        x_t = x_t + self.attn.step(self.norm1(x_t), cache, window)
        x_t = x_t + self.ffn(self.norm2(x_t))
        return x_t


class StreamingTransformerCore(nn.Module):
    def __init__(self, dim: int, n_layers: int = 4, nhead: int = 4,
                 dim_feedforward: int = 200):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([_Layer(dim, nhead, dim_feedforward) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)

    # --------------------- streaming inference (KV cache) ---------------------
    def init_cache(self) -> list[dict]:
        return [{"k": None, "v": None} for _ in self.layers]

    def step(self, x_t: torch.Tensor, caches: list[dict], window: int | None = None):
        """x_t: (B, D). Returns (y (B,D), caches). window=None -> growing KV cache."""
        h = x_t.unsqueeze(1)                                          # (B,1,D)
        for layer, cache in zip(self.layers, caches):
            h = layer.step(h, cache, window)
        return self.norm_f(h).squeeze(1), caches

    @staticmethod
    def kv_bytes(caches: list[dict]) -> int:
        """Total bytes of the retained KV cache (grows O(L), or O(W) if windowed)."""
        tot = 0
        for c in caches:
            for key in ("k", "v"):
                if c[key] is not None:
                    tot += c[key].numel() * c[key].element_size()
        return tot

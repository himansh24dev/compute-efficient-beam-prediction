"""Pure-PyTorch selective state-space (S6 / Mamba) block.

This is the CUDA-kernel-free fallback the plan sanctions: it uses a plain
sequential scan over the (short) time axis, so it runs identically on CPU/GPU and
exports cleanly (no custom ops). The `mamba-ssm` kernel can later be swapped in
behind the same `SSMCore` interface for speed on long streams; for the Phase-0
snapshot task (window=8) the scan cost is negligible.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
Spaces" (2023). Selective (input-dependent) B, C, and step size delta.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        self.dim = dim
        self.d_inner = expand * dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or max(1, math.ceil(dim / 16))

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is (d_inner, d_state); stored as log for stability, negative real part.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def _selective_scan(self, u, delta, A, B, C):
        # u,delta: (b,l,d_in); A: (d_in,n); B,C: (b,l,n) -> y: (b,l,d_in)
        b, l, d_in = u.shape
        n = A.shape[1]
        dA = torch.exp(delta.unsqueeze(-1) * A)                    # (b,l,d_in,n)
        dB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b,l,d_in,n)
        h = u.new_zeros(b, d_in, n)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dB_u[:, t]                          # (b,d_in,n)
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))      # (b,d_in)
        y = torch.stack(ys, dim=1)                                 # (b,l,d_in)
        return y + u * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        xz = self.in_proj(x)                                       # (b,l,2*d_in)
        xin, z = xz.chunk(2, dim=-1)
        # Causal depthwise conv over time.
        xin = self.conv1d(xin.transpose(1, 2))[..., :l].transpose(1, 2)
        xin = F.silu(xin)
        x_dbl = self.x_proj(xin)                                   # (b,l,dt_rank+2n)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                       # (b,l,d_in)
        A = -torch.exp(self.A_log.float())                        # (d_in,n)
        y = self._selective_scan(xin, delta, A, B, C)
        y = y * F.silu(z)
        return self.out_proj(y)

    # --------------------- recurrent (streaming) inference --------------------
    # Fixed-size state = (conv_state, ssm_state); size is INDEPENDENT of stream
    # length -> O(1) memory/step. This is the core of the C1 efficiency claim.
    def init_state(self, batch: int, device, dtype=torch.float32) -> list[torch.Tensor]:
        conv_state = torch.zeros(batch, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype)
        return [conv_state, ssm_state]

    def step(self, x_t: torch.Tensor, state: list[torch.Tensor]):
        """One timestep. x_t: (B, D). Returns (y_t (B,D), new_state)."""
        conv_state, ssm_state = state
        xz = self.in_proj(x_t)                                     # (B, 2*d_in)
        xin, z = xz.chunk(2, dim=-1)                               # (B, d_in) each
        # Roll the conv buffer and insert the new input at the end.
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = xin
        w = self.conv1d.weight.squeeze(1)                          # (d_in, d_conv)
        xin = (conv_state * w).sum(-1)                             # (B, d_in)
        if self.conv1d.bias is not None:
            xin = xin + self.conv1d.bias
        xin = F.silu(xin)
        x_dbl = self.x_proj(xin)                                   # (B, dt_rank+2n)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                       # (B, d_in)
        A = -torch.exp(self.A_log.float())                        # (d_in, n)
        dA = torch.exp(delta.unsqueeze(-1) * A)                    # (B, d_in, n)
        dB_u = delta.unsqueeze(-1) * B.unsqueeze(1) * xin.unsqueeze(-1)  # (B, d_in, n)
        ssm_state = dA * ssm_state + dB_u
        y = (ssm_state * C.unsqueeze(1)).sum(-1)                   # (B, d_in)
        y = y + xin * self.D
        y = y * F.silu(z)
        return self.out_proj(y), [conv_state, ssm_state]


class SSMCore(nn.Module):
    """Stack of pre-norm residual Mamba blocks. (B,T,D) -> (B,T,D), causal."""

    def __init__(self, dim: int, n_layers: int = 4, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": RMSNorm(dim),
                "mixer": MambaBlock(dim, d_state, d_conv, expand, dt_rank),
            }) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer["mixer"](layer["norm"](x))
        return self.norm_f(x)

    # --------------------- recurrent (streaming) inference --------------------
    def init_state(self, batch: int, device, dtype=torch.float32) -> list:
        return [layer["mixer"].init_state(batch, device, dtype) for layer in self.layers]

    def step(self, x_t: torch.Tensor, states: list):
        """One timestep through the stack. x_t: (B, D). Returns (y (B,D), states)."""
        new_states = []
        for layer, st in zip(self.layers, states):
            y, ns = layer["mixer"].step(layer["norm"](x_t), st)
            x_t = x_t + y
            new_states.append(ns)
        return self.norm_f(x_t), new_states

    @staticmethod
    def state_bytes(states: list) -> int:
        """Total bytes of the retained recurrent state (constant vs stream length)."""
        return sum(t.numel() * t.element_size() for st in states for t in st)

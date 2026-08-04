"""Efficiency profile: params, measured inference throughput (FPS) + per-sample
latency, peak GPU memory, and best-effort FLOPs. Comparable to BeMamba Fig. 9 /
Table V (note: FPS is hardware-dependent -- we log the device)."""
from __future__ import annotations

import time

import torch

from utils.device import autocast_context, channels_last


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_speed(model, sample_inputs, cfg, device, iters=100, warmup=20) -> dict:
    """sample_inputs: one batch dict already on device. Returns FPS + latency."""
    model.eval()
    bsz = next(iter(sample_inputs.values())).shape[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        with autocast_context(cfg, device):
            _ = model(sample_inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with autocast_context(cfg, device):
            _ = model(sample_inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    per_batch = dt / iters
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")
    return {"fps": bsz / per_batch, "latency_ms_per_sample": per_batch / bsz * 1e3,
            "peak_mem_mb": peak_mb, "batch": bsz, "device": device.type}


def try_flops(model, sample_inputs) -> float:
    """Best-effort FLOPs via fvcore/thop; returns GFLOPs or nan if unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis
        # fvcore needs positional args; wrap the dict.
        mods = list(sample_inputs.keys())

        class _W(torch.nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m
            def forward(self, *args):
                return self.m({k: a for k, a in zip(mods, args)})
        w = _W(model)
        flops = FlopCountAnalysis(w, tuple(sample_inputs[k] for k in mods))
        flops.unsupported_ops_warnings(False); flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except Exception:
        return float("nan")

"""State-based split-inference (P2) analysis.

Two honest, measurable outputs:

1. Analytical uplink payload for each split point, and the "bytes-to-resume with
   full history at stream length L" for SSM-state (constant) vs feature-history
   (grows) vs transformer KV-cache (grows) vs a bounded feature-window. This is
   the exact quantity the state-split claim rests on; it reveals the crossover
   length beyond which transmitting the fixed SSM state beats sending features.

2. Payload-quantization accuracy: on the trained model, quantize the transmitted
   fused feature to {fp32, fp16, int8, int4} and measure Top-1/3 + DBA -> the
   bytes-vs-accuracy Pareto for the split.

Honest note surfaced by (1): per-STEP, the SSM recurrent state (~tens of KB) is
LARGER than a single fused feature (~0.5 KB). The state only wins in the
STREAMING-resume sense (it summarizes unbounded history at constant size). The
crossover length is reported explicitly.
"""
from __future__ import annotations

import math

import torch

from data.cache import modality_shape
from experiments.phase0.train import _move
from utils.device import autocast_context, channels_last
from .metrics_full import add, finalize, full_metrics, new_accumulator


def analytical_payloads(cfg, model_cfg: dict, modalities: list[str],
                        lengths=(8, 32, 128, 1000, 10000), dtype_bytes=4) -> dict:
    dim = int(model_cfg["dim"])
    nl = int(model_cfg["core_layers"])
    s = model_cfg.get("ssm", {})
    d_inner = int(s.get("expand", 2)) * dim
    d_state = int(s.get("d_state", 16))
    d_conv = int(s.get("d_conv", 4))

    per_frame_raw = sum(math.prod(modality_shape(cfg, m)) for m in modalities) * dtype_bytes
    feature_step = dim * dtype_bytes
    ssm_state = nl * (d_inner * d_conv + d_inner * d_state) * dtype_bytes   # constant
    kv_per_len = lambda L: 2 * nl * dim * L * dtype_bytes                   # transformer

    # bytes to give the server FULL history at stream length L
    resume = {}
    for L in lengths:
        resume[L] = {
            "ssm_state_const": ssm_state,
            "feature_history": L * feature_step,
            "kv_cache": kv_per_len(L),
        }
    # crossover: ssm_state == feature_history  ->  L*
    l_star = ssm_state / max(feature_step, 1)
    return {
        "per_frame_raw_bytes": per_frame_raw,
        "feature_step_bytes": feature_step,
        "ssm_state_bytes_const": ssm_state,
        "resume_bytes_by_length": resume,
        "crossover_length_state_vs_features": round(l_star, 1),
        "note": "SSM state is constant; feature-history & KV-cache grow O(L). "
                "State wins beyond ~crossover_length frames.",
    }


def _fake_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return x
    if bits == 16:
        return x.half().float()
    qmax = (1 << bits) - 1
    lo, hi = x.min(), x.max()
    scale = (hi - lo).clamp(min=1e-8) / qmax
    q = torch.round((x - lo) / scale)
    return q * scale + lo


@torch.no_grad()
def feature_quant_accuracy(model, loader, cfg, device, bits_list=(32, 16, 8, 4)) -> dict:
    """Quantize the FUSED FEATURE (the split payload) to each bit width and eval."""
    model.eval()
    chlast = channels_last(cfg)
    out = {}
    for bits in bits_list:
        acc = new_accumulator()
        for batch in loader:
            inp = _move(batch["inputs"], device, chlast)
            tgt = batch["beam"].to(device, non_blocking=True)
            with autocast_context(cfg, device):
                feats = model.encode(inp)
                fused = model.fusion(feats)                    # (B,T,D) split payload
                fused = _fake_quant(fused.float(), bits)
                h = model.core(fused)
                logits = model.head(h[:, -1])
            add(acc, full_metrics(logits.float(), tgt))
        res = finalize(acc)
        out[f"bits{bits}"] = {"top1": round(res["top1"], 4), "top3": round(res["top3"], 4),
                              "dba": round(res["dba"], 4),
                              "feature_bytes_per_step": round(model_dim(model) * bits / 8.0, 1)}
    return out


def model_dim(model) -> int:
    return model.head.in_features

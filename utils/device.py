"""GPU/runtime helpers — set once at the top of any training/eval script.

Centralizes device selection and the Ampere-era speedups (TF32, cuDNN autotune,
channels-last, AMP) so every script uses the RTX 3070 Ti consistently, plus the
DataLoader kwargs that keep 12 CPU workers feeding the GPU.
"""
from __future__ import annotations

from data.config import DataConfig


def get_device(cfg: DataConfig):
    """Resolve runtime.device ('auto'|'cuda'|'cpu') and apply Ampere speedups."""
    import torch

    want = str(cfg.runtime.get("device", "auto")).lower()
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(want)

    if device.type == "cuda":
        rt = cfg.runtime
        if rt.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
        if rt.get("matmul_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    return device


def autocast_context(cfg: DataConfig, device):
    """AMP autocast context (no-op on CPU or when runtime.amp is false)."""
    import contextlib

    import torch

    if device.type == "cuda" and cfg.runtime.get("amp", True):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def channels_last(cfg: DataConfig) -> bool:
    return bool(cfg.runtime.get("channels_last", True))


def dataloader_kwargs(cfg: DataConfig, train: bool = True) -> dict:
    """Kwargs for torch.utils.data.DataLoader tuned for this machine.

    persistent_workers/prefetch_factor only apply when num_workers > 0.
    """
    ld = cfg.loader
    nw = int(ld.get("num_workers", 0))
    kw = {
        "batch_size": int(ld.get("batch_size", 64)),
        "num_workers": nw,
        "pin_memory": bool(ld.get("pin_memory", True)),
        "shuffle": bool(train),
        "drop_last": bool(ld.get("drop_last", True)) and train,
    }
    if nw > 0:
        kw["persistent_workers"] = bool(ld.get("persistent_workers", True))
        kw["prefetch_factor"] = int(ld.get("prefetch_factor", 4))
    return kw

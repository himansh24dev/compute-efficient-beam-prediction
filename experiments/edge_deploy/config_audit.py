"""Reviewer C2/C3: honest, configuration-matched param/FLOP accounting. The paper's
27x/100x pair our W=2 cam+GPS compute with W=5 accuracy against the baseline's FULL
multimodal budget. Here we count our model at BOTH windows and the replica, so ratios
can be stated at the configuration where they are compared.

  .venv/bin/python -m experiments.edge_deploy.config_audit
"""
from __future__ import annotations

import copy

import torch

from data.config import load_data_config
from utils.yaml_io import load_yaml
from models.beam_model import BeamModel


def _params(m):
    return sum(p.numel() for p in m.parameters())


def _gflops(m, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        return FlopCountAnalysis(m, inputs).total() / 1e9
    except Exception as e:
        return None


class _Wrap(torch.nn.Module):
    """fvcore-friendly wrapper: positional (camera, gps) -> dict."""
    def __init__(self, model): super().__init__(); self.m = model
    def forward(self, camera, gps): return self.m({"camera": camera, "gps": gps})


def main():
    exp = load_yaml("configs/experiments/deploy_w2.yaml")
    mcfg = exp["model"]
    rows = []
    for W in (2, 5):
        m = BeamModel(["gps", "camera"], 64, "ssm", mcfg).eval()
        cam = torch.randn(1, W, 3, 224, 224); gps = torch.randn(1, W, 3)
        g = _gflops(_Wrap(m), (cam, gps))
        rows.append(("ours cam+GPS", W, _params(m), g))

    # BeMamba replica (full multimodal) for reference
    try:
        from experiments.edge_deploy.bemamba_replica import BeMambaReplica
        rep = BeMambaReplica(window=5).eval()
        cam = torch.randn(1, 5, 3, 224, 224); gps = torch.randn(1, 5, 3)
        lid = torch.randn(1, 5, 3, 128, 128); rad = torch.randn(1, 5, 2, 128, 128)
        try:
            from fvcore.nn import FlopCountAnalysis
            gr = FlopCountAnalysis(rep, (cam, gps, lid, rad)).total() / 1e9
        except Exception:
            gr = None
        rows.append(("BeMamba replica (full 4-mod)", 5, _params(rep), gr))
    except Exception as e:
        print("replica skipped:", e)

    print(f"\n{'config':30s} {'W':>2s} {'params':>12s} {'GFLOPs':>9s}")
    for name, W, p, g in rows:
        gs = f"{g:.3f}" if isinstance(g, float) else "n/a"
        print(f"{name:30s} {W:>2d} {p:>12,d} {gs:>9s}")

    # honest config-matched ratios vs the baseline's reported full budget (16.19M / 23.4 GFLOPs)
    BM_P, BM_G = 16.19e6, 23.4
    print("\n-- ratios vs BeMamba's reported FULL multimodal (16.19M / 23.4 GFLOPs) --")
    for name, W, p, g in rows:
        if "ours" in name:
            pr = BM_P / p
            gr = (BM_G / g) if isinstance(g, float) and g else None
            grs = f"{gr:.0f}x FLOPs" if gr else "FLOPs n/a"
            print(f"  ours cam+GPS W={W}: {pr:.0f}x params, {grs}")


if __name__ == "__main__":
    raise SystemExit(main())

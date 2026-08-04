#!/usr/bin/env python3
"""C1 / E2 streaming benchmark: per-step latency + retained-state memory vs
stream length, for the SSM (fixed state) vs Transformer (growing KV cache) vs
Transformer (sliding window), at matched parameters.

    .venv/bin/python -m experiments.c1_streaming.benchmark
    .venv/bin/python -m experiments.c1_streaming.benchmark --lengths 10 100 1000 --devices cuda

Outputs a tidy CSV + a JSON summary + a console table. First verifies the
recurrent step() is numerically equivalent to the full-sequence forward (so the
latency we measure is for a CORRECT implementation).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.ssm.s6 import SSMCore
from models.transformer.streaming import StreamingTransformerCore
from models.param_utils import count_parameters
from utils.seed import seed_everything
from utils.yaml_io import load_yaml, save_json


# --------------------------------------------------------------------------- #
# Core builders + variants
# --------------------------------------------------------------------------- #
def build_cores(cfg: dict):
    dim, nl = int(cfg["dim"]), int(cfg["n_layers"])
    s = cfg["ssm"]; t = cfg["transformer"]
    ssm = SSMCore(dim, n_layers=nl, d_state=int(s["d_state"]), d_conv=int(s["d_conv"]),
                  expand=int(s["expand"]), dt_rank=int(s["dt_rank"])).eval()
    xf = StreamingTransformerCore(dim, n_layers=nl, nhead=int(t["nhead"]),
                                  dim_feedforward=int(t["dim_feedforward"])).eval()
    return ssm, xf


# --------------------------------------------------------------------------- #
# Correctness: recurrent step() == full-sequence forward()
# --------------------------------------------------------------------------- #
@torch.no_grad()
def check_equivalence(ssm, xf, dim, device, seq=64, tol=1e-4) -> dict:
    x = torch.randn(1, seq, dim, device=device)

    # SSM
    y_full = ssm(x)
    st = ssm.init_state(1, device)
    ys = []
    for t in range(seq):
        y, st = ssm.step(x[:, t], st)
        ys.append(y)
    y_step = torch.stack(ys, dim=1)
    ssm_err = (y_full - y_step).abs().max().item()

    # Transformer (unbounded growing cache must equal full causal forward)
    yf = xf(x)
    ca = xf.init_cache()
    ys = []
    for t in range(seq):
        y, ca = xf.step(x[:, t], ca, window=None)
        ys.append(y)
    ys = torch.stack(ys, dim=1)
    xf_err = (yf - ys).abs().max().item()

    return {"ssm_max_err": ssm_err, "xf_max_err": xf_err,
            "ssm_ok": ssm_err < tol, "xf_ok": xf_err < tol, "tol": tol}


# --------------------------------------------------------------------------- #
# Timing one step at a given context length
# --------------------------------------------------------------------------- #
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure(core, kind, dim, length, device, cfg) -> dict:
    """Warm the state/cache to context `length`, then time single steps."""
    B = int(cfg["batch"])
    warm, timed = int(cfg["warmup_repeats"]), int(cfg["timed_repeats"])
    window = int(cfg["window"]) if kind == "xf_window" else None

    def new_stream(n):
        return torch.randn(n, B, dim, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Build the state/cache and warm it to (length-1) real frames.
    if kind == "ssm":
        state = core.init_state(B, device)
        stream = new_stream(max(length - 1, 0))
        for i in range(stream.shape[0]):
            _, state = core.step(stream[i], state)
        state_bytes = SSMCore.state_bytes(state)
    else:
        cache = core.init_cache()
        stream = new_stream(max(length - 1, 0))
        for i in range(stream.shape[0]):
            _, cache = core.step(stream[i], cache, window=window)
        state_bytes = StreamingTransformerCore.kv_bytes(cache)

    gpu_peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else float("nan")

    # Warmup timing (untimed).
    warm_stream = new_stream(warm + timed)
    idx = 0
    for _ in range(warm):
        if kind == "ssm":
            _, state = core.step(warm_stream[idx], state)
        else:
            _, cache = core.step(warm_stream[idx], cache, window=window)
        idx += 1
    _sync(device)

    # Timed single steps.
    lat_ms = []
    for _ in range(timed):
        x_t = warm_stream[idx]; idx += 1
        _sync(device)
        t0 = time.perf_counter()
        if kind == "ssm":
            _, state = core.step(x_t, state)
        else:
            _, cache = core.step(x_t, cache, window=window)
        _sync(device)
        lat_ms.append((time.perf_counter() - t0) * 1e3)

    lat_ms.sort()
    def pct(p):
        return lat_ms[min(len(lat_ms) - 1, int(p * len(lat_ms)))]
    return {
        "kind": kind, "device": device.type, "length": length,
        "lat_p50_ms": statistics.median(lat_ms),
        "lat_p95_ms": pct(0.95), "lat_p99_ms": pct(0.99),
        "lat_mean_ms": statistics.mean(lat_ms),
        "state_kb": state_bytes / 1024.0,
        "gpu_peak_mb": gpu_peak_mb,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/c1_streaming.yaml")
    ap.add_argument("--lengths", type=int, nargs="+", default=None)
    ap.add_argument("--devices", nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.lengths:
        cfg["lengths"] = args.lengths
    if args.devices:
        cfg["devices"] = args.devices
    seed_everything(int(cfg["seed"]))
    dim = int(cfg["dim"])

    want = cfg["devices"]
    devices = []
    for d in want:
        if d == "cuda" and not torch.cuda.is_available():
            print("[warn] cuda requested but unavailable; skipping.")
            continue
        devices.append(torch.device(d))

    ssm, xf = build_cores(cfg)
    ssm_params = count_parameters(ssm)
    xf_params = count_parameters(xf)
    rel = abs(ssm_params - xf_params) / max(ssm_params, xf_params)
    print(f"[cores] ssm={ssm_params:,}  transformer={xf_params:,}  "
          f"rel_diff={rel*100:.1f}%  (matched={'yes' if rel <= 0.10 else 'NO — tune ff'})")

    variants = [("ssm", ssm), ("xf_grow", xf), ("xf_window", xf)]
    rows, equiv = [], {}
    for device in devices:
        ssm.to(device); xf.to(device)
        equiv[device.type] = check_equivalence(ssm, xf, dim, device)
        e = equiv[device.type]
        print(f"[equiv:{device.type}] ssm_err={e['ssm_max_err']:.2e} ({'OK' if e['ssm_ok'] else 'FAIL'})  "
              f"xf_err={e['xf_max_err']:.2e} ({'OK' if e['xf_ok'] else 'FAIL'})")
        print(f"\n[{device.type}] {'kind':10s} {'L':>6s} {'p50(ms)':>9s} {'p95(ms)':>9s} "
              f"{'state(KB)':>10s} {'gpuPk(MB)':>10s}")
        for kind, core in variants:
            for L in cfg["lengths"]:
                r = measure(core, kind, dim, L, device, cfg)
                rows.append(r)
                print(f"[{device.type}] {kind:10s} {L:6d} {r['lat_p50_ms']:9.3f} "
                      f"{r['lat_p95_ms']:9.3f} {r['state_kb']:10.1f} "
                      f"{r['gpu_peak_mb'] if r['gpu_peak_mb']==r['gpu_peak_mb'] else 0:10.1f}")

    # Save CSV + JSON.
    results_dir = Path(cfg["results_dir"]); results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "streaming_curves.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {"config": cfg, "params": {"ssm": ssm_params, "transformer": xf_params,
                                         "rel_diff": rel},
               "equivalence": equiv, "n_rows": len(rows)}
    save_json(summary, results_dir / "streaming_summary.json")

    # Headline contrast: growth factor of state + latency from shortest to longest L.
    _headline(rows)
    print(f"\nsaved -> {csv_path}\n         {results_dir/'streaming_summary.json'}")
    return 0


def _headline(rows):
    Ls = sorted({r["length"] for r in rows})
    lo, hi = Ls[0], Ls[-1]
    print(f"\n=== headline: L {lo} -> {hi} ===")
    for dev in sorted({r["device"] for r in rows}):
        for kind in ("ssm", "xf_grow", "xf_window"):
            def g(L, key):
                for r in rows:
                    if r["device"] == dev and r["kind"] == kind and r["length"] == L:
                        return r[key]
                return None
            s_lo, s_hi = g(lo, "state_kb"), g(hi, "state_kb")
            l_lo, l_hi = g(lo, "lat_p50_ms"), g(hi, "lat_p50_ms")
            if s_lo is None:
                continue
            print(f"  [{dev}] {kind:10s} state {s_lo:8.1f}->{s_hi:9.1f} KB "
                  f"({s_hi/max(s_lo,1e-9):5.1f}x)   p50 {l_lo:6.3f}->{l_hi:7.3f} ms "
                  f"({l_hi/max(l_lo,1e-9):5.1f}x)")


if __name__ == "__main__":
    raise SystemExit(main())

"""Edge latency benchmark that AUTO-DETECTS each model's inputs.

Unlike benchmark_edge.py (hard-coded camera+GPS), this reads the ONNX graph's
input names/shapes and synthesises correctly-shaped random feeds, so the SAME
script benchmarks the deployed cam+GPS SSM, the multimodal BeMamba replica
(camera/gps/lidar/radar), and any scale proxy. Latency depends only on tensor
shapes, so random inputs are valid.

Runs onnxruntime + numpy only (no PyTorch) — works on a bare Raspberry Pi.

Usage on the Pi:
  python bench_auto.py --model M.onnx \
      --window 2 --threads 4 --runs 300 --sustain-sec 0 --out r.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort


def read_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_throttled():
    try:
        import subprocess
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3)
        return out.stdout.strip().split("=")[-1] if out.returncode == 0 else None
    except Exception:
        return None


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def _np_dtype(ort_type: str):
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    return np.float32


def build_feed(sess, window):
    """Resolve every input to a concrete shape and make a random tensor.

    Rule for symbolic/dynamic dims: axis 0 -> batch 1, axis 1 -> `window`,
    any other unknown axis -> 1. Concrete dims are kept as exported.
    """
    feed, resolved = {}, {}
    for inp in sess.get_inputs():
        shape = []
        for ax, d in enumerate(inp.shape):
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                shape.append(1 if ax == 0 else (window if ax == 1 else 1))
        dt = _np_dtype(inp.type)
        arr = (np.random.randn(*shape).astype(np.float32) if dt == np.float32
               else np.zeros(shape, dtype=dt))
        feed[inp.name] = arr
        resolved[inp.name] = shape
    return feed, resolved


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=2, help="fills any dynamic time axis")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sustain-sec", type=int, default=0)
    ap.add_argument("--out", default="edge_result.json")
    args = ap.parse_args()

    print(f"warm-up temp: {read_temp_c()} C | throttled: {read_throttled()}")
    sess = make_session(args.model, args.threads)
    feed, resolved = build_feed(sess, args.window)
    print(f"inputs: {resolved}")

    out = sess.run(None, feed)[0]
    print(f"model OK: output {out.shape}, argmax {int(np.asarray(out).argmax())}")

    for _ in range(args.warmup):
        sess.run(None, feed)
    t_start = read_temp_c()
    lat = np.empty(args.runs)
    for i in range(args.runs):
        t = time.perf_counter()
        sess.run(None, feed)
        lat[i] = (time.perf_counter() - t) * 1000.0
    t_end = read_temp_c()

    burst = {"p50": pct(lat, 50), "p90": pct(lat, 90), "p95": pct(lat, 95),
             "p99": pct(lat, 99), "mean": float(lat.mean()), "min": float(lat.min()),
             "hz_p50": 1000.0 / pct(lat, 50)}
    print(f"\n=== BURST (threads={args.threads}, n={args.runs}) ===")
    print(f"  p50={burst['p50']:.1f}  p90={burst['p90']:.1f}  p95={burst['p95']:.1f}  "
          f"p99={burst['p99']:.1f} ms  |  {burst['hz_p50']:.1f} Hz   (min {burst['min']:.1f})")
    print(f"  temp {t_start} -> {t_end} C")

    sustained = None
    if args.sustain_sec > 0:
        print(f"\n=== SUSTAINED {args.sustain_sec}s ===")
        samples, t0, win, nxt = [], time.perf_counter(), [], 5.0
        while time.perf_counter() - t0 < args.sustain_sec:
            t = time.perf_counter()
            sess.run(None, feed)
            win.append((time.perf_counter() - t) * 1000.0)
            el = time.perf_counter() - t0
            if el >= nxt:
                p50 = float(np.percentile(win, 50)); temp = read_temp_c()
                samples.append({"t_s": round(el, 1), "p50_ms": round(p50, 1), "temp_c": temp})
                print(f"  t={el:5.1f}s  p50={p50:5.1f}ms  temp={temp}C  throttled={read_throttled()}")
                win, nxt = [], nxt + 5.0
        if samples:
            f0, fl = samples[0]["p50_ms"], samples[-1]["p50_ms"]
            sustained = {"samples": samples, "first_p50_ms": f0, "last_p50_ms": fl,
                         "slowdown_pct": round(100.0 * (fl - f0) / f0, 1),
                         "final_throttled": read_throttled(),
                         "max_temp_c": max(s["temp_c"] for s in samples if s["temp_c"])}
            print(f"  --> {f0}->{fl} ms ({sustained['slowdown_pct']:+.1f}%), "
                  f"max {sustained['max_temp_c']}C, throttle {sustained['final_throttled']}")

    result = {"model": os.path.basename(args.model), "window": args.window,
              "threads": args.threads, "runs": args.runs, "inputs": resolved,
              "burst_ms": burst, "temp_start_c": t_start, "temp_end_c": t_end,
              "sustained": sustained}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

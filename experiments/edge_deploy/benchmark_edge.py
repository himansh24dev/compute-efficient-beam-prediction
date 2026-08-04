"""Stage 3 (edge deploy): measure the beam predictor on the target device.

Runs ONLY onnxruntime + numpy, so it works on a bare Raspberry Pi (no PyTorch).
Reports what the paper needs:
  - latency p50/p90/p95/p99 (single-inference, the real-time metric),
  - throughput (Hz),
  - CPU temperature + throttling state before/after,
  - a sustained-load loop to expose thermal throttling (burst vs sustained).

Usage on the Pi:
  python benchmark_edge.py \
      --model beam_ssm_fp32.onnx --window 8 --threads 4 \
      --runs 300 --sustain-sec 120 --out edge_result.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort


def read_temp_c() -> float | None:
    """CPU temperature in Celsius (portable sysfs path; works without vcgencmd)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def read_throttled() -> str | None:
    """Raspberry Pi throttling flags, if vcgencmd is present (0x0 = healthy)."""
    try:
        import subprocess
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3)
        return out.stdout.strip().split("=")[-1] if out.returncode == 0 else None
    except Exception:
        return None


def make_session(model_path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


def make_feed(window: int) -> dict:
    # Latency depends only on tensor shapes, not values -> random is fine.
    return {"gps": np.random.randn(1, window, 3).astype(np.float32),
            "camera": np.random.randn(1, window, 3, 224, 224).astype(np.float32)}


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=300, help="timed single inferences (burst)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sustain-sec", type=int, default=0, help="0 = skip sustained-load test")
    ap.add_argument("--out", default="edge_result.json")
    args = ap.parse_args()

    print(f"device warm-up temp: {read_temp_c()} C | throttled flags: {read_throttled()}")
    sess = make_session(args.model, args.threads)
    feed = make_feed(args.window)

    # correctness: model actually produces a 64-beam prediction
    out = sess.run(None, feed)[0]
    print(f"model OK: output shape {out.shape}, predicted beam index {int(out.argmax())}")

    # ---- burst latency ----
    for _ in range(args.warmup):
        sess.run(None, feed)
    t_start_temp = read_temp_c()
    lat = np.empty(args.runs)
    for i in range(args.runs):
        t = time.perf_counter()
        sess.run(None, feed)
        lat[i] = (time.perf_counter() - t) * 1000.0
    t_end_temp = read_temp_c()

    burst = {"p50": pct(lat, 50), "p90": pct(lat, 90), "p95": pct(lat, 95),
             "p99": pct(lat, 99), "mean": float(lat.mean()), "min": float(lat.min()),
             "hz_p50": 1000.0 / pct(lat, 50)}
    print(f"\n=== BURST latency (threads={args.threads}, W={args.window}, n={args.runs}) ===")
    print(f"  p50={burst['p50']:.1f}  p90={burst['p90']:.1f}  p95={burst['p95']:.1f}  "
          f"p99={burst['p99']:.1f} ms  |  {burst['hz_p50']:.0f} Hz")
    print(f"  temp {t_start_temp} -> {t_end_temp} C")

    # ---- sustained load (thermal throttling) ----
    sustained = None
    if args.sustain_sec > 0:
        print(f"\n=== SUSTAINED load for {args.sustain_sec}s (watching for throttling) ===")
        samples = []  # (elapsed_s, rolling_p50_ms, temp_c)
        t0 = time.perf_counter()
        win = []
        next_log = 5.0
        while time.perf_counter() - t0 < args.sustain_sec:
            t = time.perf_counter()
            sess.run(None, feed)
            win.append((time.perf_counter() - t) * 1000.0)
            el = time.perf_counter() - t0
            if el >= next_log:
                p50 = float(np.percentile(win, 50)); temp = read_temp_c()
                samples.append({"t_s": round(el, 1), "p50_ms": round(p50, 1), "temp_c": temp})
                print(f"  t={el:5.1f}s  p50={p50:5.1f}ms  temp={temp}C  throttled={read_throttled()}")
                win = []; next_log += 5.0
        if samples:
            first_p50 = samples[0]["p50_ms"]; last_p50 = samples[-1]["p50_ms"]
            slowdown = 100.0 * (last_p50 - first_p50) / first_p50
            sustained = {"samples": samples, "first_p50_ms": first_p50, "last_p50_ms": last_p50,
                         "slowdown_pct": round(slowdown, 1),
                         "final_throttled": read_throttled(), "max_temp_c": max(s["temp_c"] for s in samples if s["temp_c"])}
            print(f"  --> start p50 {first_p50}ms, end p50 {last_p50}ms "
                  f"({slowdown:+.1f}% ), max temp {sustained['max_temp_c']}C, "
                  f"final throttle flags {sustained['final_throttled']}")

    result = {"model": os.path.basename(args.model), "window": args.window, "threads": args.threads,
              "runs": args.runs, "burst_ms": burst, "temp_start_c": t_start_temp,
              "temp_end_c": t_end_temp, "sustained": sustained}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

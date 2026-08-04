"""Reviewer N10: THERMAL soak of the FULL end-to-end pipeline (not inference-only).
Runs the complete per-step sensor-to-decision loop (JPEG decode -> resize -> normalise
-> marshal W=2 window -> ONNX infer -> argmax) continuously for a fixed wall-clock
duration, sampling CPU temperature and the throttled flags throughout, and reports
latency drift (first vs last window) + peak temp + whether throttling ever asserted.

  python soak_e2e.py --model artifacts/deploy_w2/beam_ssm_fp32.onnx \
      --window 2 --threads 4 --seconds 150 --frame 960x540
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import time

import numpy as np
import onnxruntime as ort
from PIL import Image


def vc(cmd):
    try:
        return subprocess.run(["vcgencmd", cmd], capture_output=True, text=True, timeout=4).stdout.strip()
    except Exception:
        return "?"


def temp_c():
    s = vc("measure_temp")            # temp=51.0'C
    try:
        return float(s.split("=")[1].split("'")[0])
    except Exception:
        return float("nan")


def throttled():
    return vc("get_throttled")        # throttled=0x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--frame", default="960x540")
    ap.add_argument("--out", default="soak_e2e_result.json")
    args = ap.parse_args()

    fw, fh = (int(x) for x in args.frame.lower().split("x"))
    rng = np.random.default_rng(0)
    raw = (rng.random((fh, fw, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(raw).save(buf, format="JPEG", quality=85)
    jpg = buf.getvalue()
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])
    W = args.window
    win = np.zeros((1, W, 3, 224, 224), np.float32)

    # warm up
    for _ in range(20):
        img = Image.open(io.BytesIO(jpg)); img.load()
        a = (np.asarray(img.resize((224, 224), Image.BILINEAR), np.float32) / 255.0 - mean) / std
        win[:, :-1] = win[:, 1:]; win[0, -1] = a.transpose(2, 0, 1)
        sess.run(None, {"camera": win, "gps": rng.standard_normal((1, W, 3)).astype(np.float32)})

    t_start = time.perf_counter()
    deadline = t_start + args.seconds
    lat = []                                  # (elapsed_s, ms)
    samples = []                              # (elapsed_s, temp, throttled)
    next_sample = 0.0
    peak_temp = temp_c()
    t0_sample = temp_c(); thr0 = throttled()
    while time.perf_counter() < deadline:
        s0 = time.perf_counter()
        img = Image.open(io.BytesIO(jpg)); img.load()
        img = img.resize((224, 224), Image.BILINEAR)
        a = (np.asarray(img, np.float32) / 255.0 - mean) / std
        win[:, :-1] = win[:, 1:]; win[0, -1] = a.transpose(2, 0, 1)
        gps = rng.standard_normal((1, W, 3)).astype(np.float32)
        out = sess.run(None, {"camera": win, "gps": gps})[0]
        _ = int(np.asarray(out).argmax())
        s1 = time.perf_counter()
        el = s1 - t_start
        lat.append((el, (s1 - s0) * 1e3))
        if el >= next_sample:                 # sample temp/throttle ~every 5 s
            tc = temp_c(); th = throttled()
            samples.append((round(el, 1), tc, th))
            peak_temp = max(peak_temp, tc)
            next_sample = el + 5.0

    lat = np.array(lat)
    n = len(lat)
    first = lat[lat[:, 0] <= 10.0][:, 1]      # first 10 s
    last = lat[lat[:, 0] >= (args.seconds - 10.0)][:, 1]   # last 10 s
    p50_first, p50_last = float(np.percentile(first, 50)), float(np.percentile(last, 50))
    thr_end = throttled()
    ever_throttled = any(s[2] != "throttled=0x0" for s in samples) or thr_end != "throttled=0x0"

    res = {"model": args.model, "seconds": args.seconds, "threads": args.threads,
           "frame": args.frame, "iterations": n, "throughput_hz": round(n / args.seconds, 1),
           "p50_first10s_ms": round(p50_first, 2), "p50_last10s_ms": round(p50_last, 2),
           "drift_pct": round(100 * (p50_last - p50_first) / p50_first, 1),
           "temp_start_c": t0_sample, "temp_peak_c": peak_temp, "temp_end_c": samples[-1][1],
           "throttled_start": thr0, "throttled_end": thr_end, "ever_throttled": ever_throttled,
           "samples": samples}
    print(f"\n=== FULL-PIPELINE SOAK ({args.seconds:.0f}s, {args.threads} threads, {args.frame}) ===")
    print(f"iterations {n}  throughput {res['throughput_hz']} Hz")
    print(f"p50 latency: first10s {p50_first:.1f} ms -> last10s {p50_last:.1f} ms  (drift {res['drift_pct']:+.1f}%)")
    print(f"temp: start {t0_sample:.1f} -> peak {peak_temp:.1f} -> end {samples[-1][1]:.1f} C")
    print(f"throttled: start {thr0}  end {thr_end}  EVER={ever_throttled}")
    print("trajectory (s, C, throttled):")
    for s in samples:
        print(f"  {s[0]:6.1f}  {s[1]:5.1f}  {s[2]}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

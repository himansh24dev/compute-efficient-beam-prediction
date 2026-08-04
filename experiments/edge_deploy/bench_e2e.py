"""Reviewer C8: END-TO-END sensor-to-decision latency on the Pi, not just the model
forward pass. Per streaming step we decode ONE new camera frame (JPEG), resize to
224x224, normalise, marshal into the buffered W=2 window, parse GPS, run the ONNX
model, and argmax the beam. Reports per-stage and total p50/p95/p99 and peak RSS.

Runs onnxruntime + numpy + Pillow only. Usage on the Pi:
  python bench_e2e.py --model artifacts/deploy_w2/beam_ssm_fp32.onnx \
      --window 2 --threads 4 --runs 300 --frame 960x540
"""
from __future__ import annotations

import argparse
import io
import json
import resource
import time

import numpy as np
import onnxruntime as ort
from PIL import Image


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--frame", default="960x540", help="raw camera frame WxH")
    ap.add_argument("--out", default="e2e_result.json")
    args = ap.parse_args()

    fw, fh = (int(x) for x in args.frame.lower().split("x"))
    rng = np.random.default_rng(0)
    # a representative JPEG-encoded camera frame (decoded fresh each step)
    raw = (rng.random((fh, fw, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(raw).save(buf, format="JPEG", quality=85)
    jpg = buf.getvalue()
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    sess = make_session(args.model, args.threads)
    W = args.window
    win = np.zeros((1, W, 3, 224, 224), np.float32)   # buffered window

    def preprocess_one():
        img = Image.open(io.BytesIO(jpg)); img.load()          # decode
        img = img.resize((224, 224), Image.BILINEAR)           # resize
        a = (np.asarray(img, np.float32) / 255.0 - mean) / std  # normalise
        return a.transpose(2, 0, 1)                             # CHW

    st = {k: [] for k in ("decode", "resize", "normalize", "marshal", "infer", "argmax", "total")}
    for i in range(args.runs + args.warmup):
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(jpg)); img.load()
        t1 = time.perf_counter()
        img = img.resize((224, 224), Image.BILINEAR)
        t2 = time.perf_counter()
        a = (np.asarray(img, np.float32) / 255.0 - mean) / std
        a = a.transpose(2, 0, 1)
        t3 = time.perf_counter()
        win[:, :-1] = win[:, 1:]; win[0, -1] = a               # slide buffer, add new frame
        gps = rng.standard_normal((1, W, 3)).astype(np.float32)  # GPS parse (negligible)
        t4 = time.perf_counter()
        out = sess.run(None, {"camera": win, "gps": gps})[0]
        t5 = time.perf_counter()
        _ = int(np.asarray(out).argmax())
        t6 = time.perf_counter()
        if i >= args.warmup:
            st["decode"].append((t1 - t0) * 1e3); st["resize"].append((t2 - t1) * 1e3)
            st["normalize"].append((t3 - t2) * 1e3); st["marshal"].append((t4 - t3) * 1e3)
            st["infer"].append((t5 - t4) * 1e3); st["argmax"].append((t6 - t5) * 1e3)
            st["total"].append((t6 - t0) * 1e3)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # MB
    res = {"model": args.model, "window": W, "threads": args.threads, "runs": args.runs,
           "frame": args.frame, "peak_rss_mb": round(peak_rss, 1), "stages": {}}
    print(f"\n=== END-TO-END pipeline (W={W}, {args.threads} threads, {args.frame} frame, n={args.runs}) ===")
    print(f"{'stage':10s} {'p50':>7s} {'p95':>7s} {'p99':>7s}  (ms)")
    for k in ("decode", "resize", "normalize", "marshal", "infer", "argmax", "total"):
        p50, p95, p99 = pct(st[k], 50), pct(st[k], 95), pct(st[k], 99)
        res["stages"][k] = {"p50": p50, "p95": p95, "p99": p99}
        print(f"{k:10s} {p50:7.2f} {p95:7.2f} {p99:7.2f}")
    tot = res["stages"]["total"]
    print(f"\ntotal p50 {tot['p50']:.1f} ms -> {1000/tot['p50']:.1f} Hz | inference is "
          f"{100*res['stages']['infer']['p50']/tot['p50']:.0f}% of it | peak RSS {peak_rss:.0f} MB")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

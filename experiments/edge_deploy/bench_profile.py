"""Reviewer C4: per-operator profiling on the Pi, so the replica's latency can be
attributed to specific ops (conv, scan, matmul, ...) rather than asserted. Uses
onnxruntime's built-in profiler; aggregates node time by op type.

  python bench_profile.py --model M.onnx --window W --threads 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import onnxruntime as ort


def build_feed(sess, window):
    feed = {}
    for inp in sess.get_inputs():
        shape = [(1 if ax == 0 else (window if ax == 1 else (d if isinstance(d, int) and d > 0 else 1)))
                 for ax, d in enumerate(inp.shape)]
        feed[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])
    feed = build_feed(sess, args.window)
    for _ in range(3):
        sess.run(None, feed)
    for _ in range(args.runs):
        sess.run(None, feed)
    prof = sess.end_profiling()

    events = json.load(open(prof))
    by_type, total = {}, 0.0
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            t = e["dur"]; op = e.get("args", {}).get("op_name", "?")
            by_type[op] = by_type.get(op, 0.0) + t; total += t
    os.remove(prof)
    for f in glob.glob("onnxruntime_profile__*.json"):
        try: os.remove(f)
        except OSError: pass

    print(f"\n=== per-op-type kernel time: {os.path.basename(args.model)} (W={args.window}) ===")
    print(f"{'op':16s} {'ms/inf':>9s} {'%':>6s}")
    for op, t in sorted(by_type.items(), key=lambda x: -x[1]):
        ms = t / 1000.0 / args.runs
        print(f"{op:16s} {ms:9.2f} {100*t/total:6.1f}")
    print(f"{'TOTAL(kernels)':16s} {total/1000.0/args.runs:9.2f} {100.0:6.1f}")


if __name__ == "__main__":
    main()

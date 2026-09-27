"""On-device processing-pipeline benchmark (Raspberry Pi 4), revised.

Differences from experiments/edge_deploy/bench_e2e.py (kept for provenance):
  * decodes REAL DeepSense 960x540 JPEG frames (cycled from --frames-dir) instead of a
    synthetic random-noise JPEG, whose poor compressibility inflates decode time;
  * `--backend cv2` (default) decodes and resizes exactly as in training
    (cv2.imread -> BGR2RGB -> INTER_AREA 224x224 -> ImageNet normalization), so the
    on-device input matches the training input; `--backend pil` reproduces the
    original PIL/bilinear pipeline for comparison.
Timed stages per step: read (file bytes from page cache), decode, resize, normalize,
window marshal, inference, argmax. Excludes camera exposure/read-out, GPS fix, sensor
transport and synchronization, which are not simulated.

  python bench_pipeline.py --model beam_ssm_fp32.onnx --frames-dir frames --backend cv2 \
      --runs 300 --out pipeline_fp32_cv2.json
  python bench_pipeline.py ... --soak-seconds 150 --out soak_fp32_cv2.json   # thermal soak
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import resource
import subprocess
import time

import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def vc(cmd):
    try:
        return subprocess.run(["vcgencmd", cmd], capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return None


def temp_c():
    t = vc("measure_temp")
    return float(t.split("=")[1].rstrip("'C")) if t else None


def pct(x, q):
    return round(float(np.percentile(np.asarray(x), q)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--backend", choices=["cv2", "pil"], default="cv2")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--soak-seconds", type=float, default=0.0)
    ap.add_argument("--gps-only", action="store_true", help="model has only a gps input")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))
    assert files, "no frames"
    blobs = [open(f, "rb").read() for f in files]            # page-cache resident bytes
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(a.model, sess_options=so, providers=["CPUExecutionProvider"])
    W = a.window
    win = np.zeros((1, W, 3, 224, 224), np.float32)
    rng = np.random.default_rng(0)

    if a.backend == "cv2":
        import cv2
        cv2.setNumThreads(1)

        def decode(b):
            return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

        def resize(img):
            return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA)
    else:
        import io
        from PIL import Image

        def decode(b):
            im = Image.open(io.BytesIO(b)); im.load(); return im

        def resize(img):
            return np.asarray(img.resize((224, 224), Image.BILINEAR))

    stages = ("decode", "resize", "normalize", "marshal", "infer", "argmax", "total")
    st = {k: [] for k in stages}
    temps, t_start = [], time.time()
    n_total = a.runs + a.warmup
    i = 0
    while True:
        b = blobs[i % len(blobs)]
        t0 = time.perf_counter()
        if a.gps_only:
            t1 = t2 = t3 = t0
        else:
            img = decode(b); t1 = time.perf_counter()
            img = resize(img); t2 = time.perf_counter()
            x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1); t3 = time.perf_counter()
            win[:, :-1] = win[:, 1:]; win[0, -1] = x
        gps = rng.standard_normal((1, W, 3)).astype(np.float32)
        feed = {"gps": gps} if a.gps_only else {"gps": gps, "camera": win}
        t4 = time.perf_counter()
        out = sess.run(None, feed)[0]; t5 = time.perf_counter()
        _ = int(np.asarray(out).argmax()); t6 = time.perf_counter()
        if i >= a.warmup:
            for k, v in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0)):
                st[k].append(v * 1e3)
        i += 1
        el = time.time() - t_start
        if a.soak_seconds and (i % 50 == 0):
            temps.append((round(el, 1), temp_c(), vc("get_throttled")))
        if a.soak_seconds:
            if el >= a.soak_seconds:
                break
        elif i >= n_total:
            break

    res = {"model": os.path.basename(a.model), "backend": a.backend, "window": W,
           "threads": a.threads, "frames": len(files), "timed_steps": len(st["total"]),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
           "onnxruntime": ort.__version__, "platform": platform.platform(),
           "stages_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99)} for k, v in st.items() if v}}
    res["rate_hz_p50"] = round(1000.0 / res["stages_ms"]["total"]["p50"], 2)
    if a.soak_seconds:
        tot = np.asarray(st["total"]); n10 = max(1, int(len(tot) * 10 / a.soak_seconds))
        res["soak"] = {"seconds": a.soak_seconds, "p50_first10s_ms": pct(tot[:n10], 50),
                       "p50_last10s_ms": pct(tot[-n10:], 50), "samples": temps,
                       "temp_peak_c": max(t for _, t, _ in temps if t is not None) if temps and temps[0][1] else None,
                       "ever_throttled": any(th not in (None, "throttled=0x0") for _, _, th in temps)}
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res["stages_ms"], indent=1), "\nrate", res["rate_hz_p50"], "Hz ->", a.out)


if __name__ == "__main__":
    main()

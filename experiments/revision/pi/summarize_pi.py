"""Summarize the Raspberry Pi 4 revision run (results pulled into
experiments/revision/results/pi/) into one JSON table for the manuscript.

  .venv/bin/python -m experiments.revision.pi.summarize_pi
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PI_ROOT = Path("experiments/revision/results/pi")
RUN1, RUN2 = PI_ROOT / "run1", PI_ROOT / "run2"
# run 1: only the files listed as VALID in run1/README_VALIDITY.md are used
RUN1_VALID = {"fwd_deploy_fp32_t1", "fwd_deploy_fp32_t2", "fwd_deploy_fp32_t3", "fwd_deploy_fp32_t4",
              "fwd_deploy_int8_t4", "fwd_deploy_int8_selective_t4", "fwd_gps_fp32_t4", "fwd_cam_fp32_t4",
              "fwd_core_transformer_t4", "fwd_core_gru_t4", "fwd_core_lstm_t4", "fwd_core_mlp_t4",
              "pipe_deploy_fp32_cv2", "pipe_deploy_fp32_pil", "pipe_deploy_int8_cv2",
              "pipe_deploy_int8_selective_cv2", "pipe_gps_fp32"}


def src(name, ext="json"):
    if name in RUN1_VALID and (RUN1 / f"{name}.{ext}").is_file():
        return RUN1 / f"{name}.{ext}"
    return RUN2 / f"{name}.{ext}"


def fwd(name):
    d = json.load(open(src(name)))
    b = d["burst_ms"]
    return {"p50": round(b["p50"], 2), "p95": round(b["p95"], 2), "p99": round(b["p99"], 2),
            "hz": round(b["hz_p50"], 1), "runs": d["runs"],
            "temp": [round(d["temp_start_c"], 1), round(d["temp_end_c"], 1)] if d.get("temp_start_c") else None}


def pipe(name):
    d = json.load(open(src(name)))
    st = {k: v["p50"] for k, v in d["stages_ms"].items()}
    out = {"stages_p50_ms": st, "total_p50": st["total"], "total_p95": d["stages_ms"]["total"]["p95"],
           "total_p99": d["stages_ms"]["total"]["p99"], "rate_hz": d["rate_hz_p50"],
           "peak_rss_mb": d["peak_rss_mb"], "backend": d.get("backend"), "steps": d["timed_steps"]}
    if "soak" in d:
        out["soak"] = d["soak"]
    return out


def profile(name):
    txt = src(name, "txt").read_text()
    rows = {}
    for line in txt.splitlines():
        m = re.match(r"^(\S+)\s+([\d.]+)\s+([\d.]+)$", line.strip())
        if m and m.group(1) != "op":
            rows[m.group(1)] = {"ms": float(m.group(2)), "pct": float(m.group(3))}
    return rows


def main():
    res = {"env_run1": (RUN1 / "env.txt").read_text(), "env_run2": (RUN2 / "env.txt").read_text(), "source": {}}
    for n in ("fwd_deploy_fp32_t1", "fwd_deploy_fp32_t2", "fwd_deploy_fp32_t3", "fwd_deploy_fp32_t4",
              "fwd_deploy_int8_t4", "fwd_deploy_int8_selective_t4", "fwd_gps_fp32_t4", "fwd_cam_fp32_t4",
              "fwd_core_transformer_t4", "fwd_core_gru_t4", "fwd_core_lstm_t4", "fwd_core_mlp_t4",
              "replica_full_t4", "replica_half_t4", "fwd_deploy_fp32_t4_control",
              "fwd_abl_L2_t4", "fwd_abl_L6_t4", "fwd_abl_fuse_mean_t4", "fwd_abl_fuse_concat_t4",
              "fwd_abl_lidar_t4", "fwd_abl_lidar_radar_t4", "fwd_abl_pretrained_t4",
              "fwd_win_w4_t4", "fwd_win_w5_t4", "fwd_win_w8_t4", "fwd_win_w16_t4"):
        if src(n).is_file():
            res[n] = fwd(n); res["source"][n] = src(n).parent.name
    for n in ("pipe_deploy_fp32_cv2", "pipe_deploy_int8_cv2", "pipe_deploy_int8_selective_cv2",
              "pipe_deploy_fp32_pil", "pipe_gps_fp32", "soak_deploy_fp32_cv2"):
        if src(n).is_file():
            res[n] = pipe(n); res["source"][n] = src(n).parent.name
    for n in ("profile_replica_full", "profile_replica_half", "profile_deploy_fp32"):
        if src(n, "txt").is_file():
            res[n] = profile(n); res["source"][n] = src(n, "txt").parent.name
    json.dump(res, open(PI_ROOT / "pi_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("env")}, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())

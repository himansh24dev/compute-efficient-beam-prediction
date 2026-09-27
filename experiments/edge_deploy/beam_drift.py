"""Reviewer N5b: quantify beam-index STALENESS. We predict the CURRENT beam, but the
pipeline emits it ~49.5 ms later, so the truly-optimal beam may already have moved.
Using the raw DeepSense time_stamps and beam labels, we measure how far the optimal
beam index drifts over the 49.5 ms end-to-end latency, and compare to the DBA
tolerance (Delta=5) and one beam-width.

  .venv/bin/python -m experiments.edge_deploy.beam_drift
"""
from __future__ import annotations

import csv
import glob
import numpy as np


def parse_ts(s):
    # "HH:MM:SS-ffffff"
    hms, frac = s.split("-")
    h, m, sec = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(frac) / 1e6


def scenario_csv(sid):
    for p in (f"Scenario{sid}/scenario{sid}.csv", f"Scenario{sid}/scenario{sid}_dev.csv"):
        if glob.glob(p):
            return p
    return None


def main():
    import os, sys, json
    LAT = float(os.environ.get("LAT_MS", "49.5")) / 1000.0   # latency / staleness (s)
    DELTA = 5              # DBA tolerance (beam indices)
    dts, dbeam, drift, speeds = [], [], [], []
    for sid in (31, 32, 33, 34):
        path = scenario_csv(sid)
        if not path:
            continue
        rows = list(csv.DictReader(open(path)))
        # group by episode (seq_index), time-order by index
        eps = {}
        for r in rows:
            eps.setdefault(r["seq_index"], []).append(r)
        for ep in eps.values():
            ep.sort(key=lambda r: int(r["index"]))
            if len(ep) < 4:
                continue
            for a, b in zip(ep[:-1], ep[1:]):
                try:
                    dt = parse_ts(b["time_stamp"]) - parse_ts(a["time_stamp"])
                except Exception:
                    continue
                if not (0.02 < dt < 1.0):     # skip gaps / bad stamps
                    continue
                db = abs(int(b["unit1_beam"]) - int(a["unit1_beam"]))
                dts.append(dt); dbeam.append(db)
                drift.append(db * LAT / dt)   # beams moved over LAT seconds
                sp = a.get("unit1_speed") or a.get("speed") or None
    dts = np.array(dts); dbeam = np.array(dbeam); drift = np.array(drift)
    print(f"pairs={len(dts)}  frame interval dt: median {np.median(dts)*1e3:.0f} ms "
          f"(-> {1/np.median(dts):.1f} Hz)")
    print(f"per-frame |Δbeam|: mean {dbeam.mean():.2f}, median {np.median(dbeam):.0f}, "
          f"P(change)={100*(dbeam>0).mean():.0f}%, P(|Δ|>{DELTA})={100*(dbeam>DELTA).mean():.0f}%")
    print(f"\nbeam-index drift over our {LAT*1e3:.0f} ms latency "
          f"(|Δbeam| scaled by LAT/dt; locally-constant beam velocity):")
    print(f"  mean {drift.mean():.2f}, median {np.median(drift):.2f}, "
          f"p90 {np.percentile(drift,90):.2f}, p95 {np.percentile(drift,95):.2f} indices")
    print(f"  P(drift > Delta={DELTA}) = {100*(drift>DELTA).mean():.1f}%   "
          f"P(drift >= 1) = {100*(drift>=1).mean():.0f}%")
    ang = 100.0 / 64                          # ~1.6 deg per index (approx, +-50deg / 64)
    os.makedirs("experiments/revision/results", exist_ok=True)
    json.dump({"lat_ms": LAT * 1e3, "pairs": len(dts), "dt_median_ms": float(np.median(dts) * 1e3),
               "drift_median": float(np.median(drift)), "drift_mean": float(drift.mean()),
               "drift_p95": float(np.percentile(drift, 95)), "p_drift_gt_delta": float((drift > DELTA).mean())},
              open(f"experiments/revision/results/beam_drift_{LAT*1e3:.1f}ms.json", "w"), indent=1)
    print(f"  median drift {np.median(drift):.2f} indices ~ {np.median(drift)*1.6:.1f} deg "
          f"(one index ~1.6 deg; Delta=5 ~8 deg ~ one beamwidth)")


if __name__ == "__main__":
    raise SystemExit(main())

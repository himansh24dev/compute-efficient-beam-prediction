#!/usr/bin/env python3
"""Data sanity + split-integrity report. Runs with numpy/pandas/opencv only.

Usage:
    python scripts/validate_data.py [--config configs/data.yaml] [--modalities]

Reports per-scenario sample/episode counts, split summary (frozen, leakage
checked), and optionally preprocesses one frame end-to-end to confirm every
modality decodes to the expected shape. Never touches scenario 34.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.config import load_data_config
from data.index import build_episodes, validate_continuity
from data.paths import HeldoutScenarioError, resolve_relative
from data.splits import resolve_splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--modalities", action="store_true",
                    help="also preprocess one frame per modality to check shapes")
    args = ap.parse_args()

    cfg = load_data_config(args.config)
    print(f"config version: {cfg.version}")
    print(f"data_root:      {cfg.data_root}")
    print(f"derived_root:   {cfg.derived_root}\n")

    # Per-scenario counts (31/32/33 only; 34 must stay untouched).
    print("== per-scenario index ==")
    for sid in (31, 32, 33):
        df, eps = build_episodes(cfg, sid)
        rep = validate_continuity(cfg, df, eps)
        print(f"  S{sid}: {rep['n_samples']} samples, {rep['n_episodes']} episodes, "
              f"len[min={rep['episode_len_min']}, mean={rep['episode_len_mean']:.1f}, "
              f"max={rep['episode_len_max']}], dup_idx={rep['duplicate_indices']}")

    # Confirm the guard actually blocks 34.
    print("\n== heldout guard (scenario 34) ==")
    try:
        build_episodes(cfg, 34)
        print("  ERROR: scenario 34 loaded without allow_heldout! GUARD BROKEN.")
        return 2
    except HeldoutScenarioError:
        print("  OK: scenario 34 is blocked (final-eval only).")

    # Frozen, leakage-checked splits.
    print("\n== splits (frozen, leakage-checked) ==")
    splits = resolve_splits(cfg)
    for name, s in splits.summary().items():
        print(f"  {name:5s}: {s['episodes']:4d} episodes, {s['samples']:6d} samples, "
              f"scenarios={s['scenarios']}")
    print(f"  manifest: {cfg.derived_root / 'splits' / (cfg.version + '.json')}")

    if args.modalities:
        print("\n== modality decode check (first train frame) ==")
        _check_modalities(cfg, splits)

    print("\nOK")
    return 0


def _check_modalities(cfg, splits) -> None:
    import numpy as np
    from data.index import build_sample_index
    from data.preprocess import camera as cam, gps, lidar, radar, beam as beam_pp

    ep = splits.train[0]
    df = build_sample_index(cfg, ep.scenario_id)
    row = df.iloc[ep.row_start]
    cols = cfg.columns
    mods = cfg.modalities

    if mods["camera"]["enabled"]:
        a = cam.load_rgb(resolve_relative(cfg, ep.scenario_id, row[cols["rgb"]]),
                         tuple(mods["camera"]["resize_hw"]), mods["camera"]["mean"], mods["camera"]["std"])
        print(f"  camera : {a.shape} {a.dtype}  (min={a.min():.2f} max={a.max():.2f})")
    if mods["lidar"]["enabled"]:
        a = lidar.load_bev(resolve_relative(cfg, ep.scenario_id, row[cols["lidar"]]), mods["lidar"])
        print(f"  lidar  : {a.shape} {a.dtype}  (nonzero cells={int((a[1] > 0).sum())})")
    if mods["radar"]["enabled"]:
        a = radar.load_maps(resolve_relative(cfg, ep.scenario_id, row[cols["radar"]]), mods["radar"])
        print(f"  radar  : {a.shape} {a.dtype}  (RA[min={a[0].min():.2f} max={a[0].max():.2f}])")
    if mods["gps"]["enabled"]:
        bs = gps.read_latlon(resolve_relative(cfg, ep.scenario_id, row[cols["bs_loc"]]))
        ue = gps.read_latlon(resolve_relative(cfg, ep.scenario_id, row[cols["ue_loc"]]))
        raw = gps.raw_features(ue[0], ue[1], bs[0], bs[1], row.get(cols["ue_speed"], 0.0))
        print(f"  gps    : raw [dlat_m, dlon_m, speed] = {np.round(raw, 2).tolist()}")
    pw = beam_pp.load_power_vector(resolve_relative(cfg, ep.scenario_id, row[cols["pwr"]]), cfg.beam["num_beams"])
    print(f"  beam   : label={int(row['beam_label'])} genie={beam_pp.genie_beam(pw)} "
          f"(power vec len={pw.shape[0]})")


if __name__ == "__main__":
    raise SystemExit(main())

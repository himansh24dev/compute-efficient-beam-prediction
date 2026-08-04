#!/usr/bin/env python3
"""Build the preprocessing cache in parallel across all CPU cores.

    python scripts/build_cache.py --scenarios 31
    python scripts/build_cache.py --scenarios 31 32 33 --workers 24
    python scripts/build_cache.py --scenarios 31 --force

Refuses scenario 34 unless --allow-heldout is passed (final-eval only).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import build_scenario_cache, is_cached
from data.config import load_data_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--scenarios", type=int, nargs="+", default=[31])
    ap.add_argument("--workers", type=int, default=None, help="default: all cores")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("--allow-heldout", action="store_true",
                    help="permit scenario 34 (FINAL EVAL ONLY)")
    args = ap.parse_args()

    cfg = load_data_config(args.config)
    if not cfg.cache.get("enabled", False):
        print("cache.enabled is false in config; nothing to do.")
        return 0

    for sid in args.scenarios:
        if sid == 34 and not args.allow_heldout:
            print(f"REFUSING scenario 34 (heldout). Pass --allow-heldout only in final eval.")
            continue
        t0 = time.time()
        build_scenario_cache(cfg, sid, workers=args.workers,
                             allow_heldout=args.allow_heldout, force=args.force)
        dt = time.time() - t0
        ok = is_cached(cfg, sid)
        print(f"[cache] s{sid}: {'OK' if ok else 'FAILED'} in {dt:.1f}s "
              f"-> {cfg.derived_root / 'cache' / cfg.version / f's{sid}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

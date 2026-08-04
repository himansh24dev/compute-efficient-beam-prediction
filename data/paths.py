"""Filesystem path resolution for scenario data + the heldout-34 hard guard."""
from __future__ import annotations

from pathlib import Path

from .config import DataConfig


class HeldoutScenarioError(RuntimeError):
    """Raised on any attempt to touch a heldout scenario (34) without opt-in."""


def assert_loadable(cfg: DataConfig, scenario_id: int, allow_heldout: bool = False) -> None:
    """Gatekeeper for every scenario access.

    Scenario 34 is fully held out. Only code that passes allow_heldout=True
    (i.e. experiments/final_eval/) may load it. Everything else raises.
    """
    spec = cfg.scenario(scenario_id)
    if spec.heldout and not allow_heldout:
        raise HeldoutScenarioError(
            f"Scenario {scenario_id} is HELD OUT (final-eval only). "
            f"Refusing to load it. If this is the final evaluation script, pass "
            f"allow_heldout=True explicitly. See CLAUDE.md dataset rules."
        )


def scenario_dir(cfg: DataConfig, scenario_id: int, allow_heldout: bool = False) -> Path:
    assert_loadable(cfg, scenario_id, allow_heldout)
    spec = cfg.scenario(scenario_id)
    d = cfg.data_root / spec.dir
    if not d.is_dir():
        raise FileNotFoundError(f"scenario dir not found: {d}")
    return d


def scenario_csv(cfg: DataConfig, scenario_id: int, allow_heldout: bool = False) -> Path:
    d = scenario_dir(cfg, scenario_id, allow_heldout)
    csv = d / cfg.scenario(scenario_id).csv
    if not csv.is_file():
        raise FileNotFoundError(f"scenario csv not found: {csv}")
    return csv


def resolve_relative(cfg: DataConfig, scenario_id: int, rel: str) -> Path:
    """Resolve a CSV path field (e.g. './unit1/camera_data/image_173.jpg')
    against the scenario directory."""
    d = cfg.data_root / cfg.scenario(scenario_id).dir
    return (d / rel.lstrip("./")).resolve()

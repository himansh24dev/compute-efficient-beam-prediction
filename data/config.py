"""Typed access to configs/data.yaml.

We keep the parsed YAML as the source of truth (a plain dict) and wrap it in a
thin dataclass that exposes validated, typed accessors for the parts the code
depends on. Anything not modeled here is still reachable via `cfg.raw`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.yaml_io import load_yaml


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: int
    dir: str
    csv: str
    condition: str
    role: str
    heldout: bool


@dataclass(frozen=True)
class DataConfig:
    raw: dict[str, Any]
    config_path: Path

    # ---- top-level ----
    @property
    def version(self) -> str:
        return str(self.raw["version"])

    @property
    def data_root(self) -> Path:
        # data_root is resolved relative to the config file's grandparent
        # (repo root), so runs work regardless of CWD.
        root = self.raw.get("data_root", ".")
        base = self.config_path.parent.parent  # configs/ -> repo root
        return (base / root).resolve()

    @property
    def derived_root(self) -> Path:
        return (self.data_root / self.raw.get("derived_root", "data/derived")).resolve()

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 1337))

    # ---- scenarios ----
    def scenario(self, scenario_id: int) -> ScenarioSpec:
        s = self.raw["scenarios"][scenario_id] if scenario_id in self.raw["scenarios"] \
            else self.raw["scenarios"][str(scenario_id)]
        return ScenarioSpec(
            scenario_id=int(scenario_id),
            dir=s["dir"],
            csv=s["csv"],
            condition=s["condition"],
            role=s["role"],
            heldout=bool(s["heldout"]),
        )

    @property
    def scenario_ids(self) -> list[int]:
        return sorted(int(k) for k in self.raw["scenarios"].keys())

    # ---- convenient sub-dicts ----
    @property
    def columns(self) -> dict[str, str]:
        return self.raw["columns"]

    @property
    def beam(self) -> dict[str, Any]:
        return self.raw["beam"]

    @property
    def split(self) -> dict[str, Any]:
        return self.raw["split"]

    @property
    def streaming(self) -> dict[str, Any]:
        return self.raw["streaming"]

    @property
    def snapshot(self) -> dict[str, Any]:
        return self.raw["snapshot"]

    @property
    def modalities(self) -> dict[str, Any]:
        return self.raw["modalities"]

    @property
    def cache(self) -> dict[str, Any]:
        return self.raw.get("cache", {"enabled": False})

    @property
    def loader(self) -> dict[str, Any]:
        return self.raw.get("loader", {})

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw.get("runtime", {})


def load_data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    path = Path(path).resolve()
    raw = load_yaml(path)
    _validate(raw)
    return DataConfig(raw=raw, config_path=path)


def _validate(raw: dict[str, Any]) -> None:
    for key in ("version", "scenarios", "columns", "beam", "split", "modalities"):
        if key not in raw:
            raise ValueError(f"configs/data.yaml missing required top-level key: {key!r}")
    # scenario keys may be ints or strings depending on YAML; normalize check
    scn = raw["scenarios"]
    ids = {int(k) for k in scn.keys()}
    if not {31, 32, 33, 34}.issubset(ids):
        raise ValueError(f"expected scenarios 31-34 in config, got {sorted(ids)}")
    # 34 must be heldout, 33 must not be in train/val -- fail loud if config drifts.
    s34 = scn[34] if 34 in scn else scn["34"]
    if not s34.get("heldout", False):
        raise ValueError("SAFETY: scenario 34 must be heldout=true in configs/data.yaml")
    tv = set(raw["split"]["train_val_scenarios"])
    if 33 in tv or 34 in tv:
        raise ValueError("SAFETY: scenarios 33/34 must never be in train_val_scenarios")

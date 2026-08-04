"""YAML load/save with deterministic key ordering (for reproducible manifests)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    """Write JSON with sorted keys so frozen manifests are byte-stable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")

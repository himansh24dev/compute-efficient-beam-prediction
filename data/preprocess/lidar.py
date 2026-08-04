"""LiDAR preprocessing: ASCII PLY -> top-down BEV multi-channel image.

The DeepSense LiDAR files are ASCII PLY with vertex properties
(x, y, z: double; intensity: ushort). We parse them with a tiny self-contained
reader (no open3d/plyfile dependency needed for this fixed format), then project
to a fixed-size BEV grid. BEV + small CNN is the default (edge-friendly);
raw-point encoders are ablation-only per the plan.

Default channels:
  - height    : max z per cell (normalized into z_range)
  - density   : log(1+point count) per cell (optional log)
  - intensity : mean intensity per cell (normalized to [0,1])
Output shape: (C, H, W) float32.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ply_xyzi(path: str | Path) -> np.ndarray:
    """Read an ASCII PLY with (x,y,z,intensity) vertices -> (N,4) float32.

    Falls back to `plyfile` for any non-ASCII/other layout if it is installed.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header_lines = []
        fmt = None
        n_vertices = None
        prop_names: list[str] = []
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path}: unexpected EOF in PLY header")
            line = raw.decode("latin1").strip()
            header_lines.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                prop_names.append(line.split()[-1])
            elif line == "end_header":
                break
        # Body position is right after the header for ascii.
        if fmt == "ascii" and n_vertices is not None:
            data = np.loadtxt(f, dtype=np.float64, max_rows=n_vertices)
            data = np.atleast_2d(data)
            cols = {name: i for i, name in enumerate(prop_names)}
            xyzi = np.stack(
                [
                    data[:, cols["x"]],
                    data[:, cols["y"]],
                    data[:, cols["z"]],
                    data[:, cols.get("intensity", cols["z"])],
                ],
                axis=1,
            )
            return xyzi.astype(np.float32)

    # Fallback: binary or unusual property order -> use plyfile if available.
    try:
        from plyfile import PlyData
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"{path}: non-ASCII PLY and `plyfile` not installed (pip install plyfile)"
        ) from e
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    inten = v["intensity"] if "intensity" in v.dtype.names else np.zeros(len(v))
    return np.stack([v["x"], v["y"], v["z"], inten], axis=1).astype(np.float32)


def to_bev(
    points_xyzi: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    grid_hw: tuple[int, int],
    channels: list[str],
    density_log: bool = True,
) -> np.ndarray:
    """Project (N,4) xyzi points to a (C, H, W) BEV image.

    Grid rows (H) map to the y (lateral) axis, cols (W) to x (forward). Points
    outside any configured range are dropped.
    """
    H, W = grid_hw
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    x, y, z, inten = points_xyzi[:, 0], points_xyzi[:, 1], points_xyzi[:, 2], points_xyzi[:, 3]
    m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z >= z0) & (z < z1)
    x, y, z, inten = x[m], y[m], z[m], inten[m]

    # Cell indices.
    col = ((x - x0) / (x1 - x0) * W).astype(np.int64)
    row = ((y - y0) / (y1 - y0) * H).astype(np.int64)
    np.clip(col, 0, W - 1, out=col)
    np.clip(row, 0, H - 1, out=row)
    flat = row * W + col

    out = np.zeros((len(channels), H, W), dtype=np.float32)
    for ci, ch in enumerate(channels):
        plane = out[ci].reshape(-1)
        if ch == "height":
            # max z per cell, normalized into [0,1] over z_range.
            zi = (z - z0) / max(z1 - z0, 1e-6)
            np.maximum.at(plane, flat, zi.astype(np.float32))
        elif ch == "density":
            np.add.at(plane, flat, 1.0)
            if density_log:
                np.log1p(plane, out=plane)
        elif ch == "intensity":
            # mean intensity per cell = sum / count.
            counts = np.zeros(H * W, dtype=np.float32)
            np.add.at(counts, flat, 1.0)
            np.add.at(plane, flat, inten.astype(np.float32))
            nz = counts > 0
            plane[nz] /= counts[nz]
            imax = float(plane.max()) if plane.size else 0.0
            if imax > 0:
                plane /= imax
        else:
            raise ValueError(f"unknown lidar BEV channel: {ch!r}")
    return out


def load_bev(path: str | Path, mod_cfg: dict) -> np.ndarray:
    pts = read_ply_xyzi(path)
    return to_bev(
        pts,
        x_range=tuple(mod_cfg["x_range"]),
        y_range=tuple(mod_cfg["y_range"]),
        z_range=tuple(mod_cfg["z_range"]),
        grid_hw=tuple(mod_cfg["grid_hw"]),
        channels=list(mod_cfg["channels"]),
        density_log=bool(mod_cfg.get("density_log", True)),
    )

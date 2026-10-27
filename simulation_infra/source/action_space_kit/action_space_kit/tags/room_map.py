"""The real room's AprilTag map, placed in the simulation world frame.

Single source of truth: ``drone_deployment_infra/config/map.yaml`` (written by
``cal_and_map/scripts/05_map.py``) and ``config/tags.yaml`` (black-square sizes). Both are
read directly, so re-mapping the room updates the simulation with no regeneration step.

Frames:

  map (tag 10)   x right, y up, z out of the printed face, metres. ``R`` maps tag axes into
                 it (its columns are the tag's x, y, z), ``t`` is the tag centre.
  sim world      z up. The tag-10 wall is the plane X = 0 and the room lies at X > 0, so
                 a drone approaching tag 10 flies along -X. The origin is on the floor
                 directly below tag 10's centre:

                     X = z_map,  Y = x_map,  Z = y_map + tag10_height

                 (a right-handed relabelling, so rotations carry over as P @ R).

The floor is not in the map; ``tag10_height`` (centre of tag 10 above the floor) is the
one hand-measured number, configurable on the task.

Only numpy and yaml are imported, so this works without Isaac (tooling, fit scripts, tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# repo root: .../simulation_infra/source/action_space_kit/action_space_kit/tags/room_map.py
_REPO = Path(__file__).resolve().parents[5]
CONFIG_DIR = Path(os.environ.get("AS_DEPLOY_CONFIG", _REPO / "drone_deployment_infra" / "config"))

# map frame -> sim world frame (rows: world X, Y, Z in map coordinates)
P_MAP_TO_WORLD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# tag36h11 is 8 cells of black-bordered code inside a 1-cell white quiet zone
TAG36H11_CELLS = 8
TAG36H11_CELLS_WITH_BORDER = 10


@dataclass(frozen=True)
class Tag:
    tag_id: int
    size_m: float
    """Black-square edge (what the detector's tag_size means)."""
    pos: np.ndarray
    """(3,) centre in the sim world frame."""
    rot: np.ndarray
    """(3, 3) tag axes in the sim world frame; column 2 is the face normal (out of the print)."""

    @property
    def normal(self) -> np.ndarray:
        return self.rot[:, 2]

    @property
    def printed_size_m(self) -> float:
        """Edge of the printed square including the white quiet zone."""
        return self.size_m * TAG36H11_CELLS_WITH_BORDER / TAG36H11_CELLS


def load_room_tags(tag10_height: float = 1.2, map_path: Path | None = None,
                   tags_path: Path | None = None) -> dict[int, Tag]:
    """Every mapped tag with a known size, in the sim world frame."""
    map_path = Path(map_path) if map_path else CONFIG_DIR / "map.yaml"
    tags_path = Path(tags_path) if tags_path else CONFIG_DIR / "tags.yaml"
    if not map_path.is_file():
        raise FileNotFoundError(f"no room map at {map_path} (set AS_DEPLOY_CONFIG or build it with 05_map.py)")
    room = yaml.safe_load(map_path.read_text())
    sizes = {int(k): v for k, v in yaml.safe_load(tags_path.read_text())["sizes_mm"].items()}
    lift = np.array([0.0, 0.0, tag10_height])
    out = {}
    for key, item in room["tags"].items():
        tid = int(key)
        if sizes.get(tid) is None:
            continue  # unsized tags can be detected but never give a distance
        R = np.asarray(item["R"], dtype=float).reshape(3, 3)
        t = np.asarray(item["t"], dtype=float).reshape(3)
        out[tid] = Tag(tid, sizes[tid] / 1000.0, P_MAP_TO_WORLD @ t + lift, P_MAP_TO_WORLD @ R)
    if int(room["origin_tag"]) not in out:
        raise ValueError(f"origin tag {room['origin_tag']} missing from {map_path}")
    return out


def world_to_map(p_world: np.ndarray, tag10_height: float = 1.2) -> np.ndarray:
    """Sim world point(s) -> the tag-10 map frame the real drone reports (positional.py ``xyz``)."""
    p = np.asarray(p_world, dtype=float) - np.array([0.0, 0.0, tag10_height])
    return p @ P_MAP_TO_WORLD  # inverse of a permutation is its transpose


def map_to_world(p_map: np.ndarray, tag10_height: float = 1.2) -> np.ndarray:
    return np.asarray(p_map, dtype=float) @ P_MAP_TO_WORLD.T + np.array([0.0, 0.0, tag10_height])


def tag_quat_wxyz(tag: Tag) -> tuple[float, float, float, float]:
    """Orientation of a tag as a (w, x, y, z) quaternion (orthonormalized first)."""
    u, _, vt = np.linalg.svd(tag.rot)
    R = u @ vt
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    return tuple(float(v) for v in (q if q[0] >= 0 else -q))

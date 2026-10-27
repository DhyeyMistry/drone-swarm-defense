"""core/positional.py's Positioner, made robust to tag print orientation AND to a stale map.

1. Print orientation. The repo localizer (tello_calibration, IPPE) takes each tag's frame
   from the orientation of its printed code, as cv2.aruco draws it. The official
   tag36h11 images (the real prints, per drone_gui/sim_drone.py) are rotated 180 deg from
   that, which mirrors the position. Like swarm_ops/map_tags.py, this picks for every
   detection the in-plane rotation that makes the tag's y axis point UP in the image
   (tags hang upright), then applies the map. Works with either print, or a mix.

2. Stale map (tags moved since mapping -- swarm_formation's "MAP CHECK"). Each visible tag
   gives its own estimate of where the camera is. A tag whose estimate disagrees with tag
   10's (or, without tag 10, with the other tags') by more than MAP_CHECK_M is distrusted
   for the rest of the run and printed once; a tag whose map pose would put the camera
   upside down is dropped for that frame. --ignore_tags drops tags up front.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from positional import Positioner  # noqa: E402

MAP_CHECK_M = 0.5            # a tag this far from the consensus position is distrusted


def _rotz(k: int) -> np.ndarray:
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def upright_tag_to_cam(rvec, tvec) -> np.ndarray:
    """4x4 tag -> camera with the tag frame turned so its y axis points UP in the image."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))
    # camera y is down, so "tag y up in the image" = most negative (R @ Rz)[1, 1]
    k = min(range(4), key=lambda j: (R @ _rotz(j))[1, 1])
    M = np.eye(4)
    M[:3, :3] = R @ _rotz(k)
    M[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return M


class UprightPositioner(Positioner):
    """Same interface and return value as Positioner.locate(); print- and stale-map-proof."""

    def __init__(self, *args, ignore: set[int] | None = None, origin: int = 10, name: str = "", **kw) -> None:
        super().__init__(*args, **kw)
        self.ignore = set(ignore or [])
        self.distrusted: set[int] = set()
        self.origin = origin
        self.name = name

    def _distrust(self, tag: int, why: str) -> None:
        if tag not in self.distrusted:
            self.distrusted.add(tag)
            print(f"{self.name} MAP CHECK: tag {tag} {why} -> ignored for this flight (re-map it)", flush=True)

    def locate(self, frame_bgr: np.ndarray) -> dict | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        seen = []
        for obs in self.detector.detect(gray):
            tid = int(obs.tag_id)
            if not (obs.accepted and tid in self.tag_map.poses and obs.tvec is not None):
                continue
            if tid in self.ignore or tid in self.distrusted:
                continue
            m_tag_cam = upright_tag_to_cam(obs.rvec, obs.tvec)
            M = np.asarray(self.tag_map.poses[tid]) @ np.linalg.inv(m_tag_cam)   # camera -> origin
            if M[1, 1] > 0.0:                    # this tag's map pose puts the camera upside down
                continue
            d = obs.distance_m if obs.distance_m is not None else float(np.linalg.norm(M[:3, 3]))
            seen.append((d, tid, M, M[:3, 3]))
        if not seen:
            return None
        # Map check: keep only tags that agree with tag 10 (or with each other).
        ref = next((s for s in seen if s[1] == self.origin), None)
        if ref is not None:
            keep = [ref]
            for s in seen:
                if s is ref:
                    continue
                err = float(np.linalg.norm(s[3] - ref[3]))
                if err > MAP_CHECK_M:
                    self._distrust(s[1], f"is {err:.2f} m off its mapped spot (vs tag {self.origin})")
                else:
                    keep.append(s)
            seen = keep
        elif len(seen) >= 3:
            med = np.median([s[3] for s in seen], axis=0)
            seen = [s for s in seen if float(np.linalg.norm(s[3] - med)) <= MAP_CHECK_M] or seen
        seen.sort(key=lambda s: (s[1] != self.origin, s[0]))   # tag 10 first, then nearest
        xyz = np.mean([s[3] for s in seen], axis=0)
        rot = seen[0][2][:3, :3]
        to_origin = rot.T @ (-xyz)
        norm = float(np.linalg.norm(to_origin)) or 1.0
        return {
            "tags": [s[1] for s in seen],
            "xyz": [round(float(v), 3) for v in xyz],
            "range": round(float(np.linalg.norm(xyz)), 3),
            "to_origin_cam": [float(v / norm) for v in to_origin],
            "R": [[float(rot[i, j]) for j in range(3)] for i in range(3)],
        }

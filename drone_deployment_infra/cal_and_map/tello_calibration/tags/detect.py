"""AprilTag (tag36h11) detection, quality gate, and single-tag pose.

Tag frame (the OpenCV SOLVEPNP_IPPE_SQUARE / ArUco convention):
    origin at the center of the black square, x to the right, y up, z out of the printed
    face toward the viewer (right-handed), for the tag as drawn by
    cv2.aruco.generateImageMarker(DICT_APRILTAG_36h11, id). With s = black-square edge,
    TagObservation.corners[i] is the image of object point
        0: (-s/2, +s/2, 0) top-left      1: (+s/2, +s/2, 0) top-right
        2: (+s/2, -s/2, 0) bottom-right  3: (-s/2, -s/2, 0) bottom-left
Pose is tag -> camera: X_cam = R @ X_tag + t, in the OpenCV camera frame (x right,
y down, z forward), t in meters.

pupil-apriltags returns the corners in a different order: measured TR, TL, BL, BR of
that same image (counter-clockwise), for any in-plane rotation. PUPIL_TO_IPPE reorders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pupil_apriltags

from tello_calibration.utils.config import CONFIG_DIR

PUPIL_TO_IPPE = [1, 0, 3, 2]


@dataclass(frozen=True)
class TagGate:
    min_decision_margin: float
    min_side_px: float
    max_edge_width_px: float      # 10-90% rise across the outer border edge
    min_edge_contrast: float      # gray levels; flatter profiles are not measured
    max_view_angle_deg: float
    min_ambiguity_ratio: float
    ambiguity_ignore_below_deg: float


@dataclass(frozen=True)
class TagConfig:
    family: str
    origin_tag: int
    sizes_mm: dict[int, float | None]
    gate: TagGate
    detector: dict[str, Any]

    def size_m(self, tag_id: int) -> float | None:
        size = self.sizes_mm.get(tag_id)
        return None if size is None else size / 1000.0


def load_tag_config(path: Path = CONFIG_DIR / "tags.yaml") -> TagConfig:
    import yaml

    with path.open() as f:
        data = yaml.safe_load(f)
    try:
        gate = TagGate(**{k: float(data["gate"][k]) for k in TagGate.__dataclass_fields__})
        return TagConfig(
            family=str(data["family"]),
            origin_tag=int(data["origin_tag"]),
            sizes_mm={int(k): (None if v is None else float(v))
                      for k, v in (data["sizes_mm"] or {}).items()},
            gate=gate,
            detector=dict(data["detector"]),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    corners: np.ndarray            # (4, 2) raw image px, IPPE order (see module doc)
    decision_margin: float
    side_px: float
    edge_width_px: float           # median 10-90% rise across the border; inf if none
    view_angle_deg: float | None
    ambiguity_ratio: float | None  # reprojection error 2nd-best / best IPPE pose
    flip_angle_deg: float | None   # rotation between the two IPPE poses
    rvec: np.ndarray | None        # tag -> camera; scale-free, also without a size
    tvec: np.ndarray | None        # meters; None if the tag has no size in tags.yaml
    reasons: tuple[str, ...]       # why it was rejected; empty = accepted

    @property
    def accepted(self) -> bool:
        return not self.reasons

    @property
    def distance_m(self) -> float | None:
        return None if self.tvec is None else float(np.linalg.norm(self.tvec))


def object_corners(size: float) -> np.ndarray:
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def solve_ippe(corners: np.ndarray, K: np.ndarray, dist: np.ndarray, size: float
               ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Both IPPE_SQUARE poses (best first) for corners in IPPE order, distortion removed."""
    undistorted = cv2.undistortPoints(corners.reshape(-1, 1, 2).astype(np.float64), K, dist,
                                      P=K).reshape(-1, 2)
    _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        object_corners(size), undistorted, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return [r.reshape(3) for r in rvecs], [t.reshape(3) for t in tvecs], \
        [float(e) for e in np.asarray(errors).reshape(-1)]


def view_angle_deg(rvec: np.ndarray, tvec: np.ndarray) -> float:
    """Angle between the tag's normal (+z) and the direction from the tag to the camera."""
    R, _ = cv2.Rodrigues(rvec)
    to_camera = -tvec / np.linalg.norm(tvec)
    return math.degrees(math.acos(float(np.clip(R[:, 2] @ to_camera, -1.0, 1.0))))


def rotation_between_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    Ra, _ = cv2.Rodrigues(rvec_a)
    Rb, _ = cv2.Rodrigues(rvec_b)
    return math.degrees(float(np.linalg.norm(cv2.Rodrigues(Ra.T @ Rb)[0])))


EDGE_PROFILE_POSITIONS = (0.2, 0.35, 0.5, 0.65, 0.8)  # along each edge, away from corners
EDGE_SAMPLES_PER_PX = 4


def _rise_width(profile: np.ndarray, step: float, min_contrast: float) -> float | None:
    """Distance between the 10% and 90% crossings of a dark-to-bright profile, px."""
    lo, hi = float(profile.min()), float(profile.max())
    if hi - lo < min_contrast:
        return None
    crossings = []
    for level in (lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)):
        above = np.nonzero(profile >= level)[0]
        i = int(above[0])
        if i == 0:
            return None  # no dark start: profile does not straddle the edge
        frac = (level - profile[i - 1]) / (profile[i] - profile[i - 1])
        crossings.append((i - 1 + frac) * step)
    return crossings[1] - crossings[0]


def edge_width_px(gray: np.ndarray, corners: np.ndarray, min_contrast: float) -> float:
    """Median 10-90% edge rise across the tag's outer border, sampled perpendicular to
    each of the 4 edges from the black border (inside) to the white margin (outside).

    Measures blur directly in pixels, independent of tag size and of how much flat
    black/white area the tag has. Profiles reach side/10 each way, so they stay
    within the 1-cell (side/8) black border.
    """
    center = corners.mean(axis=0)
    side = float(min(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
    half = max(side / 10.0, 2.0)
    n = int(round(2 * half * EDGE_SAMPLES_PER_PX)) + 1
    offsets = np.linspace(-half, half, n)
    points = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        along = (b - a) / np.linalg.norm(b - a)
        normal = np.array([along[1], -along[0]])
        if normal @ ((a + b) / 2 - center) < 0:
            normal = -normal  # point outward
        for t in EDGE_PROFILE_POSITIONS:
            points.append(a + t * (b - a) + offsets[:, None] * normal)
    pts = np.asarray(points, np.float32)  # (profiles, samples, 2)
    profiles = cv2.remap(gray.astype(np.float32), pts[..., 0], pts[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    step = offsets[1] - offsets[0]
    widths = [w for p in profiles if (w := _rise_width(p, step, min_contrast)) is not None]
    return float(np.median(widths)) if widths else math.inf


class TagDetector:
    def __init__(self, cfg: TagConfig, K: np.ndarray, dist: np.ndarray) -> None:
        self.cfg = cfg
        self.K = np.asarray(K, np.float64)
        self.dist = np.asarray(dist, np.float64)
        self._detector = pupil_apriltags.Detector(
            families=cfg.family,
            quad_decimate=float(cfg.detector["quad_decimate"]),
            refine_edges=bool(cfg.detector["refine_edges"]),
            decode_sharpening=float(cfg.detector["decode_sharpening"]),
        )

    def detect(self, gray: np.ndarray) -> list[TagObservation]:
        return [self._observe(gray, d) for d in self._detector.detect(gray)]

    def _observe(self, gray: np.ndarray, det: Any) -> TagObservation:
        g = self.cfg.gate
        corners = np.asarray(det.corners, np.float64)[PUPIL_TO_IPPE]
        side = float(min(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
        edge = edge_width_px(gray, corners, g.min_edge_contrast)
        margin = float(det.decision_margin)
        size = self.cfg.size_m(det.tag_id)

        rvecs, tvecs, errors = solve_ippe(corners, self.K, self.dist, size or 1.0)
        angle = view_angle_deg(rvecs[0], tvecs[0])
        ratio = errors[1] / max(errors[0], 1e-9) if len(errors) > 1 else math.inf
        flip = rotation_between_deg(rvecs[0], rvecs[1]) if len(rvecs) > 1 else 0.0

        reasons = []
        if margin < g.min_decision_margin:
            reasons.append(f"margin {margin:.0f}<{g.min_decision_margin:.0f}")
        if side < g.min_side_px:
            reasons.append(f"small {side:.0f}px<{g.min_side_px:.0f}")
        if edge > g.max_edge_width_px:
            reasons.append(f"blurry edge {edge:.1f}px>{g.max_edge_width_px:.1f}")
        if angle > g.max_view_angle_deg:
            reasons.append(f"angle {angle:.0f}>{g.max_view_angle_deg:.0f}")
        if ratio < g.min_ambiguity_ratio and flip > g.ambiguity_ignore_below_deg:
            reasons.append(f"ambiguous r={ratio:.1f}<{g.min_ambiguity_ratio:.1f}")
        if size is None:
            reasons.append("no size in tags.yaml")

        return TagObservation(
            tag_id=int(det.tag_id), corners=corners, decision_margin=margin, side_px=side,
            edge_width_px=edge, view_angle_deg=angle, ambiguity_ratio=ratio, flip_angle_deg=flip,
            rvec=rvecs[0], tvec=tvecs[0] if size is not None else None,
            reasons=tuple(reasons),
        )

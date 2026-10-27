"""Decide whether a live frame is worth saving for calibration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from tello_calibration.calib.board import Detection, object_points


@dataclass(frozen=True)
class GateConfig:
    min_corner_fraction: float = 0.6  # of all inner corners
    min_sharpness: float = 150.0      # Laplacian variance inside the board's bounding box
    min_interval_s: float = 0.5
    # A view is a near-duplicate of a saved one only if it is close in ALL of these:
    min_center_shift: float = 0.08    # board center moved, fraction of image size
    min_size_change: float = 0.15     # apparent size, relative
    min_tilt_change: float = 0.10     # log ratio of opposite edge lengths

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ViewDescriptor:
    cx: float      # board center, fraction of image width
    cy: float      # board center, fraction of image height
    size: float    # sqrt(board quad area / image area)
    tilt_x: float  # log(left edge / right edge): turned left/right
    tilt_y: float  # log(top edge / bottom edge): tipped forward/back


@dataclass(frozen=True)
class Decision:
    accept: bool
    reason: str
    sharpness: float | None = None
    view: ViewDescriptor | None = None


def sharpness(gray: np.ndarray, detection: Detection, pad: int = 10) -> float:
    """Laplacian variance inside the bounding box of the detected corners."""
    x0, y0 = np.floor(detection.corners.min(axis=0)).astype(int) - pad
    x1, y1 = np.ceil(detection.corners.max(axis=0)).astype(int) + pad
    h, w = gray.shape[:2]
    roi = gray[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def describe_view(board: cv2.aruco.CharucoBoard, detection: Detection,
                  image_size: tuple[int, int]) -> ViewDescriptor:
    """Where the full inner-corner rectangle lands in the image (via a homography)."""
    obj = object_points(board, detection)[:, :2]
    H, _ = cv2.findHomography(obj, detection.corners, 0)
    all_obj = np.asarray(board.getChessboardCorners(), np.float32)[:, :2]
    (x0, y0), (x1, y1) = all_obj.min(axis=0), all_obj.max(axis=0)
    quad_obj = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], np.float32)
    tl, tr, br, bl = cv2.perspectiveTransform(quad_obj, H).reshape(4, 2)
    center = cv2.perspectiveTransform(
        np.array([[[(x0 + x1) / 2, (y0 + y1) / 2]]], np.float32), H).reshape(2)

    width, height = image_size
    area = abs(cv2.contourArea(np.array([tl, tr, br, bl], np.float32)))

    def length(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    return ViewDescriptor(
        cx=float(center[0]) / width,
        cy=float(center[1]) / height,
        size=math.sqrt(area / (width * height)),
        tilt_x=math.log(length(tl, bl) / length(tr, br)),
        tilt_y=math.log(length(tl, tr) / length(bl, br)),
    )


def is_near_duplicate(a: ViewDescriptor, b: ViewDescriptor, cfg: GateConfig) -> bool:
    return (math.hypot(a.cx - b.cx, a.cy - b.cy) < cfg.min_center_shift
            and abs(a.size / b.size - 1.0) < cfg.min_size_change
            and abs(a.tilt_x - b.tilt_x) < cfg.min_tilt_change
            and abs(a.tilt_y - b.tilt_y) < cfg.min_tilt_change)


class CaptureGate:
    """Holds the views saved so far; evaluate() says whether a new frame adds one."""

    def __init__(self, board: cv2.aruco.CharucoBoard, n_corners: int,
                 image_size: tuple[int, int], cfg: GateConfig = GateConfig()) -> None:
        self.board = board
        self.min_corners = math.ceil(cfg.min_corner_fraction * n_corners)
        self.n_corners = n_corners
        self.image_size = image_size
        self.cfg = cfg
        self.saved: list[ViewDescriptor] = []
        self.last_t: float | None = None

    def evaluate(self, detection: Detection | None, gray: np.ndarray, t: float) -> Decision:
        if self.last_t is not None and t - self.last_t < self.cfg.min_interval_s:
            return Decision(False, "too soon after last capture")
        found = 0 if detection is None else len(detection)
        if detection is None or found < self.min_corners:
            return Decision(False, f"only {found}/{self.n_corners} corners "
                                   f"(need {self.min_corners})")
        sharp = sharpness(gray, detection)
        if sharp < self.cfg.min_sharpness:
            return Decision(False, f"blurry ({sharp:.0f} < {self.cfg.min_sharpness:.0f})", sharp)
        view = describe_view(self.board, detection, self.image_size)
        for i, saved in enumerate(self.saved):
            if is_near_duplicate(view, saved, self.cfg):
                return Decision(False, f"too similar to frame {i:03d}; move or tilt more",
                                sharp, view)
        return Decision(True, "ok", sharp, view)

    def commit(self, decision: Decision, t: float) -> None:
        """Record an accepted frame (call after it was saved)."""
        assert decision.accept and decision.view is not None
        self.saved.append(decision.view)
        self.last_t = t

"""ChArUco board definition, rendering, and detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICTIONARIES = {"DICT_5X5_100": cv2.aruco.DICT_5X5_100}


@dataclass(frozen=True)
class BoardSpec:
    """Everything needed to rebuild the exact board that was displayed."""

    dictionary: str
    squares_x: int
    squares_y: int
    square_px: int
    marker_px: int
    image_width: int   # size of the PNG (= screen it is shown on)
    image_height: int
    legacy_pattern: bool = False

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def marker_mm(self, square_mm: float) -> float:
        return square_mm * self.marker_px / self.square_px


@dataclass(frozen=True)
class Detection:
    """ChArUco corners found in one image: ids (N,) int32 and pixels (N, 2) float32."""

    ids: np.ndarray
    corners: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


def spec_for_screen(width: int, height: int, squares_x: int = 7, squares_y: int = 5,
                    marker_ratio: float = 0.7, margin_px: int = 40,
                    dictionary: str = "DICT_5X5_100") -> BoardSpec:
    """Largest whole-pixel squares that fit the screen with a white margin."""
    square = min((width - 2 * margin_px) // squares_x, (height - 2 * margin_px) // squares_y)
    return BoardSpec(dictionary, squares_x, squares_y, square, round(square * marker_ratio),
                     width, height)


def make_board(spec: BoardSpec, square_length: float) -> cv2.aruco.CharucoBoard:
    """Board in the given length unit (mm for calibration, px for rendering)."""
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[spec.dictionary])
    marker_length = square_length * spec.marker_px / spec.square_px
    board = cv2.aruco.CharucoBoard((spec.squares_x, spec.squares_y), square_length,
                                   marker_length, dictionary)
    board.setLegacyPattern(spec.legacy_pattern)
    return board


def render_board(spec: BoardSpec) -> np.ndarray:
    """Grayscale image of spec.image_width x image_height, board centered on white."""
    board = make_board(spec, float(spec.square_px))
    w, h = spec.squares_x * spec.square_px, spec.squares_y * spec.square_px
    pattern = board.generateImage((w, h), marginSize=0, borderBits=1)
    canvas = np.full((spec.image_height, spec.image_width), 255, np.uint8)
    x0, y0 = (spec.image_width - w) // 2, (spec.image_height - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = pattern
    return canvas


def make_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    return cv2.aruco.CharucoDetector(board)


def detect(detector: cv2.aruco.CharucoDetector, gray: np.ndarray) -> Detection | None:
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) == 0:
        return None
    return Detection(ids=ids.reshape(-1).astype(np.int32),
                     corners=corners.reshape(-1, 2).astype(np.float32))


def object_points(board: cv2.aruco.CharucoBoard, detection: Detection) -> np.ndarray:
    """3D board coordinates (N, 3) float32 of the detected corner ids."""
    return np.asarray(board.getChessboardCorners(), np.float32)[detection.ids]

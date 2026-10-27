"""How well the saved frames cover the image: corner counts on a coarse grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from tello_calibration.calib.board import Detection

GRID_COLS = 4
GRID_ROWS = 3
MIN_CORNERS_PER_CELL = 3

_ROW_NAMES = {3: ["top", "middle", "bottom"]}
_COL_NAMES = {4: ["left", "center-left", "center-right", "right"]}


@dataclass(frozen=True)
class Coverage:
    counts: np.ndarray  # (rows, cols) int: corners seen per cell over all frames
    min_per_cell: int

    @property
    def covered(self) -> np.ndarray:
        return self.counts >= self.min_per_cell

    @property
    def fraction(self) -> float:
        return float(self.covered.mean())

    @property
    def missing(self) -> list[str]:
        rows, cols = self.counts.shape
        return [cell_name(r, c, rows, cols)
                for r in range(rows) for c in range(cols) if not self.covered[r, c]]


def cell_name(row: int, col: int, rows: int, cols: int) -> str:
    row_name = _ROW_NAMES.get(rows, [str(i) for i in range(rows)])[row]
    col_name = _COL_NAMES.get(cols, [str(i) for i in range(cols)])[col]
    return f"{row_name}-{col_name}"


def cell_of(x: float, y: float, width: int, height: int,
            cols: int = GRID_COLS, rows: int = GRID_ROWS) -> tuple[int, int]:
    """(row, col) of a pixel; points on or past the far edge go to the last cell."""
    col = min(max(int(x * cols / width), 0), cols - 1)
    row = min(max(int(y * rows / height), 0), rows - 1)
    return row, col


def compute_coverage(detections: Sequence[Detection], image_size: tuple[int, int],
                     cols: int = GRID_COLS, rows: int = GRID_ROWS,
                     min_per_cell: int = MIN_CORNERS_PER_CELL) -> Coverage:
    width, height = image_size
    counts = np.zeros((rows, cols), np.int64)
    for det in detections:
        for x, y in det.corners:
            counts[cell_of(x, y, width, height, cols, rows)] += 1
    return Coverage(counts=counts, min_per_cell=min_per_cell)


def draw_coverage_map(detections: Sequence[Detection], image_size: tuple[int, int],
                      coverage: Coverage) -> np.ndarray:
    """BGR image: covered cells green, missing red, every corner a dot, counts printed."""
    width, height = image_size
    rows, cols = coverage.counts.shape
    image = np.full((height, width, 3), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c * width // cols, (c + 1) * width // cols
            y0, y1 = r * height // rows, (r + 1) * height // rows
            color = (200, 240, 200) if coverage.covered[r, c] else (200, 200, 245)
            cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, -1)
            cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), 1)
            cv2.putText(image, str(coverage.counts[r, c]), (x0 + 8, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    for det in detections:
        for x, y in det.corners:
            cv2.circle(image, (round(float(x)), round(float(y))), 2, (160, 60, 0), -1)
    cv2.putText(image, f"coverage {coverage.fraction:.0%} ({len(detections)} frames)",
                (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return image

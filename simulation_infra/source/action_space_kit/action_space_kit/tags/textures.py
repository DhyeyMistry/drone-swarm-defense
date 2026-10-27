"""tag36h11 images for the simulated tags, so a rendered camera sees decodable AprilTags.

Images come from OpenCV's ``DICT_APRILTAG_36h11`` (the same code book pupil-apriltags
detects), padded with the 1-cell white quiet zone and upscaled with nearest-neighbour so
the renderer's texture filtering keeps the cell edges sharp. They are generated on first
use into ``textures/`` next to this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TEXTURE_DIR = Path(__file__).resolve().parent / "textures"
PX_PER_CELL = 48


def tag_image(tag_id: int, px_per_cell: int = PX_PER_CELL) -> np.ndarray:
    """(10 * px, 10 * px) uint8 image: 8x8 code cells inside a 1-cell white border."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    code = cv2.aruco.generateImageMarker(dictionary, tag_id, 8, borderBits=1)  # 8x8, 1 px per cell
    code = np.pad(code, 1, constant_values=255)
    return np.kron(code, np.ones((px_per_cell, px_per_cell), dtype=np.uint8))


def texture_path(tag_id: int) -> Path:
    """PNG for ``tag_id``, generated if missing."""
    path = TEXTURE_DIR / f"tag36h11_{tag_id:05d}.png"
    if not path.is_file():
        import cv2

        TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), tag_image(tag_id))
    return path

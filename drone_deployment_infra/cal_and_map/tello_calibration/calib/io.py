"""YAML/PNG files of a calibration session and the resulting camera config.

Session directory layout::

    session.yaml          board spec, square_mm, thresholds, drone identity
    frame_000.png         raw captured frame (lossless)
    frame_000.yaml        its ChArUco detection + sharpness
    calib_report.yaml     results of both calibration passes   (written by calibrate)
    coverage_map.png, undistorted_sample.png                   (written by calibrate)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from tello_calibration.calib.board import BoardSpec, Detection

SESSION_FILE = "session.yaml"


class _Dumper(yaml.SafeDumper):
    """Mappings in block style, lists on one line: readable matrices and corner lists."""


_Dumper.add_representer(
    list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data,
                                                         flow_style=True))


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.dump(data, f, Dumper=_Dumper, sort_keys=False, default_flow_style=False)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def board_spec_to_dict(spec: BoardSpec) -> dict[str, Any]:
    data = asdict(spec)
    data["marker_ratio"] = spec.marker_px / spec.square_px  # informational
    return data


def board_spec_from_dict(data: dict[str, Any]) -> BoardSpec:
    fields = {k: v for k, v in data.items() if k != "marker_ratio"}
    return BoardSpec(**fields)


def save_board_spec(path: Path, spec: BoardSpec) -> None:
    save_yaml(path, board_spec_to_dict(spec))


def load_board_spec(path: Path) -> BoardSpec:
    return board_spec_from_dict(load_yaml(path))


def detection_to_dict(detection: Detection) -> dict[str, Any]:
    # float(np.float32) is exact, and yaml writes repr(), so this round-trips bit-for-bit.
    return {"ids": [int(i) for i in detection.ids],
            "corners": [[float(x), float(y)] for x, y in detection.corners]}


def detection_from_dict(data: dict[str, Any]) -> Detection:
    return Detection(ids=np.asarray(data["ids"], np.int32),
                     corners=np.asarray(data["corners"], np.float32).reshape(-1, 2))


@dataclass(frozen=True)
class SessionFrame:
    name: str          # e.g. "frame_007"
    image_path: Path
    detection: Detection
    sharpness: float


def save_frame(session_dir: Path, index: int, image: np.ndarray, detection: Detection,
               sharpness: float, t_recv: float | None) -> SessionFrame:
    name = f"frame_{index:03d}"
    image_path = session_dir / f"{name}.png"
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"could not write {image_path}")
    save_yaml(session_dir / f"{name}.yaml", {
        "image": image_path.name,
        "t_recv": t_recv,
        "sharpness": float(sharpness),
        "n_corners": len(detection),
        **detection_to_dict(detection),
    })
    return SessionFrame(name, image_path, detection, float(sharpness))


def load_session(session_dir: Path) -> tuple[dict[str, Any], list[SessionFrame]]:
    session = load_yaml(session_dir / SESSION_FILE)
    frames = []
    for path in sorted(session_dir.glob("frame_*.yaml")):
        data = load_yaml(path)
        frames.append(SessionFrame(path.stem, session_dir / data["image"],
                                   detection_from_dict(data), float(data["sharpness"])))
    return session, frames


def camera_to_dict(K: np.ndarray, dist: np.ndarray) -> dict[str, Any]:
    return {"camera_matrix": [[float(v) for v in row] for row in K],
            "dist_coeffs": [float(v) for v in dist.reshape(-1)]}


def camera_from_dict(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(data["camera_matrix"], np.float64),
            np.asarray(data["dist_coeffs"], np.float64))

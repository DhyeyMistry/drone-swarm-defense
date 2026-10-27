"""Pinhole + 5-coefficient calibration from saved ChArUco detections, and its report.

OpenCV 5 has no aruco.calibrateCameraCharuco; this is its equivalent: the detected
ChArUco corners and their board coordinates go into cv2.calibrateCamera.
One or more session directories can be combined; each frame keeps its own session's
board scale (square_mm), which only affects extrinsics, not K or distortion.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tello_calibration.calib.board import Detection, detect, make_board, make_detector, object_points
from tello_calibration.calib.coverage import Coverage, cell_of, compute_coverage, draw_coverage_map
from tello_calibration.calib.io import (
    SessionFrame,
    board_spec_from_dict,
    camera_to_dict,
    load_session,
    save_yaml,
)

log = logging.getLogger(__name__)

OUTLIER_FACTOR = 3.0


@dataclass(frozen=True)
class Criteria:
    max_rms: float = 1.0
    min_coverage: float = 0.8
    min_frames: int = 20
    max_fx_spread: float = 0.005  # leave-one-out (max - min) / fx


@dataclass(frozen=True)
class CalibFrame:
    name: str
    detection: Detection
    object_points: np.ndarray  # (N, 3) float32 board coordinates of the detected ids


@dataclass(frozen=True)
class CalibResult:
    K: np.ndarray
    dist: np.ndarray                 # k1, k2, p1, p2, k3
    rms: float
    frame_errors: dict[str, float]   # per-frame RMS reprojection error, px
    coverage: Coverage
    image_size: tuple[int, int]
    cell_errors: np.ndarray          # (rows, cols) RMS reprojection error per grid cell, px
    flags: int = 0                   # cv2.calibrateCamera flags used

    @property
    def n_frames(self) -> int:
        return len(self.frame_errors)


@dataclass(frozen=True)
class Stability:
    """fx/fy when each frame of the final set is left out once."""

    fx_ref: float
    fy_ref: float
    fx: dict[str, float]  # frame left out -> fx
    fy: dict[str, float]

    @property
    def fx_spread(self) -> float:
        return (max(self.fx.values()) - min(self.fx.values())) / self.fx_ref

    @property
    def fy_spread(self) -> float:
        return (max(self.fy.values()) - min(self.fy.values())) / self.fy_ref

    @property
    def most_influential(self) -> tuple[str, float]:
        """Frame whose removal moves fx the most, and by how much (relative)."""
        name = max(self.fx, key=lambda n: abs(self.fx[n] - self.fx_ref))
        return name, (self.fx[name] - self.fx_ref) / self.fx_ref


@dataclass(frozen=True)
class SessionCalibration:
    first: CalibResult
    second: CalibResult | None       # after dropping outliers; None if none dropped
    dropped: list[str]
    stability: Stability
    failures: list[str] = field(default_factory=list)  # empty = PASS

    @property
    def final(self) -> CalibResult:
        return self.second or self.first

    @property
    def passed(self) -> bool:
        return not self.failures


def _calibrate_camera(frames: Sequence[CalibFrame], image_size: tuple[int, int], flags: int
                      ) -> tuple[float, np.ndarray, np.ndarray, Any, Any]:
    obj = [f.object_points for f in frames]
    img = [f.detection.corners.reshape(-1, 1, 2) for f in frames]
    # Multithreaded reductions change float summation order between runs; one thread
    # makes repeated calibrations of the same data bit-identical.
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        return cv2.calibrateCamera(obj, img, image_size, None, None, flags=flags)
    finally:
        cv2.setNumThreads(threads)


def calibrate(frames: Sequence[CalibFrame], image_size: tuple[int, int], flags: int = 0
              ) -> CalibResult:
    rms, K, dist, rvecs, tvecs = _calibrate_camera(frames, image_size, flags)
    coverage = compute_coverage([f.detection for f in frames], image_size)
    errors = {}
    cell_sq = np.zeros(coverage.counts.shape)
    for frame, rvec, tvec in zip(frames, rvecs, tvecs):
        projected, _ = cv2.projectPoints(frame.object_points, rvec, tvec, K, dist)
        sq = np.sum((projected.reshape(-1, 2) - frame.detection.corners) ** 2, axis=1)
        errors[frame.name] = float(np.sqrt(np.mean(sq)))
        for (x, y), e in zip(frame.detection.corners, sq):
            cell_sq[cell_of(x, y, *image_size, *coverage.counts.shape[::-1])] += e
    with np.errstate(invalid="ignore"):  # empty cells -> nan
        cell_errors = np.sqrt(cell_sq / coverage.counts)
    return CalibResult(K, dist.reshape(-1), float(rms), errors, coverage, image_size,
                       cell_errors, flags)


def leave_one_out(frames: Sequence[CalibFrame], image_size: tuple[int, int], flags: int,
                  reference: CalibResult) -> Stability:
    fx, fy = {}, {}
    for i, frame in enumerate(frames):
        _, K, _, _, _ = _calibrate_camera([*frames[:i], *frames[i + 1:]], image_size, flags)
        fx[frame.name], fy[frame.name] = float(K[0, 0]), float(K[1, 1])
    return Stability(float(reference.K[0, 0]), float(reference.K[1, 1]), fx, fy)


def calibrate_with_rejection(frames: Sequence[CalibFrame], image_size: tuple[int, int],
                             criteria: Criteria = Criteria(), flags: int = 0
                             ) -> SessionCalibration:
    """Calibrate, drop frames with error > 3x median, recalibrate once, then check
    stability by leaving each remaining frame out once."""
    first = calibrate(frames, image_size, flags)
    limit = OUTLIER_FACTOR * float(np.median(list(first.frame_errors.values())))
    dropped = [name for name, err in first.frame_errors.items() if err > limit]
    kept = [f for f in frames if f.name not in dropped]
    second = calibrate(kept, image_size, flags) if dropped else None
    final = second or first
    stability = leave_one_out(kept, image_size, flags, final)
    return SessionCalibration(first, second, dropped, stability,
                              judge(final, criteria, stability))


def judge(result: CalibResult, criteria: Criteria = Criteria(),
          stability: Stability | None = None) -> list[str]:
    """Reasons the calibration is not good enough, each with what to capture; [] = PASS."""
    failures = []
    if result.n_frames < criteria.min_frames:
        failures.append(f"only {result.n_frames} frames used (need {criteria.min_frames}): "
                        "capture more distinct views")
    if result.coverage.fraction < criteria.min_coverage:
        failures.append(f"coverage {result.coverage.fraction:.0%} < "
                        f"{criteria.min_coverage:.0%}: hold the board so it reaches "
                        f"{', '.join(result.coverage.missing)}")
    if result.rms >= criteria.max_rms:
        worst = sorted(result.frame_errors, key=result.frame_errors.get, reverse=True)[:3]
        failures.append(f"RMS {result.rms:.3f} px >= {criteria.max_rms} px (worst: "
                        f"{', '.join(worst)}): hold still at each beep, avoid glare, "
                        "raise --min-sharpness")
    if stability is not None and stability.fx_spread >= criteria.max_fx_spread:
        name, shift = stability.most_influential
        failures.append(f"fx unstable: leave-one-out spread {stability.fx_spread:.2%} >= "
                        f"{criteria.max_fx_spread:.1%} (dropping {name} moves fx {shift:+.2%})"
                        ": add strongly tilted (40-50 deg) and near/far views")
    return failures


def verify_detections(frames: Sequence[SessionFrame], detector: cv2.aruco.CharucoDetector
                      ) -> list[str]:
    """Re-detect every saved PNG; return names whose detection differs from the yaml."""
    mismatched = []
    for frame in frames:
        gray = cv2.cvtColor(cv2.imread(str(frame.image_path)), cv2.COLOR_BGR2GRAY)
        again = detect(detector, gray)
        same = (again is not None and np.array_equal(again.ids, frame.detection.ids)
                and np.array_equal(again.corners, frame.detection.corners))
        if not same:
            mismatched.append(frame.name)
    return mismatched


def load_sessions(session_dirs: Sequence[Path]
                  ) -> tuple[list[dict[str, Any]], list[SessionFrame], list[CalibFrame],
                             tuple[int, int]]:
    """Load and check one or more sessions of the same drone, image size and board.

    With several sessions, frame names get a "<session dir name>/" prefix.
    """
    sessions, session_frames, calib_frames = [], [], []
    for session_dir in session_dirs:
        session, frames = load_session(session_dir)
        if not frames:
            raise ValueError(f"no frame_*.yaml in {session_dir}")
        if sessions:
            for key in ("drone", "image_width", "image_height", "board"):
                if session[key] != sessions[0][key]:
                    raise ValueError(f"{session_dir}: {key} differs from {session_dirs[0]}")
            if session["square_mm"] != sessions[0]["square_mm"]:
                log.warning("%s: square_mm %s differs from %s (fine: affects only "
                            "extrinsics)", session_dir.name, session["square_mm"],
                            sessions[0]["square_mm"])
        board = make_board(board_spec_from_dict(session["board"]), float(session["square_mm"]))

        mismatched = verify_detections(frames, make_detector(board))
        if mismatched:
            log.warning("%s: re-detection differs from saved detections for %d frame(s) (%s);"
                        " using the saved ones. Was the OpenCV version changed?",
                        session_dir.name, len(mismatched), ", ".join(mismatched[:5]))

        for frame in frames:
            if len(session_dirs) > 1:
                frame = dataclasses.replace(frame, name=f"{session_dir.name}/{frame.name}")
            session_frames.append(frame)
            calib_frames.append(CalibFrame(frame.name, frame.detection,
                                           object_points(board, frame.detection)))
        sessions.append(session)
    image_size = (int(sessions[0]["image_width"]), int(sessions[0]["image_height"]))
    return sessions, session_frames, calib_frames, image_size


def calibrate_sessions(session_dirs: Sequence[Path], criteria: Criteria = Criteria(),
                       flags: int = 0
                       ) -> tuple[list[dict[str, Any]], list[SessionFrame], SessionCalibration]:
    """Calibrate from what is on disk. Used for live runs and --from-dir alike."""
    sessions, frames, calib_frames, image_size = load_sessions(session_dirs)
    return sessions, frames, calibrate_with_rejection(calib_frames, image_size, criteria, flags)


def calibrate_session(session_dir: Path, criteria: Criteria = Criteria(), flags: int = 0
                      ) -> tuple[dict[str, Any], list[SessionFrame], SessionCalibration]:
    sessions, frames, calib = calibrate_sessions([session_dir], criteria, flags)
    return sessions[0], frames, calib


def result_to_dict(result: CalibResult) -> dict[str, Any]:
    return {
        **camera_to_dict(result.K, result.dist),
        "rms_px": result.rms,
        "n_frames": result.n_frames,
        "coverage": result.coverage.fraction,
        "coverage_counts": result.coverage.counts.tolist(),
        "coverage_missing": result.coverage.missing,
        "cell_errors_px": [[None if np.isnan(e) else float(e) for e in row]
                           for row in result.cell_errors],
        "frame_errors_px": result.frame_errors,
        "fix_k3": bool(result.flags & cv2.CALIB_FIX_K3),
    }


def stability_to_dict(stability: Stability) -> dict[str, Any]:
    name, shift = stability.most_influential
    return {
        "fx_spread": stability.fx_spread,
        "fy_spread": stability.fy_spread,
        "most_influential_frame": name,
        "most_influential_fx_shift": shift,
        "fx_without_frame": stability.fx,
        "fy_without_frame": stability.fy,
    }


def camera_config(sessions: Sequence[dict[str, Any]], session_dirs: Sequence[Path],
                  calib: SessionCalibration, criteria: Criteria = Criteria()
                  ) -> dict[str, Any]:
    result, first = calib.final, sessions[0]
    width, height = result.image_size
    return {
        "drone": first["drone"],
        "ssid": first.get("ssid"),
        "bssid": first.get("bssid"),
        "serial": first.get("serial"),
        "image_width": width,
        "image_height": height,
        "model": "pinhole",
        "distortion_model": "plumb_bob (k1, k2, p1, p2, k3)",
        "fix_k3": bool(result.flags & cv2.CALIB_FIX_K3),
        **camera_to_dict(result.K, result.dist),
        "rms_px": result.rms,
        "n_frames": result.n_frames,
        "coverage": result.coverage.fraction,
        "fx_spread_loo": calib.stability.fx_spread,
        "fy_spread_loo": calib.stability.fy_spread,
        "max_fx_spread": criteria.max_fx_spread,  # PASS threshold used (fraction)
        "date": dt.date.today().isoformat(),
        "source_sessions": [str(d.resolve()) for d in session_dirs],
    }


def write_outputs(out_dir: Path, frames: Sequence[SessionFrame],
                  calib: SessionCalibration, sources: Sequence[Path]) -> None:
    """calib_report.yaml, coverage_map.png, undistorted_sample.png in out_dir."""
    save_yaml(out_dir / "calib_report.yaml", {
        "source_sessions": [str(d.resolve()) for d in sources],
        "first_pass": result_to_dict(calib.first),
        "dropped_outliers": calib.dropped,
        "second_pass": result_to_dict(calib.second) if calib.second else None,
        "leave_one_out": stability_to_dict(calib.stability),
        "verdict": "PASS" if calib.passed else "FAIL",
        "failures": calib.failures,
    })

    final = calib.final
    used = [f for f in frames if f.name in final.frame_errors]
    cv2.imwrite(str(out_dir / "coverage_map.png"),
                draw_coverage_map([f.detection for f in used], final.image_size,
                                  final.coverage))

    sharpest = max(used, key=lambda f: f.sharpness)
    raw = cv2.imread(str(sharpest.image_path))
    undistorted = cv2.undistort(raw, final.K, final.dist)
    for image, label in ((raw, f"raw {sharpest.name}"), (undistorted, "undistorted")):
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / "undistorted_sample.png"), cv2.hconcat([raw, undistorted]))

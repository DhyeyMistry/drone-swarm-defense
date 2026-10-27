"""Calibrate a Tello camera from a ChArUco board shown on this laptop's screen.

Live: verify the drone (same identity gate as 01), stream, auto-capture good distinct
views into data/calib_<drone>_<timestamp>/, then calibrate from that directory.
Offline: --from-dir DIR [DIR ...] recalibrates saved session(s) with identical results;
several sessions are combined. PASS also needs a stable fx (leave-one-out spread < 0.5%).
On PASS writes config/camera/<drone>.yaml; on FAIL writes nothing to config/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tello_calibration.calib.board import BoardSpec, detect, make_board, make_detector
from tello_calibration.calib.calibrate import (
    CalibResult,
    Criteria,
    calibrate_sessions,
    camera_config,
    write_outputs,
)
from tello_calibration.calib.coverage import compute_coverage
from tello_calibration.calib.gate import CaptureGate, GateConfig
from tello_calibration.calib.io import (
    SESSION_FILE,
    board_spec_to_dict,
    load_board_spec,
    load_yaml,
    save_frame,
    save_yaml,
)
from tello_calibration.session import (
    IdentityError,
    SessionError,
    report,
    resolve_drone,
    verified_link,
    video_stream,
)
from tello_calibration.utils.config import DroneConfig
from tello_calibration.utils.sound import beep
from tello_calibration.video.stream import VideoReceiver

log = logging.getLogger("calibrate")

ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = ROOT / "data" / "boards"
CAMERA_DIR = ROOT / "config" / "camera"
EXPECTED_SIZE = (960, 720)
STATUS_EVERY_S = 2.0
STALL_WARN_S = 2.0
WINDOW = "calibration preview"


def default_board() -> Path | None:
    boards = sorted(BOARD_DIR.glob("charuco_*.yaml"))
    return boards[0] if len(boards) == 1 else None


def new_session(drone: DroneConfig, spec: BoardSpec, square_mm: float, gate_cfg: GateConfig,
                target: int) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = ROOT / "data" / f"calib_{drone.name}_{stamp}"
    session_dir.mkdir(parents=True)
    save_yaml(session_dir / SESSION_FILE, {
        "drone": drone.name,
        "ssid": drone.ssid,
        "bssid": drone.bssid,
        "serial": drone.serial,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "opencv_version": cv2.__version__,
        "image_width": EXPECTED_SIZE[0],
        "image_height": EXPECTED_SIZE[1],
        "square_mm": square_mm,
        "marker_mm": spec.marker_mm(square_mm),
        "board": board_spec_to_dict(spec),
        "gate": gate_cfg.to_dict(),
        "target_frames": target,
    })
    return session_dir


def capture(video: VideoReceiver, spec: BoardSpec, square_mm: float, gate_cfg: GateConfig,
            session_dir: Path, target: int, headless: bool) -> int:
    """Auto-capture until `target` frames, 'q' (preview only) or Ctrl+C. Returns count."""
    board = make_board(spec, square_mm)
    detector = make_detector(board)
    gate = CaptureGate(board, spec.n_corners, EXPECTED_SIZE, gate_cfg)
    detections = []
    last_id, last_reason, last_sharp = 0, "waiting for frames", None
    last_new = last_status = time.monotonic()

    log.info("Capturing up to %d frames into %s. Move the drone slowly; hold still for "
             "each beep. Ctrl+C to stop early.", target, session_dir)
    try:
        while len(detections) < target:
            frame, frame_id, t_recv = video.get_latest()
            now = time.monotonic()
            if frame is None or frame_id == last_id:
                if now - last_new > STALL_WARN_S:
                    log.warning("No new video frames for %.0f s", now - last_new)
                    last_new = now
                if headless:
                    time.sleep(0.005)
                elif cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
            last_id, last_new = frame_id, now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection = detect(detector, gray)
            decision = gate.evaluate(detection, gray, t_recv or now)
            last_reason = decision.reason
            last_sharp = decision.sharpness if decision.sharpness is not None else last_sharp

            if decision.accept:
                assert detection is not None and decision.sharpness is not None
                save_frame(session_dir, len(detections), frame, detection,
                           decision.sharpness, t_recv)
                gate.commit(decision, t_recv or now)
                detections.append(detection)
                beep("capture")
                cov = compute_coverage(detections, EXPECTED_SIZE)
                log.info("saved %2d/%d | %2d corners | sharpness %4.0f | coverage %3.0f%% | "
                         "need: %s", len(detections), target, len(detection),
                         decision.sharpness, 100 * cov.fraction,
                         ", ".join(cov.missing) or "nothing, vary distance/tilt")

            if now - last_status >= STATUS_EVERY_S:
                sharp = f"{last_sharp:.0f}" if last_sharp is not None else "-"
                log.info("status: %d/%d saved | last frame: %s | sharpness %s (min %.0f)",
                         len(detections), target, last_reason, sharp, gate_cfg.min_sharpness)
                last_status = now

            if not headless:
                preview = frame.copy()
                if detection is not None:
                    cv2.aruco.drawDetectedCornersCharuco(
                        preview, detection.corners.reshape(-1, 1, 2),
                        detection.ids.reshape(-1, 1))
                cv2.putText(preview, f"{len(detections)}/{target} {last_reason}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(WINDOW, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("'q' pressed")
                    break
    except KeyboardInterrupt:
        log.info("Ctrl+C: stopping capture with %d frames", len(detections))
    finally:
        if not headless:
            cv2.destroyAllWindows()
    return len(detections)


def log_result(title: str, result: CalibResult) -> None:
    K, d = result.K, result.dist
    log.info("%s: %d frames, RMS %.3f px, coverage %.0f%%", title, result.n_frames,
             result.rms, 100 * result.coverage.fraction)
    log.info("  fx=%.2f fy=%.2f cx=%.2f cy=%.2f", K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    log.info("  dist k1=%.5f k2=%.5f p1=%.5f p2=%.5f k3=%.5f", *d)
    log.info("  coverage grid (corners per cell, top row first):")
    for row in result.coverage.counts:
        log.info("    %s", "  ".join(f"{n:4d}" for n in row))
    log.info("  reprojection error per cell, px (top row first):")
    for row in result.cell_errors:
        log.info("    %s", "  ".join("   -" if np.isnan(e) else f"{e:4.2f}" for e in row))
    errors = "  ".join(f"{name[-3:]}:{err:.2f}" for name, err in result.frame_errors.items())
    log.info("  per-frame error px: %s", errors)


def calib_flags(args: argparse.Namespace) -> int:
    return cv2.CALIB_FIX_K3 if args.fix_k3 else 0


def output_dir(session_dirs: list[Path], out_dir: Path | None) -> Path:
    """A single session keeps its report; a combination gets its own directory."""
    if out_dir is not None:
        return out_dir
    if len(session_dirs) == 1:
        return session_dirs[0]
    drone = load_yaml(session_dirs[0] / SESSION_FILE)["drone"]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "data" / f"calib_{drone}_combined_{stamp}"


def finish(session_dirs: list[Path], camera_dir: Path, criteria: Criteria, flags: int = 0,
           out_dir: Path | None = None) -> int:
    """Calibrate from session directories, write outputs; PASS -> camera yaml."""
    log.info("Calibrating from %s", ", ".join(str(d) for d in session_dirs))
    if flags & cv2.CALIB_FIX_K3:
        log.info("k3 fixed at 0 (CALIB_FIX_K3)")
    sessions, frames, calib = calibrate_sessions(session_dirs, criteria, flags)
    log_result("Pass 1", calib.first)
    if calib.second is not None:
        log.info("Dropped %d outlier frame(s) with error > 3x median: %s",
                 len(calib.dropped), ", ".join(calib.dropped))
        log_result("Pass 2 (after dropping outliers)", calib.second)
    else:
        log.info("No outlier frames (none above 3x median); single pass")
    stab = calib.stability
    name, shift = stab.most_influential
    log.info("Leave-one-out over %d frames: fx %.2f..%.2f (spread %.2f%%), "
             "fy %.2f..%.2f (spread %.2f%%); most influential %s (fx %+.2f%%)",
             len(stab.fx), min(stab.fx.values()), max(stab.fx.values()),
             100 * stab.fx_spread, min(stab.fy.values()), max(stab.fy.values()),
             100 * stab.fy_spread, name, 100 * shift)

    out = output_dir(session_dirs, out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_outputs(out, frames, calib, session_dirs)
    log.info("Wrote calib_report.yaml, coverage_map.png, undistorted_sample.png to %s", out)

    if not calib.passed:
        for reason in calib.failures:
            report("calibration", False, reason)
        log.info("Nothing written to %s", camera_dir)
        return 1
    final = calib.final
    report("calibration", True, f"RMS {final.rms:.3f} px < {criteria.max_rms}, coverage "
           f"{final.coverage.fraction:.0%}, {final.n_frames} frames, fx spread "
           f"{stab.fx_spread:.2%} < {criteria.max_fx_spread:.1%}")
    camera_dir.mkdir(parents=True, exist_ok=True)
    camera_file = camera_dir / f"{sessions[0]['drone']}.yaml"
    save_yaml(camera_file, camera_config(sessions, session_dirs, calib, criteria))
    log.info("Wrote %s", camera_file)
    return 0


def live(args: argparse.Namespace, criteria: Criteria) -> int:
    board_path = args.board or default_board()
    if board_path is None:
        log.error("Pass --board (none or several charuco_*.yaml in %s; run "
                  "02_make_charuco.py first)", BOARD_DIR)
        return 1
    spec = load_board_spec(board_path)
    gate_cfg = GateConfig(min_sharpness=args.min_sharpness)

    try:
        drone, local_ip = resolve_drone(args.drone)
        with verified_link(drone, local_ip) as link, video_stream(link, local_ip) as video:
            frame = video.get_latest()[0]
            assert frame is not None
            size = (frame.shape[1], frame.shape[0])
            report("frame size", size == EXPECTED_SIZE,
                   f"{size[0]}x{size[1]} (expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]})")
            if size != EXPECTED_SIZE:
                return 1
            session_dir = new_session(drone, spec, args.square_mm, gate_cfg, args.target)
            n = capture(video, spec, args.square_mm, gate_cfg, session_dir, args.target,
                        args.headless)
    except IdentityError:
        return 2
    except SessionError:
        return 1
    except KeyboardInterrupt:
        log.info("Ctrl+C before capture started")
        return 130
    beep("done")

    if n < criteria.min_frames:
        log.error("Only %d frames captured (need %d); not calibrating. Session kept in %s",
                  n, criteria.min_frames, session_dir)
        return 1
    return finish([session_dir], args.camera_dir, criteria, calib_flags(args))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", help="drone name in config/fleet.yaml (live capture)")
    parser.add_argument("--square-mm", type=float, help="measured square edge in mm (live)")
    parser.add_argument("--board", type=Path, help="board yaml from 02_make_charuco.py")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True,
                        help="no preview window (default on: the board fills the screen)")
    parser.add_argument("--max-fx-spread", type=float, default=0.5, metavar="PERCENT",
                        help="PASS needs leave-one-out fx spread below this, in %% "
                             "(default %(default)s)")
    parser.add_argument("--fix-k3", action="store_true",
                        help="hold k3 at 0 (cv2.CALIB_FIX_K3); often steadier for mild lenses")
    parser.add_argument("--min-sharpness", type=float, default=GateConfig.min_sharpness,
                        help="Laplacian variance threshold inside the board (default %(default)s)")
    parser.add_argument("--target", type=int, default=40, help="frames to capture")
    parser.add_argument("--from-dir", type=Path, nargs="+", metavar="DIR",
                        help="recalibrate saved session(s), combined if several; no drone")
    parser.add_argument("--out-dir", type=Path,
                        help="where reports go (default: the session, or a new "
                             "data/calib_<drone>_combined_<time>/ for several)")
    parser.add_argument("--camera-dir", type=Path, default=CAMERA_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s")
    if args.max_fx_spread <= 0:
        parser.error("--max-fx-spread must be > 0")
    criteria = Criteria(max_fx_spread=args.max_fx_spread / 100)

    if args.from_dir:
        missing = [d for d in args.from_dir if not (d / SESSION_FILE).exists()]
        if missing:
            log.error("No %s in %s", SESSION_FILE, ", ".join(map(str, missing)))
            return 1
        try:
            return finish(args.from_dir, args.camera_dir, criteria, calib_flags(args),
                          args.out_dir)
        except ValueError as exc:
            log.error("%s", exc)
            return 1
    if not args.drone or args.square_mm is None:
        parser.error("live capture needs --drone and --square-mm (or use --from-dir)")
    return live(args, criteria)


if __name__ == "__main__":
    sys.exit(main())

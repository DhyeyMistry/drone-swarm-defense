"""Live AprilTag check: per tag id, distance, view angle and ambiguity ratio.

Accepted tags are drawn green; rejected ones red with the reason. Thresholds come from
config/tags.yaml so they can be tuned on real video. 'q' or Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tello_calibration.calib.io import camera_from_dict, load_yaml
from tello_calibration.session import IdentityError, SessionError, resolve_drone, verified_link, \
    video_stream
from tello_calibration.tags.detect import TagDetector, TagObservation, load_tag_config
from tello_calibration.utils.config import CONFIG_DIR
from tello_calibration.video.stream import VideoReceiver

log = logging.getLogger("tag_check")

WINDOW = "tags"
GREEN, RED = (0, 220, 0), (0, 0, 255)


def label(obs: TagObservation) -> str:
    ratio = f"r={obs.ambiguity_ratio:.1f}" if obs.ambiguity_ratio is not None else "r=?"
    angle = f"{obs.view_angle_deg:.0f}deg" if obs.view_angle_deg is not None else ""
    quality = f"ew={obs.edge_width_px:.1f} m={obs.decision_margin:.0f}"
    if obs.accepted:
        return f"id {obs.tag_id} {obs.distance_m:.2f}m {angle} {ratio} {quality}"
    return f"id {obs.tag_id} {obs.reasons[0]} {angle} {ratio} {quality}"


def draw(image: np.ndarray, observations: list[TagObservation], fps: float,
         detector: TagDetector) -> None:
    for obs in observations:
        color = GREEN if obs.accepted else RED
        size = detector.cfg.size_m(obs.tag_id)
        if obs.accepted and obs.rvec is not None and obs.tvec is not None and size:
            # x red, y green, z blue (out of the tag face); half a tag long
            cv2.drawFrameAxes(image, detector.K, detector.dist, obs.rvec, obs.tvec,
                              size / 2, 2)
        pts = obs.corners.round().astype(np.int32)
        cv2.polylines(image, [pts], True, color, 2)
        cv2.circle(image, tuple(pts[0]), 4, color, -1)  # top-left corner of the tag frame
        x, y = pts.min(axis=0)
        cv2.putText(image, label(obs), (int(x), max(int(y) - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(image, label(obs), (int(x), max(int(y) - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    n_ok = sum(o.accepted for o in observations)
    cv2.putText(image, f"{fps:4.1f} fps  tags {n_ok}/{len(observations)} accepted", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def view_loop(video: VideoReceiver, detector: TagDetector) -> None:
    last_id, frames, t0, fps = 0, 0, time.monotonic(), 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        frame, frame_id, _ = video.get_latest()
        if frame is not None and frame_id != last_id:
            last_id, frames = frame_id, frames + 1
            observations = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            image = frame.copy()
            draw(image, observations, fps, detector)
            cv2.imshow(WINDOW, image)
        now = time.monotonic()
        if now - t0 >= 1.0:
            fps, frames, t0 = frames / (now - t0), 0, now
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", required=True, help="drone name in config/fleet.yaml")
    parser.add_argument("--tags", type=Path, default=CONFIG_DIR / "tags.yaml")
    parser.add_argument("--camera", type=Path, help="default: config/camera/<drone>.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s")

    camera_path = args.camera or CONFIG_DIR / "camera" / f"{args.drone}.yaml"
    if not camera_path.exists():
        log.error("No calibration at %s; run 03_calibrate.py first", camera_path)
        return 1
    camera = load_yaml(camera_path)
    K, dist = camera_from_dict(camera)
    tags = load_tag_config(args.tags)
    if not any(size is not None for size in tags.sizes_mm.values()):
        log.error("No tag sizes in %s; fill sizes_mm (black-square edge, mm)", args.tags)
        return 1
    log.info("Camera %s (%dx%d, rms %.3f px); gate %s", camera_path.name,
             camera["image_width"], camera["image_height"], camera["rms_px"], tags.gate)
    detector = TagDetector(tags, K, dist)

    try:
        drone, local_ip = resolve_drone(args.drone)
        with verified_link(drone, local_ip) as link, video_stream(link, local_ip) as video:
            frame = video.get_latest()[0]
            assert frame is not None
            if (frame.shape[1], frame.shape[0]) != (camera["image_width"],
                                                    camera["image_height"]):
                log.error("Frame %dx%d does not match calibration", frame.shape[1],
                          frame.shape[0])
                return 1
            log.info("Showing tags. 'q' in the window or Ctrl+C to stop.")
            try:
                view_loop(video, detector)
            finally:
                cv2.destroyAllWindows()
    except IdentityError:
        return 2
    except SessionError:
        return 1
    except KeyboardInterrupt:
        log.info("Ctrl+C received")
    return 0


if __name__ == "__main__":
    sys.exit(main())

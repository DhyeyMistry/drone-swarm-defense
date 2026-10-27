"""Localize the drone in the tag-10 (origin) frame from a saved map.

Loads config/map.yaml (built by 05_map.py). For each frame it detects AprilTags, and
from any tag that is in the map it computes the camera's position in the origin frame
-- i.e. where the drone is relative to tag 10. Prints and draws it live. When several
mapped tags are visible it uses the closest one and reports the spread as a check.

This never flies: comms is the calibration allowlist (no takeoff/rc).
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
from tello_calibration.session import IdentityError, SessionError, resolve_drone, \
    verified_link, video_stream
from tello_calibration.tags.detect import TagDetector, load_tag_config
from tello_calibration.tags.mapping import TagMap
from tello_calibration.utils.config import CONFIG_DIR
from tello_calibration.video.stream import VideoReceiver

log = logging.getLogger("localize")
WINDOW = "localize"
GREEN, GREY, RED = (0, 220, 0), (160, 160, 160), (0, 0, 255)


def estimate(observations, tag_map: TagMap):
    """(position in origin frame, tag ids used, spread m) from all mapped sightings."""
    points, used = [], []
    for obs in observations:
        if obs.accepted and obs.tag_id in tag_map.poses:
            p = tag_map.camera_in_origin(obs)
            if p is not None:
                points.append((obs.distance_m or 1e9, obs.tag_id, p))
    if not points:
        return None, [], 0.0
    points.sort(key=lambda x: x[0])          # nearest tag first = most reliable
    used = [tid for _, tid, _ in points]
    coords = np.array([p for _, _, p in points])
    spread = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))) if len(coords) > 1 else 0.0
    return points[0][2], used, spread


def draw(image, observations, tag_map, pos, used, spread, fps) -> None:
    for obs in observations:
        mapped = obs.accepted and obs.tag_id in tag_map.poses
        color = GREEN if mapped else GREY if obs.accepted else RED
        pts = obs.corners.round().astype(np.int32)
        cv2.polylines(image, [pts], True, color, 2)
    if pos is not None:
        txt = f"drone @ origin  x {pos[0]:+.2f}  y {pos[1]:+.2f}  z {pos[2]:+.2f} m  via {used}"
        if spread > 0:
            txt += f"  spread {spread:.2f}m"
    else:
        txt = "no mapped tag in view"
    cv2.putText(image, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(image, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, f"{fps:4.1f} fps", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def localize_loop(video: VideoReceiver, detector: TagDetector, tag_map: TagMap) -> None:
    last_id, frames, t0, fps, last_log = 0, 0, time.monotonic(), 0.0, 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        frame, frame_id, _ = video.get_latest()
        if frame is not None and frame_id != last_id:
            last_id, frames = frame_id, frames + 1
            observations = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            pos, used, spread = estimate(observations, tag_map)
            image = frame.copy()
            draw(image, observations, tag_map, pos, used, spread, fps)
            cv2.imshow(WINDOW, image)
            now = time.monotonic()
            if pos is not None and now - last_log > 0.5:
                log.info("drone @ origin x %+.2f y %+.2f z %+.2f m  via %s", pos[0], pos[1],
                         pos[2], used)
                last_log = now
        now = time.monotonic()
        if now - t0 >= 1.0:
            fps, frames, t0 = frames / (now - t0), 0, now
        if (cv2.waitKey(1) & 0xFF == ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", required=True, help="drone name in config/fleet.yaml")
    parser.add_argument("--map", type=Path, default=CONFIG_DIR / "map.yaml")
    parser.add_argument("--tags", type=Path, default=CONFIG_DIR / "tags.yaml")
    parser.add_argument("--camera", type=Path, help="default: config/camera/<drone>.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s")

    if not args.map.exists():
        log.error("No map at %s; run 05_map.py first", args.map)
        return 1
    camera_path = args.camera or CONFIG_DIR / "camera" / f"{args.drone}.yaml"
    if not camera_path.exists():
        log.error("No calibration at %s; run 03_calibrate.py first", camera_path)
        return 1
    K, dist = camera_from_dict(load_yaml(camera_path))
    tag_map = TagMap.load(args.map)
    detector = TagDetector(load_tag_config(args.tags), K, dist)
    log.info("Map: %d tags in tag-%d frame: %s", len(tag_map.mapped()), tag_map.origin_tag,
             tag_map.mapped())

    try:
        drone, local_ip = resolve_drone(args.drone)
        with verified_link(drone, local_ip) as link, video_stream(link, local_ip) as video:
            try:
                localize_loop(video, detector, tag_map)
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

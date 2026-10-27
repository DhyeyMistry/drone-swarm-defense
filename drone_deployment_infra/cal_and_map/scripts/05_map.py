"""Build a map of AprilTags in the origin tag's (tag 10) frame and save it.

Point the drone around the room so the camera sees the origin tag together with the
other tags (two or more tags in a frame links them). Every frame's tag corners go into a
bundle adjustment (tello_calibration.tags.bundle): all camera and tag poses solved together,
so the flip-ambiguous poses of small tags never get averaged into the map. Every tag is
checked: views, 1-sigma uncertainty, view spread, and redundancy (two halves of the session
solved independently must agree). Tags turn GREEN when they pass, YELLOW shows what is
still missing (re-solved every few seconds). 'q' or Ctrl+C stops: a final solve on every
frame writes config/map.yaml only if every tag passes (--force writes anyway).

The session video is saved to data/map_<drone>_<time>.avi; rebuild from it any time with
scripts/07_map_from_video.py. This never flies: comms is the calibration allowlist.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from tello_calibration.calib.io import camera_from_dict, load_yaml
from tello_calibration.session import IdentityError, SessionError, resolve_drone, \
    verified_link, video_stream
from tello_calibration.tags.bundle import BundleMapper, BundleResult, report, result_to_dict
from tello_calibration.tags.detect import TagDetector, load_tag_config
from tello_calibration.calib.io import save_yaml
from tello_calibration.utils.config import CONFIG_DIR
from tello_calibration.video.stream import VideoReceiver

log = logging.getLogger("map")
WINDOW = "map"
GREEN, YELLOW, RED = (0, 220, 0), (0, 220, 220), (0, 0, 255)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LIVE_SOLVE_S = 4.0           # re-solve for live feedback this often
LIVE_MAX_FRAMES = 300        # live solves use at most this many frames (evenly spaced)


def usable(gate):
    """Corners count when decoded well and sharp enough; single-tag ambiguity and view angle
    don't matter (the solve uses corners, not single-tag poses)."""
    def ok(o):
        return (o.tvec is not None and o.decision_margin >= gate.min_decision_margin
                and o.side_px >= gate.min_side_px and o.edge_width_px <= gate.max_edge_width_px)
    return ok


class LiveSolver:
    """Re-solves a thinned copy of the session in the background for the on-screen status."""

    def __init__(self, mapper: BundleMapper) -> None:
        self.mapper, self.result, self.busy = mapper, None, False

    def kick(self) -> None:
        if self.busy or len(self.mapper.frames) < 10:
            return
        self.busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            m = self.mapper
            frames = list(m.frames)
            step = max(1, len(frames) // LIVE_MAX_FRAMES)
            keep = set(range(0, len(frames), step))
            sub = BundleMapper(m.origin, m.sizes, m.acc)
            sub.frames = frames
            sub.obs = [o for o in list(m.obs) if o.frame in keep]
            self.result = sub.solve()
        except Exception as exc:              # live feedback only; the final solve reports
            log.debug("live solve failed: %s", exc)
        finally:
            self.busy = False


def draw(image: np.ndarray, observations, result: BundleResult | None, origin: int,
         fps: float, frames: int) -> None:
    for obs in observations:
        q = result.quality.get(obs.tag_id) if result else None
        if obs.tag_id == origin or (q is not None and q.ok):
            color, where = GREEN, "origin" if obs.tag_id == origin else f"OK {q.sigma_cm:.1f}cm"
        elif q is not None:
            color, where = YELLOW, q.reasons[0]
        else:
            color, where = YELLOW if obs.tvec is not None else RED, "needs another tag in view"
        pts = obs.corners.round().astype(np.int32)
        cv2.polylines(image, [pts], True, color, 2)
        x, y = pts.min(axis=0)
        text = f"id {obs.tag_id} {where}"
        cv2.putText(image, text, (int(x), max(int(y) - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4)
        cv2.putText(image, text, (int(x), max(int(y) - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2)
    ok = sorted(t for t, q in (result.quality.items() if result else []) if q.ok)
    todo = sorted(t for t, q in (result.quality.items() if result else []) if not q.ok)
    text = f"{fps:4.1f} fps  {frames} linked frames  OK {ok}  still need {todo}   q = finish"
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def map_loop(video: VideoReceiver, detector: TagDetector, mapper: BundleMapper, use,
             K: np.ndarray, dist: np.ndarray, record: Path) -> None:
    live = LiveSolver(mapper)
    writer = None
    last_id, n, frames, t0, fps, last_solve = 0, 0, 0, time.monotonic(), 0.0, 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame, frame_id, _ = video.get_latest()
            if frame is not None and frame_id != last_id:
                last_id, frames = frame_id, frames + 1
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(record), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
                writer.write(frame)
                observations = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                mapper.add_frame(observations, K, dist, index=n, usable=use)
                n += 1
                now = time.monotonic()
                if now - last_solve > LIVE_SOLVE_S:
                    live.kick()
                    last_solve = now
                image = frame.copy()
                draw(image, observations, live.result, mapper.origin, fps, len(mapper.frames))
                cv2.imshow(WINDOW, image)
            now = time.monotonic()
            if now - t0 >= 1.0:
                fps, frames, t0 = frames / (now - t0), 0, now
            if (cv2.waitKey(1) & 0xFF == ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
    finally:
        if writer is not None:
            writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", required=True, help="drone name in config/fleet.yaml")
    parser.add_argument("--tags", type=Path, default=CONFIG_DIR / "tags.yaml")
    parser.add_argument("--camera", type=Path, help="default: config/camera/<drone>.yaml")
    parser.add_argument("--out", type=Path, default=CONFIG_DIR / "map.yaml")
    parser.add_argument("--fast-detect", action="store_true",
                        help="detector decimation from tags.yaml (default: full resolution, sharper corners)")
    parser.add_argument("--force", action="store_true", help="write the map even if a tag fails its checks")
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
    if not args.fast_detect:
        tags.detector["quad_decimate"] = 1.0
    detector = TagDetector(tags, K, dist)
    sizes = {t: tags.size_m(t) for t in tags.sizes_mm if tags.size_m(t)}
    mapper = BundleMapper(tags.origin_tag, sizes)
    DATA_DIR.mkdir(exist_ok=True)
    record = DATA_DIR / f"map_{args.drone}_{time.strftime('%Y%m%d_%H%M%S')}.avi"
    log.info("Origin tag %d. Show it with other tags to link them. 'q' to finish. Recording %s",
             tags.origin_tag, record)

    try:
        drone, local_ip = resolve_drone(args.drone)
        with verified_link(drone, local_ip) as link, video_stream(link, local_ip) as video:
            try:
                map_loop(video, detector, mapper, usable(tags.gate), K, dist, record)
            except KeyboardInterrupt:
                log.info("Ctrl+C received -> solving")
            finally:
                cv2.destroyAllWindows()
    except IdentityError:
        return 2
    except SessionError:
        return 1

    log.info("Final solve on %d linked frames (can take a minute)...", len(mapper.frames))
    res = mapper.solve()
    if res is None or tags.origin_tag not in res.tag_map.poses:
        log.error("Origin tag %d never seen with others; no map written. Video: %s",
                  tags.origin_tag, record)
        return 1
    log.info("\n%s", report(res))
    bad = [t for t, q in res.quality.items() if not q.ok]
    if bad and not args.force:
        log.error("tags %s failed their checks: NOT writing %s. Map again with more views of "
                  "them, or rebuild from this video with 07_map_from_video.py --video %s --force",
                  bad, args.out, record)
        return 1
    if args.out.exists():
        bak = args.out.with_name(args.out.name + time.strftime(".bak_%Y%m%d_%H%M%S"))
        bak.write_text(args.out.read_text())
        log.info("previous map backed up -> %s", bak)
    save_yaml(args.out, result_to_dict(res))
    log.info("wrote %s (%d tags in tag-%d frame). Video: %s", args.out, len(res.tag_map.poses),
             tags.origin_tag, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())

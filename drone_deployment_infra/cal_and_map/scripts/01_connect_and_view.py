"""Connection check for one Tello, then show its live camera feed.

The drone's identity (Wi-Fi SSID/BSSID, then sn?) must match config/fleet.yaml
before streamon is sent. Only allowlisted commands are used.
Press 'q' in the video window or Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from tello_calibration.comms.tello_link import TelloLink
from tello_calibration.session import (
    LOW_BATTERY_PCT,
    IdentityError,
    SessionError,
    resolve_drone,
    verified_link,
    video_stream,
)
from tello_calibration.video.stream import VideoReceiver

log = logging.getLogger("connect_and_view")

WINDOW = "Tello"


def draw_overlay(image: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        y = 28 + 28 * i
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def view_loop(link: TelloLink, video: VideoReceiver) -> None:
    assert link.state is not None
    last_id = 0
    fps, fps_frames, fps_t0 = 0.0, 0, time.monotonic()
    low_battery_warned = False
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    while True:
        frame, frame_id, t_recv = video.get_latest()
        now = time.monotonic()
        if frame is not None and frame_id != last_id:
            fps_frames += frame_id - last_id
            last_id = frame_id
            bat = link.state.latest().values.get("bat")
            if isinstance(bat, float) and bat < LOW_BATTERY_PCT and not low_battery_warned:
                log.warning("Battery low: %.0f%%", bat)
                low_battery_warned = True
            image = frame.copy()
            draw_overlay(image, [
                f"FPS: {fps:4.1f}",
                f"Battery: {bat:.0f}%" if isinstance(bat, float) else "Battery: ?",
                f"Frames: {frame_id}",
                f"Age: {(now - t_recv) * 1000:3.0f} ms" if t_recv else "",
            ])
            cv2.imshow(WINDOW, image)

        if now - fps_t0 >= 1.0:
            fps = fps_frames / (now - fps_t0)
            fps_frames, fps_t0 = 0, now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("'q' pressed")
            return
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            log.info("Window closed")
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", required=True,
                        help="drone name in config/fleet.yaml, e.g. tello_4")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        drone, local_ip = resolve_drone(args.drone)
        with verified_link(drone, local_ip) as link, video_stream(link, local_ip) as video:
            log.info("Showing video. Press 'q' in the window or Ctrl+C to stop.")
            try:
                view_loop(link, video)
            finally:
                cv2.destroyAllWindows()
    except IdentityError:
        return 2
    except SessionError:
        return 1
    except KeyboardInterrupt:
        log.info("Ctrl+C received")
    log.info("Clean exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())

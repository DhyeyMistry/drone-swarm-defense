#!/home/jaimeromero/action-space/.venv/bin/python
"""Fly the quick-connect drone(s) to the origin (tag 10), using the MAP, land at --range.

Which drones fly comes from config/fleet.yaml `quick_connect` (whichever have their
stick plugged in): one listed and present -> one flies; two -> both fly, each at its own
cruise height so their paths don't cross. They connect and wait for each other before
any takes off (core/test_link.py).

This drives off the mapping/localization pipeline (core/positional.py + config/map.yaml),
not a single tag. Each frame it computes the drone's position in the tag-10 frame from
ANY mapped tag(s) in view (5, 6, 7, 10, ...), yaws to face the origin and flies toward
it, climbing/descending to keep it level. Covering any one tag (even tag 10) does not
stop it. Lands within --range of the origin. Records every rc command + pose.

Uses the engine's stick-pinned sockets (SO_BINDTODEVICE), so it works on a dongle that
isn't the default route -- unlike djitellopy, which can't pin an interface.
KEEP THE AREA CLEAR AND BE READY TO CATCH IT.

  ./tello_to_origin.py                    # drones from fleet.yaml quick_connect, 1.5 m
  ./tello_to_origin.py --range 1.5 --seconds 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

DD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DD_ROOT / "core"))

from test_link import DroneLink, connect_all, land  # noqa: E402
from tello_dual_video import RECORD_DIR, sdk, send_rc  # noqa: E402
from positional import Positioner  # noqa: E402

# Control gains / limits (rc units, -100..100). Gentle on purpose.
K_YAW_DEG, MAX_YAW = 1.6, 20      # yaw per degree of bearing to the origin
K_UD, MAX_UD = 60.0, 25          # up/down per unit of vertical direction
K_FWD, MAX_FWD, MIN_FWD = 35.0, 25, 10
FACE_TOL_DEG = 12                # must be roughly facing the origin to land
H_GAP = 0.5                      # cruise-height separation per drone (collision avoidance)


def clamp(value: float, limit: int) -> int:
    return int(max(-limit, min(limit, value)))


def fly(dl: DroneLink, rng: float, hold_h: float | None, seconds: float,
        stop: threading.Event, barrier: threading.Barrier, idx: int) -> None:
    name = dl.name
    positioner = Positioner(name.lower().replace("-", "_"))
    print(f"{name} map tags: {positioner.tag_map.mapped()}", flush=True)
    out = Path(RECORD_DIR)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"{dl.session}_{name}.train.jsonl", "w", encoding="utf-8")

    print(f"{name} waiting for the others at the start line...", flush=True)
    try:
        barrier.wait(timeout=120)
    except threading.BrokenBarrierError:
        print(f"{name} aborting: another drone dropped out", flush=True)
        log.close()
        return
    time.sleep(idx * 0.6)
    print(f"{name} takeoff -> origin, land at {rng:.1f} m"
          + (f", hold {hold_h:.1f} m" if hold_h is not None else ""), flush=True)
    print(f"{name} takeoff {sdk(dl.cmd, 'takeoff', 15)}", flush=True)
    start = time.time()
    try:
        while not stop.is_set() and time.time() - start < seconds:
            now = time.time()
            if not dl.connected(now):
                try:
                    send_rc(dl.cmd, (0, 0, 0, 0))   # hold; don't fly blind on a dropped link
                except OSError:
                    pass
                stop.wait(0.05)
                continue
            frame = dl.frames.get(name)
            if frame is None:
                dl.ensure_video(stop)               # keep trying to bring the video up
                send_rc(dl.cmd, (0, 0, 0, 0))        # hold level until we can see a tag
                stop.wait(0.05)
                continue
            loc = positioner.locate(frame)
            h = dl.heights.get(name)
            if loc is None:
                send_rc(dl.cmd, (0, 0, 0, 0))        # no mapped tag in view -> hold
                log.write(json.dumps({"t": round(now - start, 2), "rc": [0, 0, 0, 0],
                                      "pos": None, "h": h, "mode": "search"}) + "\n")
                log.flush()
                stop.wait(0.05)
                continue
            rng_now = loc["range"]
            vx, vy, vz = loc["to_origin_cam"]        # camera frame: x right, y down, z forward
            bearing_deg = math.degrees(math.atan2(vx, vz))   # + = origin is to the right
            if rng_now <= rng and abs(bearing_deg) < FACE_TOL_DEG:
                print(f"{name} within {rng_now:.2f} m, facing origin -> landing (tags {loc['tags']})",
                      flush=True)
                break
            yaw = clamp(bearing_deg * K_YAW_DEG, MAX_YAW)
            # Hold a separation height when flying as a group; else track the origin's level.
            if hold_h is not None and h is not None:
                up = clamp((hold_h - h) * K_UD, MAX_UD)
            else:
                up = clamp(-vy * K_UD, MAX_UD)       # origin above -> climb
            error = rng_now - rng
            fwd = clamp(error * K_FWD, MAX_FWD) if error > 0 else 0
            if 0 < fwd < MIN_FWD:
                fwd = MIN_FWD
            if abs(bearing_deg) > 20:                # face the origin before committing forward
                fwd = min(fwd, MIN_FWD)
            rc = (0, fwd, up, yaw)
            send_rc(dl.cmd, rc)
            log.write(json.dumps({"t": round(now - start, 2), "rc": list(rc), "pos": loc["xyz"],
                                  "h": h, "range": round(rng_now, 2), "bearing": round(bearing_deg, 1),
                                  "tags": loc["tags"], "mode": "cruise"}) + "\n")
            log.flush()
            stop.wait(0.05)
    except KeyboardInterrupt:
        print(f"{name} Ctrl-C -> landing", flush=True)
    finally:
        try:
            send_rc(dl.cmd, (0, 0, 0, 0))
        except OSError:
            pass
        print(f"{name} land {land(dl.cmd)}", flush=True)
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", type=float, default=1.5, help="land this far from the origin, metres")
    parser.add_argument("--seconds", type=float, default=40.0, help="safety cap; land after this")
    args = parser.parse_args()

    stop = threading.Event()
    links = connect_all(stop)          # fleet.yaml quick_connect, present sticks, all connected
    n = len(links)
    barrier = threading.Barrier(n)
    print(f"to-origin: {[dl.name for dl in links]}; stop at {args.range:.1f} m. "
          f"Be ready to catch them.", flush=True)

    threads = []
    for i, dl in enumerate(links):
        # One drone: track the origin's own level (hold_h=None). Several: stack heights
        # so the lanes stay apart on the way in.
        hold_h = None if n == 1 else 0.9 + i * H_GAP
        t = threading.Thread(target=fly,
                             args=(dl, args.range, hold_h, args.seconds, stop, barrier, i),
                             daemon=True)
        threads.append(t)
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Ctrl-C -> stopping", flush=True)
    finally:
        stop.set()
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

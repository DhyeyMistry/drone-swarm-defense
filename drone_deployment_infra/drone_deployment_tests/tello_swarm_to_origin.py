#!/home/jaimeromero/action-space/.venv/bin/python
"""Send the quick-connect drones across the room, each to its OWN landing spot.

Which drones fly comes from config/fleet.yaml `quick_connect` (whichever of those have
their stick plugged in) -- set that list to the drones you want; this script connects
to all of them and waits for every one before any takes off (core/test_link.py).

Each drone localizes itself in the tag-10 map (core/positional.py, from ANY mapped tag
it sees), climbs to its OWN cruise height, flies to its OWN landing spot 1.5 m from
tag 10, and lands. The spots are spread SPREAD_M apart along the tag's x axis and the
cruise heights are stacked BASE_H + i*H_GAP, so with any number of drones no two share
a spot and no two share a height -- their flight paths can't cross. Every rc command and
pose is recorded to recordings/<session>_<drone>.train.jsonl for replay in the simulation.

The spots/heights here are specific to THIS test, so they live in the script (not a
config file). KEEP THE AREA CLEAR AND BE READY TO CATCH THEM.

  ./tello_swarm_to_origin.py                 # drones from fleet.yaml quick_connect
  ./tello_swarm_to_origin.py --seconds 50
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

# --- This test's layout (was config/to_origin_swarm.yaml) -------------------------
STOP_RANGE_M = 1.5     # land this far (forward) from tag 10
LAND_TOL_M = 0.2       # within this of the target spot (horizontal) -> land
SPREAD_M = 0.5         # gap between adjacent landing spots along the tag's x axis
BASE_H = 0.9           # cruise height of the first drone, metres
H_GAP = 0.5            # extra cruise height per additional drone (keeps lanes apart)

# --- Control gains / limits (rc units, -100..100). Gentle. ------------------------
K_YAW, MAX_YAW = 1.4, 20       # yaw per degree of bearing; capped low so it won't spin off the tags
K_UP, MAX_UP = 60.0, 25        # up/down per metre of cruise-height error
K_H, MAX_H, MIN_H = 35.0, 22, 10   # horizontal speed toward the landing spot


def clamp(v: float, lim: int) -> int:
    return int(max(-lim, min(lim, v)))


def spot_for(idx: int, n: int) -> tuple[float, float]:
    """(lateral x, cruise height) for drone idx of n: spots centred and SPREAD_M apart."""
    lateral = (idx - (n - 1) / 2.0) * SPREAD_M
    return lateral, BASE_H + idx * H_GAP


def fly(dl: DroneLink, target_x: float, cruise_h: float, seconds: float,
        stop: threading.Event, barrier: threading.Barrier, idx: int) -> None:
    name = dl.name
    positioner = Positioner(name.lower().replace("-", "_"))  # map + camera (default.yaml fallback)
    out = Path(RECORD_DIR)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"{dl.session}_{name}.train.jsonl", "w", encoding="utf-8")

    # Barrier: no drone takes off until EVERY connected drone reaches here.
    print(f"{name} waiting for the others at the start line...", flush=True)
    try:
        barrier.wait(timeout=120)
    except threading.BrokenBarrierError:
        print(f"{name} aborting: another drone dropped out", flush=True)
        log.close()
        return
    time.sleep(idx * 0.6)   # tiny stagger so they don't lift off the same instant
    print(f"{name} all connected -> takeoff -> spot x {target_x:+.2f}, {STOP_RANGE_M:.1f} m, "
          f"height {cruise_h:.1f} m", flush=True)
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
                row = {"t": round(now - start, 2), "rc": [0, 0, 0, 0], "pos": None,
                       "h": h, "mode": "search"}
            else:
                px, py, pz = loc["xyz"]
                horiz_err = math.hypot(target_x - px, STOP_RANGE_M - pz)
                if horiz_err <= LAND_TOL_M:
                    print(f"{name} at spot ({horiz_err:.2f} m) -> landing (tags {loc['tags']})",
                          flush=True)
                    break
                d = positioner.direction_to(loc, (target_x, py, STOP_RANGE_M))  # camera-frame dir
                nh = math.hypot(d[0], d[2]) or 1.0
                mag = clamp(horiz_err * K_H, MAX_H)
                if 0 < mag < MIN_H:
                    mag = MIN_H
                right = int(d[0] / nh * mag)
                fwd = int(d[2] / nh * mag)
                bearing = math.degrees(math.atan2(loc["to_origin_cam"][0], loc["to_origin_cam"][2]))
                yaw = clamp(bearing * K_YAW, MAX_YAW)
                up = clamp((cruise_h - h) * K_UP, MAX_UP) if h is not None else 0
                rc = (right, fwd, up, yaw)
                send_rc(dl.cmd, rc)
                row = {"t": round(now - start, 2), "rc": list(rc), "pos": loc["xyz"],
                       "h": h, "horiz_err": round(horiz_err, 2), "tags": loc["tags"], "mode": "cruise"}
            log.write(json.dumps(row) + "\n")
            log.flush()
            stop.wait(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"{name} land {land(dl.cmd)}", flush=True)
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=50.0, help="per-drone safety cap")
    args = parser.parse_args()

    stop = threading.Event()
    links = connect_all(stop)          # fleet.yaml quick_connect, present sticks, all connected
    n = len(links)
    barrier = threading.Barrier(n)
    print(f"swarm to origin: {[dl.name for dl in links]}; session {links[0].session}. "
          f"Be ready to catch them.", flush=True)

    threads = []
    for i, dl in enumerate(links):
        target_x, cruise_h = spot_for(i, n)
        t = threading.Thread(target=fly,
                             args=(dl, target_x, cruise_h, args.seconds, stop, barrier, i),
                             daemon=True)
        threads.append(t)
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):   # the fly threads land themselves
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Ctrl-C -> stopping", flush=True)
    finally:
        stop.set()
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

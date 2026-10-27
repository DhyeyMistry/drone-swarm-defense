#!/usr/bin/env python3
"""A practice attacker: takes off facing tag 10 and shuttles toward / away from it, while
publishing its own position as repo fleet telemetry (exactly what drone_gui sends), so
defend.py can hunt it like a real attacker.

Motion: anchors where it first sees the tag after takeoff, then oscillates along the tag
normal between that spot and --amp_m closer to the tag (one cycle per --period_s),
holding its sideways position and height. Faces the tag throughout (to stay localized).

Telemetry: UDP JSON v1 to --feed (default 127.0.0.1:9100, defend.py's default), 10 Hz,
pose = camera in tag 10's frame in the README convention (x right, y down, z into the
wall) -- what drone_gui publishes with official tag prints. `fly` goes false once it is
down (landed or knocked to the floor), so the defender lets go.

Default is DRY RUN: connects, localizes and publishes, prints the rc it would send, no
takeoff -- carry it by hand to rehearse. --fly to fly.
Safety: tag lost > --lost_timeout_s, link drop, Ctrl-C or --seconds -> land.
KEEP THE AREA CLEAR AND BE READY TO CATCH IT.

  python dummy_attacker.py TELLO-4 --map home/map.yaml --tags home/tags.yaml            # dry run
  python dummy_attacker.py TELLO-4 --map home/map.yaml --tags home/tags.yaml --fly
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DD_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DD_ROOT / "core"))

from defend import RC_PER_MPS, facing_R, rc_from, velocity_cmd  # noqa: E402
from defense_core import Plan  # noqa: E402

FIX_MAX_AGE_S = 0.4
GROUND_H_M = 0.2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drone", help="e.g. TELLO-4")
    ap.add_argument("--map", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--fly", action="store_true", help="really take off (default: dry run)")
    ap.add_argument("--amp_m", type=float, default=0.5, help="how far it moves toward the tag")
    ap.add_argument("--period_s", type=float, default=6.0, help="one toward-and-back cycle")
    ap.add_argument("--speed_cap", type=float, default=0.3, help="m/s, at most 0.5")
    ap.add_argument("--min_dist_m", type=float, default=0.8, help="never closer to the tag wall")
    ap.add_argument("--feed", default="127.0.0.1:9100", help="publish telemetry here (comma list ok)")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--lost_timeout_s", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--climb_cm", type=int, default=30,
                    help="climb this much right after takeoff (SDK `up`, 20-500; 0 = off) so the tag "
                         "comes into view")
    args = ap.parse_args()
    if args.climb_cm and not 20 <= args.climb_cm <= 500:
        ap.error("--climb_cm must be 0 or 20..500")
    if not 0 < args.speed_cap <= 0.5:
        ap.error("--speed_cap must be in (0, 0.5]")
    dests = []
    for item in args.feed.split(","):
        host, _, port = item.strip().rpartition(":")
        dests.append((host, int(port)))

    import test_link
    from tello_dual_video import send_rc
    from tello_calibration.utils.config import CONFIG_DIR
    from upright_positional import UprightPositioner

    test_link.quick_connect_drones = lambda: [args.drone]
    stop = threading.Event()
    dl = test_link.connect_all(stop)[0]
    name = dl.name
    pos = UprightPositioner(name.lower().replace("-", "_"),
                            map_path=Path(args.map) if args.map else CONFIG_DIR / "map.yaml",
                            tags_path=Path(args.tags) if args.tags else None)
    face = np.array([0.0, 0.0, -1.0])
    rc_cap = min(45, int(round(args.speed_cap * RC_PER_MPS)))
    st = {"loc": None, "t": 0.0, "frames": -1}
    tel = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0

    def fix():
        frame = dl.frames.get(name)
        if frame is None:
            dl.ensure_video(stop)
            return
        n = dl.stats.get(name, {}).get("frames", 0)
        if n != st["frames"]:
            st["frames"] = n
            loc = pos.locate(frame)
            if loc is not None:
                st["loc"], st["t"] = loc, dl.stats.get(name, {}).get("view_t", time.time())

    def publish(airborne: bool, mode: str) -> None:
        nonlocal seq
        seq += 1
        # Dry run: the drone is in your hand, its height sensor reads ~0 -> report no `h` (and
        # fly=true) so defend.py takes the height from the tag and treats it as airborne.
        h = dl.heights.get(name) if args.fly else None
        pose = None
        if st["loc"] is not None:
            x, y, z = st["loc"]["xyz"]                 # map frame = tag 10 frame (x right, y up, z out)
            pose = {"tag": 10, "xyz": [round(x, 3), round(-y, 3), round(-z, 3)],   # README convention
                    "yaw": 0.0, "age": round(time.time() - st["t"], 2)}
        fly = airborne and (not args.fly or h is None or h >= GROUND_H_M)
        m = dl.motion.get(name) or {}
        msg = {"v": 1, "src": "dummy", "seq": seq, "t": time.time(), "hz": args.hz,
               "drones": [{"name": name, "slot": 1, "team": "attack", "link": dl.connected(),
                           "fly": fly, "mode": mode, "bat": dl.batteries.get(name), "h": h, "tof": None,
                           "vel": [m.get("right", 0.0), m.get("forward", 0.0), m.get("up", 0.0)],
                           "pose": pose}]}
        data = json.dumps(msg, separators=(",", ":")).encode()
        for d in dests:
            try:
                tel.sendto(data, d)
            except OSError:
                pass

    def send(rc) -> None:
        if args.fly:
            try:
                send_rc(dl.cmd, rc)
            except OSError:
                pass

    airborne = False
    anchor = None
    lost_since = None
    abort = False
    last_print = last_pub = 0.0
    print(f"{name} dummy attacker: amp {args.amp_m} m, period {args.period_s} s, rc cap {rc_cap}, "
          f"telemetry -> {dests}, {'FLY' if args.fly else 'DRY RUN (no takeoff)'}", flush=True)
    try:
        if args.fly:
            bat = test_link.sdk(dl.cmd, "battery?", 2)
            print(f"{name} battery {bat}%", flush=True)
            if not bat.isdigit() or int(bat) < 30:
                raise SystemExit(f"{name} battery {bat!r} < 30%: not flying")
            print(f"{name} takeoff -> {test_link.sdk(dl.cmd, 'takeoff', 15)}", flush=True)
            if args.climb_cm:
                print(f"{name} up {args.climb_cm} -> {test_link.sdk(dl.cmd, f'up {args.climb_cm}', 10)}", flush=True)
            airborne = True
        t0 = time.time()
        while not stop.is_set() and time.time() - t0 < args.seconds:
            now = time.time()
            fix()
            if now - last_pub >= 1.0 / args.hz:
                publish(airborne or not args.fly, "dummy")      # dry run: report "flying" so defend reacts
                last_pub = now
            if args.fly and not dl.connected(now):
                if airborne and not abort:
                    abort = True
                    print(f"{name} lost link in flight; landing on reconnect", flush=True)
                time.sleep(0.1)
                continue
            if abort:
                print(f"{name} link back -> landing", flush=True)
                break
            h_now = dl.heights.get(name)
            if args.fly and airborne and anchor is not None and h_now is not None and h_now < GROUND_H_M:
                print(f"{name} is on the ground (knocked down?) -> done", flush=True)
                break
            fresh = st["loc"] is not None and now - st["t"] <= FIX_MAX_AGE_S
            if not fresh:
                lost_since = lost_since or now
                if now - lost_since > args.lost_timeout_s:
                    print(f"{name} no tag for {args.lost_timeout_s:.0f} s -> landing", flush=True)
                    break
                send((0, 0, 0, 0))
                time.sleep(0.05)
                continue
            lost_since = None
            p = np.asarray(st["loc"]["xyz"], float)
            if anchor is None:
                anchor = p.copy()
                print(f"{name} anchored at {np.round(anchor, 2).tolist()} (x, y, z m in tag 10's frame)",
                      flush=True)
            phase = 0.5 - 0.5 * math.cos(2 * math.pi * (now - t0) / args.period_s)   # 0 -> 1 -> 0
            target = anchor.copy()
            target[2] = max(args.min_dist_m, anchor[2] - args.amp_m * phase)
            rc = rc_from(velocity_cmd(Plan(name, "hold", target), p, args.speed_cap),
                         np.asarray(st["loc"]["R"], float), face, rc_cap)
            send(rc)
            if now - last_print > (0.25 if not args.fly else 1.0):
                print(f"{name} {'DRY ' if not args.fly else ''}rc {rc}  pos {np.round(p, 2).tolist()}  "
                      f"target z {target[2]:.2f}", flush=True)
                last_print = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"{name} Ctrl-C -> landing", flush=True)
    finally:
        if args.fly and airborne:
            send((0, 0, 0, 0))
            print(f"{name} land -> {test_link.land(dl.cmd)}", flush=True)
        for _ in range(3):                              # tell defend.py we're down
            publish(False, "ground")
            time.sleep(0.05)
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())

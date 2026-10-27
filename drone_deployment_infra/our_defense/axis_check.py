#!/usr/bin/env python3
"""Does the drone move where we tell it? Measure drift and each stick's real effect with the tag.

Takes off facing tag 10, then flies a fixed sequence of short pulses, each followed by a
hover: hover (zero sticks = pure DRIFT), forward, back, right, left, up, down. For every
segment it measures how far the drone actually moved, from the tag, in the drone's own
axes, and prints a verdict:

    segment   rc            moved (right, fwd, up) m   speed m/s   verdict
    hover     ( 0, 0,0,0)   (-0.12, -0.30, +0.01)      0.11        DRIFT 0.11 m/s -> calibrate IMU / light / floor
    forward   ( 0,30,0,0)   (+0.02, +0.33, -0.01)      0.22        ok

Yaw is held on the tag the whole time (so the camera keeps seeing it). Won't take off
unless the tag is seen steadily first. Lost tag -> hover; lost > 3 s or Ctrl-C -> land. Stops a pulse early if the drone gets within --min_dist_m
of the tag wall. KEEP THE AREA CLEAR AND BE READY TO CATCH IT.

  python axis_check.py TELLO-2 --map home/map.yaml --tags home/tags.yaml
  python axis_check.py TELLO-2 --map home/map.yaml --tags home/tags.yaml --rc 40 --pulse_s 1.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DD_ROOT = HERE.parent
sys.path.insert(0, str(DD_ROOT / "core"))

DRIFT_WARN_MPS = 0.06
LOST_LAND_S = 3.0           # no tag this long at any point in flight -> land
PREFLIGHT_FIXES = 15        # tag fixes needed in PREFLIGHT_S before we take off at all
PREFLIGHT_S = 4.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drone", help="e.g. TELLO-2")
    ap.add_argument("--map", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--rc", type=int, default=30, help="stick value for each pulse (<= 50)")
    ap.add_argument("--pulse_s", type=float, default=1.5)
    ap.add_argument("--hover_s", type=float, default=3.0, help="zero-stick drift measurement")
    ap.add_argument("--min_dist_m", type=float, default=0.8, help="never closer to the tag wall")
    ap.add_argument("--climb_cm", type=int, default=30,
                    help="climb this much right after takeoff (SDK `up`, 20-500; 0 = off) so the tag "
                         "comes into view")
    args = ap.parse_args()
    if args.climb_cm and not 20 <= args.climb_cm <= 500:
        ap.error("--climb_cm must be 0 or 20..500")
    if not 0 < args.rc <= 50:
        ap.error("--rc must be in 1..50")

    import test_link
    sys.path.insert(0, str(HERE))
    from upright_positional import UprightPositioner as Positioner   # print-orientation proof
    from tello_dual_video import send_rc
    from tello_calibration.utils.config import CONFIG_DIR

    test_link.quick_connect_drones = lambda: [args.drone]
    stop = threading.Event()
    dl = test_link.connect_all(stop)[0]
    pos = Positioner(dl.name.lower().replace("-", "_"),
                     map_path=Path(args.map) if args.map else CONFIG_DIR / "map.yaml",
                     tags_path=Path(args.tags) if args.tags else None)
    state = {"loc": None, "t": 0.0, "frames": -1, "lost_t": None}

    def fix():
        frame = dl.frames.get(dl.name)
        if frame is None:
            dl.ensure_video(stop)
            return None
        n = dl.stats.get(dl.name, {}).get("frames", 0)
        if n != state["frames"]:
            state["frames"] = n
            loc = pos.locate(frame)
            if loc is not None:
                state["loc"], state["t"] = loc, time.time()
        return state["loc"] if time.time() - state["t"] < 0.4 else None

    def yaw_hold(loc):
        if loc is None:
            return 0
        vx, _vy, vz = loc["to_origin_cam"]
        return int(max(-20, min(20, 1.5 * math.degrees(math.atan2(vx, vz)))))

    def fly_for(rc, seconds, label):
        """Send rc (with yaw held on the tag) for `seconds`; return the fixes seen."""
        fixes, t0 = [], time.time()
        while time.time() - t0 < seconds and not stop.is_set():
            loc = fix()
            if loc is None:
                # ONE clock for the whole flight: segments are short, a per-segment timer
                # never fires (that is how it hovered blind into a wall).
                state["lost_t"] = state["lost_t"] or time.time()
                if time.time() - state["lost_t"] > LOST_LAND_S:
                    raise RuntimeError(f"tag lost for {LOST_LAND_S:.0f} s -> landing")
                send_rc(dl.cmd, (0, 0, 0, 0))
            else:
                state["lost_t"] = None
                fixes.append((time.time(), np.asarray(loc["xyz"], float), np.asarray(loc["R"], float)))
                sticks = list(rc)
                if loc["xyz"][2] < args.min_dist_m and sticks[1] > 0:
                    print(f"  {label}: {loc['xyz'][2]:.2f} m from the wall -> stopping the pulse", flush=True)
                    sticks = [0, 0, 0, 0]
                    seconds = 0
                sticks[3] = yaw_hold(loc)
                send_rc(dl.cmd, tuple(sticks))
            time.sleep(0.05)
        return fixes

    def moved(fixes):
        """Displacement in the drone's axes at the start (right, forward, up) and duration."""
        if len(fixes) < 4:
            return None, 0.0
        k = max(2, len(fixes) // 5)
        p0 = np.median([f[1] for f in fixes[:k]], axis=0)
        p1 = np.median([f[1] for f in fixes[-k:]], axis=0)
        R = fixes[0][2]
        d = R.T @ (p1 - p0)                              # camera frame: x right, y down, z forward
        return np.array([d[0], d[2], -d[1]]), fixes[-1][0] - fixes[0][0]

    rc = args.rc
    plan = [("hover", (0, 0, 0, 0), args.hover_s, None),
            ("forward", (0, rc, 0, 0), args.pulse_s, 1), ("hover", (0, 0, 0, 0), 2.0, None),
            ("back", (0, -rc, 0, 0), args.pulse_s, -1), ("hover", (0, 0, 0, 0), 2.0, None),
            ("right", (rc, 0, 0, 0), args.pulse_s, 0), ("hover", (0, 0, 0, 0), 2.0, None),
            ("left", (-rc, 0, 0, 0), args.pulse_s, 0), ("hover", (0, 0, 0, 0), 2.0, None),
            ("up", (0, 0, rc, 0), args.pulse_s, 2), ("hover", (0, 0, 0, 0), 2.0, None),
            ("down", (0, 0, -rc, 0), args.pulse_s, 2), ("hover", (0, 0, 0, 0), args.hover_s, None)]
    expect = {"forward": (1, +1), "back": (1, -1), "right": (0, +1), "left": (0, -1), "up": (2, +1), "down": (2, -1)}

    out = DD_ROOT / "recordings" / f"axis_check_{dl.name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = open(out, "w", encoding="utf-8")
    print(f"battery {test_link.sdk(dl.cmd, 'battery?', 2)}%. Taking off; be ready to catch it.", flush=True)
    results = []
    # Preflight: on the floor the camera usually can't see the tag (it is above the frame),
    # so only the VIDEO must be live here. The tag must appear within LOST_LAND_S of takeoff
    # (the flight-wide lost clock in fly_for), else we land at once.
    n_frames0, t0 = None, time.time()
    while time.time() - t0 < PREFLIGHT_S + 6.0:
        fix()
        n = dl.stats.get(dl.name, {}).get("frames", 0)
        n_frames0 = n if n_frames0 is None else n_frames0
        if n - n_frames0 >= PREFLIGHT_FIXES:
            break
        time.sleep(0.05)
    if dl.stats.get(dl.name, {}).get("frames", 0) - (n_frames0 or 0) < PREFLIGHT_FIXES:
        stop.set()
        log.close()
        raise SystemExit("NOT TAKING OFF: no live video from the drone (streamon failed?).")
    seen = "tag already in view" if fix() is not None else "tag not in view from the floor (normal)"
    print(f"preflight ok: video live, {seen}. After takeoff it must see the tag within "
          f"{LOST_LAND_S:.0f} s or it lands.", flush=True)
    try:
        print(f"takeoff -> {test_link.sdk(dl.cmd, 'takeoff', 15)}", flush=True)
        if args.climb_cm:
            print(f"{dl.name} up {args.climb_cm} -> {test_link.sdk(dl.cmd, f'up {args.climb_cm}', 10)}", flush=True)
        fly_for((0, 0, 0, 0), 2.0, "settle")
        for label, sticks, seconds, _ in plan:
            fixes = fly_for(sticks, seconds, label)
            d, dur = moved(fixes)
            if d is None:
                verdict, speed = "NO TAG during this segment", None
            else:
                speed = float(np.linalg.norm(d)) / max(dur, 1e-3)
                if label == "hover":
                    verdict = (f"DRIFT {speed:.2f} m/s: calibrate IMU (Tello app), more light, textured floor"
                               if speed > DRIFT_WARN_MPS else "ok (holds still)")
                else:
                    axis, sign = expect[label]
                    along = d[axis] * sign
                    other = float(np.linalg.norm(np.delete(d, axis)))
                    if along <= 0.03:
                        verdict = "WRONG WAY / no motion"
                    elif other > along:
                        verdict = "moved, but mostly sideways (drift?)"
                    else:
                        verdict = f"ok ({along / dur:.2f} m/s along the axis)"
            row = {"segment": label, "rc": list(sticks), "moved_rfu_m": None if d is None else np.round(d, 3).tolist(),
                   "seconds": round(dur, 2), "speed": None if speed is None else round(speed, 3), "verdict": verdict}
            results.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"  {label:8s} rc {str(tuple(sticks)):16s} moved (right,fwd,up) "
                  f"{'--' if d is None else np.round(d, 2).tolist()!s:22s} {verdict}", flush=True)
    except (KeyboardInterrupt, RuntimeError) as exc:
        print(f"stopping: {exc or 'Ctrl-C'}", flush=True)
    finally:
        try:
            send_rc(dl.cmd, (0, 0, 0, 0))
        except OSError:
            pass
        print(f"land -> {test_link.land(dl.cmd)}", flush=True)
        log.close()
        stop.set()
    print(f"\nlog {out}", flush=True)
    drift = [r for r in results if r["segment"] == "hover" and r["speed"] is not None]
    if drift:
        print(f"mean zero-stick drift {np.mean([r['speed'] for r in drift]):.2f} m/s "
              f"(defend.py's --speed_cap must clearly beat this)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

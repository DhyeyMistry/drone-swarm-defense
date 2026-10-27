#!/usr/bin/env python3
"""How fast does the flight engine really send rc? Measured under a real flight's CPU load.

core/tello_dual_video.py paces its control loop with ``stop.wait(0.05)`` *after* each
tick's work, so the rc rate is 1 / (50 ms + tick work), and the work competes for the GIL
with the video decoders, the MJPG recorders and the AprilTag detector of every drone.
No drone needed: this replays recorded flights to rebuild that load, per drone:

  decode    a recording read at the Tello's 30 fps (cv2), into the shared frame slot
  record    every frame re-encoded to MJPG, like _record_loop
  detect    pupil-apriltags with the engine's settings (nthreads=2, quad_decimate=1.0)
  control   the engine's tick body (ioctl wifi check, pose lookup, JSON row + flush, a
            print every 0.5 s) followed by the same stop.wait(0.05), stamping each rc send

and reports the rc period distribution and how often a fresh tag pose is ready. Run it on the laptop that flies:

    ./rc_rate_bench.py                       # 2 drones, 60 s, recordings from ../recordings
    ./rc_rate_bench.py --drones 4 --seconds 120
    ./rc_rate_bench.py --period 0.05 --paced # what a deadline-paced loop would achieve

Writes outputs/rc_rate_<stamp>.json (tick times) for scripts/fit_tello_motion.py --timing.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def wifi_check(iface: str) -> bool:
    """The engine's has_tello_lan(): one socket + SIOCGIFADDR ioctl per tick."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", iface[:15].encode()))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def decoder(path: str, slot: dict, key: str, recorder_q: list, stop: threading.Event) -> None:
    cap = cv2.VideoCapture(path)
    period = 1.0 / 30.0
    nxt = time.time()
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        slot[key] = frame
        recorder_q.append(frame)
        nxt += period
        delay = nxt - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            nxt = time.time()


def recorder(out_path: str, q: list, stop: threading.Event) -> None:
    writer = None
    while not stop.is_set():
        if not q:
            time.sleep(0.005)
            continue
        frame = q.pop(0)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
        writer.write(frame)
        del q[:-90]  # the engine's queue holds at most 90 frames
    if writer is not None:
        writer.release()


def detector(slot: dict, key: str, poses: dict, stop: threading.Event, done: list) -> None:
    from pupil_apriltags import Detector

    det = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    seen = None
    while not stop.is_set():
        frame = slot.get(key)
        if frame is None or frame is seen:
            stop.wait(0.03)
            continue
        seen = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = 871.0 * gray.shape[1] / 960.0
        tags = det.detect(gray, estimate_tag_pose=True, camera_params=(f, f, gray.shape[1] / 2, gray.shape[0] / 2), tag_size=1.0)
        poses[key] = {"n": len(tags), "t": time.time()}
        done.append(time.time())


def control(key: str, iface: str, poses: dict, log_path: str, ticks: list, stop: threading.Event,
            period: float, paced: bool) -> None:
    log = open(log_path, "w", encoding="utf-8")
    last_status = 0.0
    start = time.time()
    nxt = start
    while not stop.is_set():
        now = time.time()
        wifi_check(iface)
        hit = poses.get(key)
        fresh = hit is not None and now - hit["t"] <= 0.4
        sticks = (0, 8 if fresh else 0, 0, 0)
        ticks.append(time.time())  # the moment send_rc() would fire
        if now - last_status >= 0.5:
            print(f"{key} cmd={sticks} fresh={fresh}", flush=True)
            last_status = now
        log.write(json.dumps({"t": round(now - start, 3), "rc": list(sticks), "fresh": fresh}) + "\n")
        log.flush()
        if paced:  # deadline pacing: the period no longer includes the tick's own work
            nxt += period
            stop.wait(max(0.0, nxt - time.time()))
        else:
            stop.wait(period)  # what the engine does today
    log.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drones", type=int, default=2)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--period", type=float, default=0.05, help="the engine's stop.wait() (s)")
    p.add_argument("--paced", action="store_true", help="deadline-paced loop instead of wait-after-work")
    p.add_argument("--iface", default="lo", help="interface for the per-tick wifi ioctl")
    p.add_argument("--recordings", default=str(ROOT / "recordings"))
    a = p.parse_args()

    videos = sorted(glob.glob(os.path.join(a.recordings, "*.avi")), key=os.path.getsize, reverse=True)
    if not videos:
        raise SystemExit(f"no .avi recordings in {a.recordings}")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    tmp = out_dir / "rc_rate_tmp"
    tmp.mkdir(exist_ok=True)
    stop = threading.Event()
    slot, poses, ticks, fixes = {}, {}, {}, {}
    threads = []
    for i in range(a.drones):
        key = f"drone{i}"
        q: list = []
        ticks[key] = []
        fixes[key] = []
        threads += [
            threading.Thread(target=decoder, args=(videos[i % len(videos)], slot, key, q, stop), daemon=True),
            threading.Thread(target=recorder, args=(str(tmp / f"{key}.avi"), q, stop), daemon=True),
            threading.Thread(target=detector, args=(slot, key, poses, stop, fixes[key]), daemon=True),
            threading.Thread(target=control, args=(key, a.iface, poses, str(tmp / f"{key}.jsonl"), ticks[key], stop,
                                                   a.period, a.paced), daemon=True),
        ]
    print(f"rc rate bench: {a.drones} drone(s), {a.seconds:.0f} s, wait {a.period * 1000:.0f} ms "
          f"{'deadline-paced' if a.paced else 'after work (engine today)'}, cpu {os.cpu_count()}", flush=True)
    for t in threads:
        t.start()
    time.sleep(a.seconds)
    stop.set()
    for t in threads:
        t.join(timeout=3)

    report = {"drones": a.drones, "period": a.period, "paced": a.paced, "host": socket.gethostname(), "ticks": {}}
    print("\n== rc send period (ms) ==")
    for key, ts in ticks.items():
        dt = np.diff(np.array(ts[20:])) * 1000.0  # skip start-up
        report["ticks"][key] = ts
        print(f"  {key}: rate {1000.0 / dt.mean():.2f} Hz | median {np.median(dt):.1f}  p5 {np.percentile(dt, 5):.1f}  "
              f"p95 {np.percentile(dt, 95):.1f}  p99 {np.percentile(dt, 99):.1f}  max {dt.max():.1f}  "
              f"| >75 ms: {np.mean(dt > 75) * 100:.1f}%  ({len(dt)} ticks)")
    print("\n== tag detections finished per second (a fresh pose for the controller) ==")
    for key, ts in fixes.items():
        dt = np.diff(np.array(ts[10:]))
        report.setdefault("detections", {})[key] = ts
        print(f"  {key}: {1.0 / dt.mean():.1f} Hz (video arrives at 30 fps) | gap p95 {np.percentile(dt, 95) * 1000:.0f} ms")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"rc_rate_{stamp}.json"
    path.write_text(json.dumps(report))
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

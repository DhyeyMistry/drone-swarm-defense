#!/usr/bin/env python3
"""Show TELLO-3 and TELLO-4 video at the same time.

Both drones are 192.168.10.1. Each USB dongle is pinned with SO_BINDTODEVICE
so the two streams do not share one socket.
"""

import json
import os
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import av
import cv2
import numpy as np

# drone_deployment_infra root; add core/ and the tests/ (policies) dir to the path so
# the flat imports below resolve wherever this is launched from.
from pathlib import Path as _Path

DD_ROOT = _Path(__file__).resolve().parent.parent
for _p in (DD_ROOT / "core", DD_ROOT / "drone_deployment_tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tello_dongle_setup import load_fleet, present_fleet
from tello_tags import TAG_SIZE_M as WORLD_TAG_SIZE_M
from tello_tags import camera_in_tag, load_map, origin_in_camera
from tello_policy import (
    LAND_RANGE_M,
    LANES,
    MIN_HEIGHT_M,
    LOCK_AFTER_S,
    LOCK_VERTICAL,
    SEARCH_CEILING_M,
    apply_yaw,
    cam_in_tag,
    make_controllers,
    predict_odom,
    predict_tag,
    track_rc,
    yaw_error_deg,
)

TELLO = "192.168.10.1"
# The fleet is whichever numbered dongles are plugged in (see present_fleet in
# tello_dongle_setup.py). Pass drone names as args to fly a fixed set instead.
# When False, a mid-flight signal loss makes the drone hold and then resume the
# flight once the link is back, instead of landing -- used to test reconnection.
LAND_ON_LOSS = True
# Low-bandwidth video commands sent once per (re)connect before streamon, to help
# the control link survive motor interference in flight. EMPTY here because this
# (non-EDU) Tello rejects them ("unknown command"); a Tello EDU accepts e.g.
# ("setresolution low", "setfps middle", "setbitrate 2").
VIDEO_SETUP = ()
# Lost-tag search: if the tag has not been seen for SEARCH_AFTER_S, stop and
# sweep yaw at SEARCH_YAW, flipping direction (left, then right) every SEARCH_LEG_S.
SEARCH_AFTER_S = 0.4
SEARCH_YAW = 25
SEARCH_LEG_S = 2.0
# Black square of tag 14. Focal length calibrated at 960 px wide.
TAG_ID = 14
TAG_SIZE_M = 0.995
CAL_F_AT_960 = 871.0
RECORD_DIR = str(DD_ROOT / "recordings")
OUTPUTS_DIR = str(DD_ROOT / "outputs")


def next_session():
    """Next 001, 002, ... from names already in the recordings folder."""
    nums = []
    if os.path.isdir(RECORD_DIR):
        for name in os.listdir(RECORD_DIR):
            head = name.split("_", 1)[0]
            if head.isdigit():
                nums.append(int(head))
    return f"{max(nums, default=0) + 1:03d}"
FRAME_DT = 1.0 / 30.0


def dev_socket(iface, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode())
    sock.bind(("0.0.0.0", port))
    return sock


def has_tello_lan(iface):
    """True when this dongle has a 192.168.10.x address from a Tello."""
    import fcntl
    import struct

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(
            sock.fileno(), 0x8915, struct.pack("256s", iface[:15].encode())
        )
    except OSError:
        return False
    finally:
        sock.close()
    return socket.inet_ntoa(packed[20:24]).startswith("192.168.10.")


def send_rc(sock, sticks):
    right, forward, up, yaw = sticks
    sock.sendto(f"rc {right} {forward} {up} {yaw}".encode(), (TELLO, 8889))


def _state_int(fields, key):
    try:
        return int(fields[key])
    except (KeyError, ValueError):
        return None


def listen_state(sock, name, heights, batteries, motion, link, stop):
    sock.settimeout(0.5)
    seen = 0.0
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            if seen and time.time() - seen > 1.0:
                heights[name] = None
                seen = 0.0
            continue
        # Every state packet is proof the control link is alive -- this is the
        # passive "connected" signal, replacing the blocking "command" probe.
        link["t"] = time.time()
        fields = {}
        for part in data.decode(errors="replace").split(";"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            fields[key] = value
        height_cm = _state_int(fields, "h")
        battery = _state_int(fields, "bat")
        vgx = _state_int(fields, "vgx")
        vgy = _state_int(fields, "vgy")
        vgz = _state_int(fields, "vgz")
        yaw = _state_int(fields, "yaw")
        if height_cm is not None:
            heights[name] = height_cm / 100.0
            seen = time.time()
        if battery is not None:
            batteries[name] = battery
        if None not in (vgx, vgy, vgz, yaw):
            # Speeds are cm/s. vgx right, vgy forward, vgz up.
            motion[name] = {
                "right": vgx / 100.0,
                "forward": vgy / 100.0,
                "up": vgz / 100.0,
                "yaw": yaw,
                "t": time.time(),
            }


def sdk(sock, cmd, timeout=3):
    # Drain buffered replies first so this command reads its own response, not a
    # stale one (fire-and-forget "command" probes can leave replies queued).
    sock.setblocking(False)
    try:
        while True:
            sock.recvfrom(256)
    except (BlockingIOError, OSError):
        pass
    sock.settimeout(timeout)
    sock.sendto(cmd.encode(), (TELLO, 8889))
    try:
        data, _ = sock.recvfrom(256)
        # Normalize case: some Tello firmware replies "OK", others "ok". All
        # callers compare against lowercase "ok", so fold it here once.
        return data.decode(errors="replace").strip().lower()
    except (socket.timeout, OSError):
        return "timeout"


def forward(video, local_port, stop):
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video.settimeout(0.5)
    while not stop.is_set():
        try:
            data, _ = video.recvfrom(2048)
        except socket.timeout:
            continue
        out.sendto(data, ("127.0.0.1", local_port))
    out.close()


def _record_loop(name, path, images, stats, stop):
    writer = None
    while not stop.is_set() or not images.empty():
        try:
            image = images.get(timeout=0.5)
        except queue.Empty:
            continue
        if writer is None:
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
            if not writer.isOpened():
                print(f"{name} record open failed", flush=True)
                return
            stats[name]["file"] = path
            print(f"{name} record {path}", flush=True)
        writer.write(image)
    if writer is not None:
        writer.release()


def decode(name, local_port, frames, stats, stop):
    url = f"udp://@127.0.0.1:{local_port}?overrun_nonfatal=1&fifo_size=5000000"
    try:
        container = av.open(url, timeout=(8, None))
    except av.FFmpegError as exc:
        print(f"{name} decode open failed: {exc}", flush=True)
        return
    os.makedirs(RECORD_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = stats[name].get("session") or "000"
    path = os.path.join(RECORD_DIR, f"{session}_{name}_{stamp}.avi")
    images = queue.Queue(maxsize=90)
    threading.Thread(
        target=_record_loop, args=(name, path, images, stats, stop), daemon=True
    ).start()
    prev = None
    try:
        for frame in container.decode(video=0):
            if stop.is_set():
                break
            image = frame.to_ndarray(format="bgr24")
            now = time.time()
            stats[name]["frames"] += 1
            if prev is not None:
                gap = now - prev
                if gap > FRAME_DT * 1.5:
                    stats[name]["drops"] += max(1, int(round(gap / FRAME_DT)) - 1)
            prev = now
            if name not in frames:
                print(f"{name} frame {image.shape[1]}x{image.shape[0]}", flush=True)
            frames[name] = image
            stats[name]["view_t"] = now
            try:
                images.put_nowait(image)
            except queue.Full:
                stats[name]["overflow"] += 1
    finally:
        container.close()


def report_text(stats, names, sync):
    lines = [f"both poses at once: {sync['count']}"]
    for name in names:
        item = stats[name]
        lines.append(
            f"{name}\n"
            f"  file: {item['file'] or 'none'}\n"
            f"  frames: {item['frames']}\n"
            f"  dropped: {item['drops']}\n"
            f"  recorder behind: {item['overflow']}\n"
            f"  tag distance: {item['tag_hit']}\n"
            f"  no tag: {item['tag_miss']}"
        )
    return "\n".join(lines) + "\n"


def write_report(stats, names, sync, stop):
    os.makedirs(RECORD_DIR, exist_ok=True)
    session = next(iter(stats.values())).get("session") or "000"
    path = os.path.join(RECORD_DIR, f"{session}_report.txt")
    while not stop.is_set():
        text = report_text(stats, names, sync)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
        stop.wait(2)


def log_track(session, name, poses, ready, modes, heights, stop):
    """Write a wall-clock track per drone: time, wifi/link state, pose, height.

    Unlike the frame-index reconstruction in scripts/extract_track.py, this is
    logged live, so WiFi connection and the drone's pose share one real-time
    axis and the flight visualization can line up signal drops with the path.
    """
    os.makedirs(RECORD_DIR, exist_ok=True)
    path = os.path.join(RECORD_DIR, f"{session}_{name}.live.jsonl")
    start = time.time()
    with open(path, "w", encoding="utf-8") as handle:
        print(f"{name} track {path}", flush=True)
        while not stop.is_set():
            now = time.time()
            row = {"t": round(now - start, 2), "wifi": bool(ready.get(name)), "mode": modes.get(name, "")}
            hit = poses.get(name)
            if hit and now - hit.get("t", 0) < 0.6:
                row["x"], row["y"], row["z"] = hit["x"], hit["y"], hit["z"]
                row["range_m"] = hit["range_m"]
                row["ids"] = [hit["id"]]
            height = heights.get(name)
            if height is not None:
                row["h"] = round(height, 2)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            stop.wait(0.1)


def _yaw_delta(prev, yaw):
    if prev is None:
        return 0.0
    delta = yaw - prev
    while delta > 180:
        delta -= 360
    while delta < -180:
        delta += 360
    return delta


def run_drone(name, iface, cmd, video, state, local_port, frames, poses, heights, batteries, motion, modes, estimates, stats, stop, takeoff_lock, flight, names, ready, play):
    threading.Thread(target=forward, args=(video, local_port, stop), daemon=True).start()
    link = {"t": 0.0}  # wall-clock of the last state packet; our passive link signal
    threading.Thread(
        target=listen_state, args=(state, name, heights, batteries, motion, link, stop), daemon=True
    ).start()
    decoder = None
    last_takeoff = 0.0
    last_mode = None
    memory = None
    rotation = None
    acquired = False
    landed = False
    flew = False
    saw_air = False
    low_since = None
    told_high = False
    seen_since = None
    held = None
    prev_sticks = (0, 0, 0, 0)
    prev_yaw = None
    prev_t = time.time()
    last_land = 0.0
    last_fresh_t = 0.0
    land_dist = LAND_RANGE_M  # approach/land standoff; sighted landing happens here
    search_since = None       # set while sweeping yaw to re-find a lost tag
    # airborne: we commanded takeoff and have not landed. abort: the link dropped
    # while airborne, so once it is back we land and never take off again.
    airborne = False
    abort = False
    last_status = 0.0
    pids = make_controllers()
    # Training data: one row per control tick -- the action (rc) and the drone's
    # measured motion/state -- to fit an MLP of action -> how the drone moves.
    run_start = time.time()
    os.makedirs(RECORD_DIR, exist_ok=True)
    train_path = os.path.join(RECORD_DIR, f"{stats[name].get('session', '000')}_{name}.train.jsonl")
    train_log = open(train_path, "w", encoding="utf-8")
    print(f"{name} training data {train_path}", flush=True)
    while not stop.is_set():
        now = time.time()
        wifi = has_tello_lan(iface)
        # Connected = a state packet arrived in the last 1.5 s. No blocking
        # "command" probe in the control loop -- that stall every 5 s is what
        # dropped the link. The link is read passively from the 8890 broadcast.
        connected = wifi and (now - link["t"] < 1.5)
        if not connected:
            ready[name] = False
            if wifi:
                # Re-enter SDK mode (fire-and-forget so it never blocks the loop);
                # success shows up as state packets resuming on 8890.
                try:
                    cmd.sendto(b"command", (TELLO, 8889))
                except OSError:
                    pass
            if LAND_ON_LOSS and airborne and not landed and not abort:
                abort = True
                print(f"{name} lost signal in flight; will land on reconnect", flush=True)
            mode = "wifi" if not wifi else "link"
            modes[name] = mode
            if mode != last_mode:
                where = f"waiting for wifi on {iface}" if not wifi else "link quiet, re-entering SDK"
                print(f"{name} {where}", flush=True)
                last_mode = mode
            try:
                send_rc(cmd, (0, 0, 0, 0))  # gentle hold; a brief blip won't trip the failsafe
            except OSError:
                pass
            stop.wait(0.1)
            continue
        ready[name] = True
        if last_mode in ("wifi", "link"):
            print(f"{name} connected", flush=True)
            last_mode = None
        if decoder is None or not decoder.is_alive():
            for setup in VIDEO_SETUP:
                print(f"{name} {setup} -> {sdk(cmd, setup)}", flush=True)
            started = sdk(cmd, "streamon")
            print(f"{name} streamon {started}", flush=True)
            if started == "ok":
                decoder = threading.Thread(
                    target=decode, args=(name, local_port, frames, stats, stop), daemon=True
                )
                decoder.start()
        if play:
            modes[name] = "play"
            if last_mode != "play":
                print(f"{name} policy play, no launch", flush=True)
                last_mode = "play"
            stop.wait(0.1)
            continue
        hit = poses.get(name)
        age = None if not hit else now - hit["t"]
        fresh = (
            hit is not None
            and hit.get("xyz") is not None
            and hit.get("R") is not None
            and age is not None
            and age <= 0.4
        )
        dt = now - prev_t
        just_locked = False
        cam = None
        raw_height = heights.get(name)
        if raw_height is not None and raw_height >= 0.4:
            saw_air = True
            low_since = None
        elif flew and saw_air and raw_height is not None and raw_height < MIN_HEIGHT_M:
            if low_since is None:
                low_since = now
            if now - low_since >= 4:
                flew = False
        height_m = raw_height
        if flew and (raw_height is None or raw_height < MIN_HEIGHT_M) and not (saw_air and low_since is not None and now - low_since >= 4):
            height_m = 0.5
        if fresh:
            held = hit
            last_fresh_t = now
            if acquired:
                # Closed-loop: snap the estimate straight to this detection so the
                # controller never flies open-loop on drifting odometry. Odometry
                # only bridges the gaps between sightings (the block further down).
                memory = list(hit["xyz"])
                rotation = tuple(tuple(float(v) for v in row) for row in hit["R"])
                just_locked = True  # skip the odometry predict on this corrected tick
        tag_high = False
        if fresh and not acquired:
            _cx, cy = hit["center"]
            _w, h = hit["size"]
            vertical = (cy - h / 2.0) / (h / 2.0)
            # Negative vertical means the tag is above center. -1 is the top edge.
            tag_high = vertical < LOCK_VERTICAL and (height_m or 0.0) < SEARCH_CEILING_M
            if seen_since is None:
                seen_since = now
            if tag_high and not told_high:
                print(f"{name} tag high, climb", flush=True)
                told_high = True
        elif held is None or now - held["t"] >= 2.0:
            seen_since = None
        at_ceiling = (height_m or 0.0) >= SEARCH_CEILING_M
        held_recent = held is not None and now - held["t"] < 2.0
        held_long = fresh and seen_since is not None and now - seen_since >= LOCK_AFTER_S
        if not acquired and held_recent and held.get("R") is not None and (
            (fresh and not tag_high) or at_ceiling or held_long
        ):
            memory = list(held["xyz"])
            rotation = tuple(tuple(float(v) for v in row) for row in held["R"])
            acquired = True
            just_locked = True
            print(f"{name} tag lock, lane {LANES.get(name, 0.0):+.1f} m", flush=True)
        if acquired and memory is not None and rotation is not None and not just_locked:
            sample = motion.get(name)
            if sample is not None and now - sample["t"] < 0.5:
                dyaw = _yaw_delta(prev_yaw, sample["yaw"])
                memory = predict_odom(
                    memory,
                    (sample["right"], sample["forward"], sample["up"]),
                    dyaw,
                    dt,
                )
                prev_yaw = sample["yaw"]
            else:
                dyaw = (prev_sticks[3] / 100.0) * 60.0 * dt
                memory = predict_tag(memory, prev_sticks, dt)
            rotation = apply_yaw(rotation, dyaw)
        if acquired and memory is not None and rotation is not None:
            cam = cam_in_tag(rotation, memory)
        sticks, mode = track_rc(
            cam,
            yaw_error_deg(rotation) if rotation is not None and acquired else 0.0,
            height_m,
            acquired,
            LANES.get(name, 0.0),
            land_dist,
            pids,
            dt,
        )
        # Lost the tag: stop translating and sweep yaw (left, then right, ...) to
        # re-find it, instead of flying on drifting odometry.
        if acquired and not fresh and (last_fresh_t == 0.0 or now - last_fresh_t > SEARCH_AFTER_S):
            if search_since is None:
                search_since = now
                print(f"{name} lost tag, searching", flush=True)
            leg = int((now - search_since) // SEARCH_LEG_S)
            yaw = -SEARCH_YAW if leg % 2 == 0 else SEARCH_YAW  # left first, then right, ...
            sticks, mode = (0, 0, 0, yaw), "search"
            for pid in pids.values():
                pid.reset()
        else:
            search_since = None
        if cam is not None:
            estimates[name] = {
                "range_m": round(float(np.linalg.norm(cam)), 2),
                "perp_m": round(float(-cam[2]), 2),
                "x": round(float(cam[0]), 2),
                "y": round(float(cam[1]), 2),
                "z": round(float(cam[2]), 2),
                "mode": mode,
            }
        battery = batteries.get(name)
        eligible = (
            not flew
            and not airborne  # never re-launch mid-flight: the blocking takeoff stalls the rc stream
            and not landed
            and not abort
            and raw_height is not None
            and raw_height < MIN_HEIGHT_M
            and (battery is None or battery >= 15)
        )
        launch = False
        with takeoff_lock:
            if eligible:
                flight["slot"][name] = now
            else:
                flight["slot"].pop(name, None)
            slots_fresh = all(
                flight["slot"].get(drone) is not None and now - flight["slot"][drone] <= 0.5
                for drone in names
            )
            batteries_ok = all(
                (batteries.get(drone) is None or batteries.get(drone) >= 15) for drone in names
            )
            if slots_fresh and batteries_ok and now - flight["stamp"] > 20:
                flight["stamp"] = now
                flight["go_names"] = set(names)
                print("launch " + " ".join(names), flush=True)
            if name in flight["go_names"]:
                flight["go_names"].discard(name)
                launch = True
        # Link came back after an in-flight loss: land and stay down. If the
        # failsafe already set it down, just latch landed so it never relaunches.
        if abort and not landed:
            if raw_height is not None and raw_height < MIN_HEIGHT_M:
                landed = True
                airborne = False
            else:
                mode = "land"
        if landed:
            sticks, mode = (0, 0, 0, 0), "down"
        elif launch:
            print(f"{name} takeoff", flush=True)
            result = sdk(cmd, "takeoff", timeout=15)
            print(f"{name} takeoff {result}", flush=True)
            last_takeoff = time.time()
            if result == "ok":
                flew = True
                airborne = True
                low_since = None
            mode = "takeoff"
            sticks = (0, 0, 0, 0)
        elif mode == "land" and now - last_land > 8:
            sticks = (0, 0, 0, 0)
            print(f"{name} land", flush=True)
            result = sdk(cmd, "land", timeout=15)
            print(f"{name} land {result}", flush=True)
            last_land = time.time()
            if result == "ok":
                landed = True
                airborne = False
                mode = "down"
        elif not flew and raw_height is not None and raw_height < MIN_HEIGHT_M:
            if battery is not None and battery < 15:
                mode = "battery"
            else:
                mode = "sync"
            sticks = (0, 0, 0, 0)
        modes[name] = mode
        if mode not in ("takeoff", "land"):
            try:
                send_rc(cmd, sticks)
            except OSError:
                pass
        prev_sticks = sticks
        prev_t = time.time()
        if mode != last_mode:
            print(f"{name} policy {mode} rc {sticks}", flush=True)
            last_mode = mode
        if now - last_status >= 0.5:
            if fresh:
                tag = f"yes {hit['range_m']:.2f}m"
            elif acquired:
                tag = "search" if mode == "search" else "lock(odom)"
            else:
                tag = "no"
            print(f"{name} link={'up' if connected else 'down'} tag={tag} cmd={sticks} {mode}", flush=True)
            last_status = now
        # training row: action (rc) + the drone's measured response and state
        sample = motion.get(name)
        row = {
            "t": round(now - run_start, 3),
            "rc": list(sticks),
            "mode": mode,
            "fresh": bool(fresh),
            "h": None if height_m is None else round(float(height_m), 3),
        }
        if sample is not None and now - sample["t"] < 0.5:
            row["vel"] = {"r": sample["right"], "f": sample["forward"], "u": sample["up"], "yaw": sample["yaw"]}
        if fresh and hit is not None:
            row["tag"] = {"x": hit["x"], "y": hit["y"], "z": hit["z"], "range": hit["range_m"]}
        train_log.write(json.dumps(row) + "\n")
        train_log.flush()
        stop.wait(0.05)  # ~20 Hz steady rc stream (no blocking keepalive)


def watch_pose(name, frames, poses, stats, stop):
    from pupil_apriltags import Detector

    det = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    seen = None
    while not stop.is_set():
        frame = frames.get(name)
        if frame is None or frame is seen:
            if stop.wait(0.03):
                return
            continue
        seen = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        focal = CAL_F_AT_960 * (width / 960.0)
        # One detect at 1 m. Translation scales with the real black-square side.
        tags = det.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(focal, focal, width / 2.0, height / 2.0),
            tag_size=1.0,
        )
        tag_map = load_map()
        world_best = None
        legacy = None
        for tag in tags:
            if tag.pose_t is None:
                continue
            tag_id = int(tag.tag_id)
            rotation = np.asarray(tag.pose_R, dtype=float).reshape(3, 3)
            raw_t = np.asarray(tag.pose_t, dtype=float).reshape(3)
            margin = float(getattr(tag, "decision_margin", 0.0))
            if tag_id in WORLD_TAG_SIZE_M:
                origin = origin_in_camera(tag_id, rotation, raw_t * WORLD_TAG_SIZE_M[tag_id], tag_map)
                if origin is None:
                    if tag_id not in getattr(watch_pose, "unmapped", set()):
                        print(f"{name} tag {tag_id} seen, no map", flush=True)
                        watch_pose.unmapped = getattr(watch_pose, "unmapped", set()) | {tag_id}
                    continue
                if world_best is None or margin > world_best[0]:
                    world_best = (margin, tag_id, origin, tag)
            elif tag_id == TAG_ID:
                if legacy is None or margin > legacy[0]:
                    legacy = (margin, raw_t * TAG_SIZE_M, tag)
        hit = None
        chosen = world_best if world_best is not None else legacy
        if chosen is not None and world_best is not None:
            _margin, tag_id, (origin_r, origin_t), tag = world_best
            cam = camera_in_tag(origin_r, origin_t)
            hit = {
                "id": tag_id,
                "world": True,
                "range_m": round(float(np.linalg.norm(cam)), 2),
                "perp_m": round(float(-cam[2]), 2),
                "x": round(float(cam[0]), 2),
                "y": round(float(cam[1]), 2),
                "z": round(float(cam[2]), 2),
                "xyz": [round(float(v), 3) for v in origin_t],
                "R": [[float(origin_r[i, j]) for j in range(3)] for i in range(3)],
                "center": [float(tag.center[0]), float(tag.center[1])],
                "size": [int(width), int(height)],
                "t": time.time(),
            }
        elif chosen is not None:
            _margin, _scaled, tag = legacy
            translation = np.asarray(tag.pose_t, dtype=float).reshape(3) * TAG_SIZE_M
            rotation = np.asarray(tag.pose_R, dtype=float).reshape(3, 3)
            cam = camera_in_tag(rotation, translation)
            hit = {
                "id": TAG_ID,
                "world": False,
                "range_m": round(float(np.linalg.norm(cam)), 2),
                "perp_m": round(float(-cam[2]), 2),
                "x": round(float(cam[0]), 2),
                "y": round(float(cam[1]), 2),
                "z": round(float(cam[2]), 2),
                "xyz": [round(float(v), 3) for v in translation],
                "R": [[float(rotation[i, j]) for j in range(3)] for i in range(3)],
                "center": [float(tag.center[0]), float(tag.center[1])],
                "size": [int(width), int(height)],
                "t": time.time(),
            }
        if hit:
            prev = poses.get(name) or {}
            poses[name] = hit
            stats[name]["tag_hit"] += 1
            if abs(hit["range_m"] - prev.get("range_m", -1)) >= 0.05:
                where = " world" if hit.get("world") else ""
                print(f"{name} tag {hit['id']}{where} {hit['range_m']:.2f} m", flush=True)
        else:
            stats[name]["tag_miss"] += 1
            prev = poses.get(name)
            if prev and time.time() - prev["t"] > 0.4:
                poses[name] = None


def jpeg(frames, name):
    frame = frames.get(name)
    if frame is None:
        frame = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(frame, "waiting " + name, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else b""


def page(names, play):
    shown = tuple(reversed(names))
    height = "100vh" if len(shown) == 1 else "50vh"
    rows = []
    for name in shown:
        rows.append(
            f'<section class="pair"><img id="v-{name}" alt="{name}" src="/{name}.jpg">'
            f'<div class="dist"><div class="who">{name}</div>'
            f'<div class="m" id="m-{name}">—</div>'
            f'<div class="xyz" id="xyz-{name}">x —\ny —\nz —</div>'
            f'<div class="sub" id="s-{name}">tag {TAG_ID}</div></div></section>'
        )
    body = "".join(rows)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Tello</title>
<style>
body{{margin:0;background:#111;color:#eee;font-family:sans-serif}}
.pair{{display:flex;height:{height}}}
img{{width:62vw;height:{height};object-fit:contain;background:#000}}
.dist{{flex:1;display:flex;flex-direction:column;justify-content:center;padding:32px}}
.who{{font-size:28px;opacity:.7}}
.m{{font-size:8vw;font-variant-numeric:tabular-nums;line-height:1}}
.xyz{{font-size:32px;font-variant-numeric:tabular-nums;white-space:pre;line-height:1.35;margin-top:16px}}
.sub{{font-size:22px;opacity:.8;margin-top:12px}}
#report{{margin:0;padding:16px 24px;white-space:pre;font:16px/1.45 ui-monospace,monospace;background:#000}}
#banner{{position:sticky;top:0;z-index:2;padding:14px 24px;background:#222;font-size:28px}}
#banner.on{{background:#16351f}}
#rel{{padding:0 24px 12px;font-size:22px;background:#222}}
</style></head>
<body><div id="banner">{'PLAY · no launch' if play else 'flight'}</div><div id="rel"></div>{body}<pre id="report"></pre>
<script>
const names = {json.dumps(list(shown))};
async function tick() {{
  try {{
    const poses = await fetch('/pose').then(r => r.json());
    for (const name of names) {{
      const p = poses[name];
      const m = document.getElementById('m-' + name);
      const s = document.getElementById('s-' + name);
      const xyz = document.getElementById('xyz-' + name);
      if (!p) {{
        m.textContent = '—';
        xyz.textContent = 'x —\\ny —\\nz —';
        s.textContent = 'no tag {TAG_ID}';
        continue;
      }}
      m.textContent = p.range_m.toFixed(2) + ' m';
      xyz.textContent = 'x ' + p.x.toFixed(2) + ' m\\ny ' + p.y.toFixed(2) + ' m\\nz ' + p.z.toFixed(2) + ' m';
      s.textContent = 'tag ' + p.id + (p.world ? '  ·  tag 10 frame' : '') + '  ·  ' + p.perp_m.toFixed(2) + ' m off the face  ·  ' + (p.mode || '');
    }}
    const banner = document.getElementById('banner');
    const rel = document.getElementById('rel');
    const views = poses.views || {{}};
    const cams = names.filter(n => views[n]).length;
    if (poses.together) {{
      banner.textContent = 'BOTH HAVE A POSE';
      banner.className = 'on';
      const b = poses.between;
      rel.textContent = b ? ('between drones  ' + b.apart.toFixed(2) + ' m   dx ' + b.dx.toFixed(2) + '  dy ' + b.dy.toFixed(2) + '  dz ' + b.dz.toFixed(2)) : '';
    }} else {{
      banner.textContent = '{('PLAY · no launch · ' if play else '')}' + cams + '/' + names.length + ' cameras';
      banner.className = '';
      rel.textContent = '';
    }}
  }} catch (e) {{}}
}}
function frames() {{
  const stamp = Date.now();
  for (const name of names) {{
    const img = document.getElementById('v-' + name);
    if (!img) continue;
    const next = new Image();
    next.onload = function () {{ img.src = next.src; }};
    next.src = '/' + name + '.jpg?t=' + stamp;
  }}
}}
async function report() {{
  try {{
    document.getElementById('report').textContent = await fetch('/report').then(r => r.text());
  }} catch (e) {{}}
}}
setInterval(tick, 200);
setInterval(frames, 100);
setInterval(report, 1000);
tick();
frames();
report();
</script></body></html>""".encode()


def serve(frames, poses, modes, estimates, stats, sync, stop, names, play):
    known = set(names)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = page(names, play)
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/report":
                body = report_text(stats, names, sync).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/pose":
                now = time.time()
                out = {"play": play, "views": {}, "together": False, "between": None}
                for name in names:
                    view_t = stats[name].get("view_t") or 0
                    out["views"][name] = bool(view_t and now - view_t < 1.0)
                    hit = poses.get(name)
                    if not hit or now - hit["t"] > 0.6:
                        est = estimates.get(name)
                        kept = ("aim", "lane", "track", "land", "down", "odom", "memory")
                        out[name] = None if not est or est.get("mode") not in kept else {
                            "id": TAG_ID,
                            "range_m": est["range_m"],
                            "perp_m": est["perp_m"],
                            "x": est["x"],
                            "y": est["y"],
                            "z": est["z"],
                            "mode": est.get("mode", ""),
                        }
                    else:
                        out[name] = {
                            "id": hit["id"],
                            "range_m": hit["range_m"],
                            "perp_m": hit["perp_m"],
                            "x": hit["x"],
                            "y": hit["y"],
                            "z": hit["z"],
                            "world": bool(hit.get("world")),
                            "mode": "play" if play else modes.get(name, ""),
                        }
                pair = [out.get(name) for name in names]
                if len(pair) > 1 and all(pair):
                    out["together"] = True
                    a, b = pair[0], pair[1]
                    dx = b["x"] - a["x"]
                    dy = b["y"] - a["y"]
                    dz = b["z"] - a["z"]
                    out["between"] = {
                        "dx": round(dx, 2),
                        "dy": round(dy, 2),
                        "dz": round(dz, 2),
                        "apart": round(float(np.hypot(np.hypot(dx, dy), dz)), 2),
                    }
                    key = tuple(round(poses[name]["t"], 2) for name in names)
                    if key != sync["last"]:
                        sync["last"] = key
                        sync["count"] += 1
                        print(f"both poses {sync['count']}", flush=True)
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            path = self.path.split("?", 1)[0].lstrip("/")
            if path.endswith(".jpg"):
                name = path[:-4]
                if name not in known:
                    self.send_error(404)
                    return
                body = jpeg(frames, name)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    return
                return
            name = path
            if name not in known:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while not stop.is_set():
                    payload = jpeg(frames, name)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 8770), Handler)
    httpd.serve_forever()


def main():
    play = "--no_fly" in sys.argv[1:]
    want = [arg for arg in sys.argv[1:] if arg != "--no_fly"]
    fleet = load_fleet(want) if want else present_fleet()
    if not fleet:
        raise SystemExit(
            "no mapped dongle is plugged in. Plug a numbered stick in and set its drone "
            "with tello_dongle_setup.py, or pass drone names as arguments."
        )
    drones = tuple((d["drone"], d["iface"], d["video_port"]) for d in fleet)
    stop = threading.Event()
    frames = {}
    threads = []
    sockets = []
    names = tuple(d[0] for d in drones)
    poses = {}
    heights = {}
    batteries = {}
    motion = {}
    modes = {}
    estimates = {}
    takeoff_lock = threading.Lock()
    flight = {"stamp": time.time(), "go_names": set(), "slot": {}}
    ready = {}
    sync = {"count": 0, "last": None}
    session = next_session()
    stats = {
        name: {
            "frames": 0, "drops": 0, "overflow": 0, "tag_hit": 0, "tag_miss": 0,
            "file": "", "session": session,
        }
        for name, _iface, _port in drones
    }

    for name, iface, local_port in drones:
        cmd = dev_socket(iface, 8889)
        video = dev_socket(iface, 11111)
        state = dev_socket(iface, 8890)
        sockets.append((cmd, video, state))
        threads.append(
            threading.Thread(
                target=run_drone,
                args=(
                    name, iface, cmd, video, state, local_port, frames, poses, heights,
                    batteries, motion, modes, estimates, stats, stop, takeoff_lock, flight, names, ready, play,
                ),
                daemon=True,
            )
        )
        threads.append(
            threading.Thread(target=watch_pose, args=(name, frames, poses, stats, stop), daemon=True)
        )
        threads.append(
            threading.Thread(target=log_track, args=(session, name, poses, ready, modes, heights, stop), daemon=True)
        )
    threads.append(threading.Thread(target=write_report, args=(stats, names, sync, stop), daemon=True))
    threads.append(threading.Thread(target=serve, args=(frames, poses, modes, estimates, stats, sync, stop, names, play), daemon=True))

    for thread in threads:
        thread.start()
    print("play, no launch" if play else "flight", flush=True)
    print("video http://127.0.0.1:8770/", flush=True)

    try:
        while not stop.wait(30):
            pass
    finally:
        stop.set()
        for cmd, _video, state in sockets:
            try:
                sdk(cmd, "streamoff", timeout=1)
            except OSError:
                pass
            cmd.close()
            state.close()


if __name__ == "__main__":
    main()

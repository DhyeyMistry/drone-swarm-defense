#!/usr/bin/env python3
"""Tello video stream + tag36h11 pose relative to the on-screen AprilTag.

Waits until the Tello AP (192.168.10.1) answers, then shows the camera
with the tag pose overlaid. Position is the camera in the tag frame:
+X right on the tag, +Y down on the tag, +Z out of the screen toward the drone.
"""

import argparse
import json
import math
import socket
import subprocess
import sys
import time

import cv2
import numpy as np
from djitellopy import Tello
from pupil_apriltags import Detector

TELLO_IP = "192.168.10.1"
TELLO_CMD_PORT = 8889
# Calibrated on the 960x720 stream: black square 119 mm, tape 1.00 m from the lens.
# The run passed --tag-size 0.088, so -z sat near 0.580 m and fx_solved near 1178.
# f = 683 * (1.00 / 0.580) * (0.088 / 0.119) = 871 px. Square pixels.
CAL_F_AT_960 = 871.0


def camera_params(width, height):
    focal = CAL_F_AT_960 * (width / 960.0)
    return focal, focal, width / 2.0, height / 2.0


def route_source():
    """Local address the kernel would use to reach the drone. Sends no packets."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((TELLO_IP, TELLO_CMD_PORT))
        return sock.getsockname()[0]
    except OSError as exc:
        return f"err:{exc}"
    finally:
        sock.close()


def wifi_snapshot():
    ssid = "?"
    visible = []
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "IN-USE,SSID,SIGNAL", "device", "wifi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nmcli:{exc}", []
    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) < 2:
            continue
        in_use, name, signal = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
        if in_use == "*":
            ssid = name or "(hidden)"
        if "TELLO" in name.upper():
            visible.append(f"{name}@{signal}")
    return ssid, visible


def ipv4_addrs():
    try:
        out = subprocess.run(
            ["ip", "-4", "-br", "addr", "show", "scope", "global"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"ip:{exc}"]
    lines = []
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            lines.append(f"{fields[0]}={fields[2]}")
    return lines or ["none"]


def wait_for_tello():
    print("waiting until route to 192.168.10.1 uses 192.168.10.x (no SDK packets yet)", flush=True)
    while True:
        src = route_source()
        ssid, visible = wifi_snapshot()
        addrs = " ".join(ipv4_addrs())
        seen = ",".join(visible) if visible else "none"
        on_lan = isinstance(src, str) and src.startswith("192.168.10.")
        print(
            f"wifi={ssid} src={src} addrs={addrs} tello_ssids={seen}",
            flush=True,
        )
        if on_lan:
            print("on 192.168.10.x", flush=True)
            return
        time.sleep(2)


def draw_tag(frame, tag, cam_in_tag):
    corners = tag.corners.astype(int)
    for i in range(4):
        cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
    center = tuple(tag.center.astype(int))
    cv2.circle(frame, center, 4, (0, 0, 255), -1)
    x, y, z = cam_in_tag
    text = f"id {tag.tag_id}  x {x:+.3f}  y {y:+.3f}  z {z:+.3f} m"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-size", type=float, default=0.15, help="black-square side, meters")
    parser.add_argument(
        "--known-distance",
        type=float,
        default=0.0,
        help="tape distance from lens to tag plane, meters. Prints solved fx.",
    )
    args = parser.parse_args()
    tag_size = args.tag_size

    wait_for_tello()
    tello = Tello(retry_count=1)
    while True:
        try:
            tello.connect()
            break
        except Exception as exc:
            print(f"connect failed: {exc}", flush=True)
            time.sleep(2)
    battery = tello.get_battery()
    print(f"battery {battery}%", flush=True)
    tello.streamon()
    time.sleep(2)
    reader = tello.get_frame_read()

    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
    window = "tello"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_print = 0.0
    params = None

    print(
        "keys: q quit | [ ] tag size down/up | tag size is the black square in meters",
        flush=True,
    )
    print(f"tag_size {tag_size:.3f} m", flush=True)

    try:
        while True:
            frame = reader.frame
            # djitellopy placeholder until the first H.264 frame; frames are RGB
            if frame.shape[1] < 600:
                time.sleep(0.05)
                continue
            view = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            h, w = view.shape[:2]
            if params is None or params[-1] != (w, h):
                fx, fy, cx, cy = camera_params(w, h)
                params = (fx, fy, cx, cy, (w, h))
                print(f"frame {w}x{h}  fx {fx:.1f} fy {fy:.1f}", flush=True)

            tags = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=params[:4],
                tag_size=tag_size,
            )
            now = time.monotonic()
            if tags:
                tag = max(tags, key=lambda item: item.decision_margin)
                cam_in_tag = (-tag.pose_R.T @ tag.pose_t).ravel()
                draw_tag(view, tag, cam_in_tag)
                if now - last_print > 0.1:
                    payload = {
                        "id": int(tag.tag_id),
                        "x": round(float(cam_in_tag[0]), 4),
                        "y": round(float(cam_in_tag[1]), 4),
                        "z": round(float(cam_in_tag[2]), 4),
                        "tag_size_m": round(tag_size, 4),
                    }
                    # D_est uses f_guess. f_true = f_guess * D_tape / D_est, if tag_size is the ruler size.
                    if args.known_distance > 0 and cam_in_tag[2] < 0:
                        d_est = -float(cam_in_tag[2])
                        payload["fx_solved"] = round(params[0] * args.known_distance / d_est, 1)
                    print(json.dumps(payload), flush=True)
                    last_print = now
            else:
                cv2.putText(
                    view,
                    "no tag",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            hud = f"tag black square {tag_size * 100:.1f} cm   bat {battery}%"
            cv2.putText(view, hud, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, hud, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("]"):
                tag_size = min(tag_size + 0.005, 1.0)
                print(f"tag_size {tag_size:.3f} m", flush=True)
            elif key == ord("["):
                tag_size = max(tag_size - 0.005, 0.01)
                print(f"tag_size {tag_size:.3f} m", flush=True)
    finally:
        if reader is not None:
            reader.stop()
        try:
            tello.streamoff()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

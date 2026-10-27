#!/home/jaimeromero/action-space/.venv/bin/python
"""Map AprilTags 10, 5, 6, 7 in the tag-10 origin frame, on a quick-connect drone.

Picks the first drone in config/fleet.yaml `quick_connect` (override with --drone),
brings up its WiFi on the stick from config/dongles.json, connects through the
cal_and_map identity gate (no takeoff/rc is ever sent), and builds the map: hold the
drone so it sees tag 10 together with 5/6/7 (two tags in a frame links them). Green =
placed in the tag-10 frame, yellow = seen but not yet linked. 'q' writes config/map.yaml.

  ./tello_map.py                 # quick-connect drone, tags 10,5,6,7
  ./tello_map.py --drone TELLO-3 --tags-list 10,5,6,7,8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tello_calibration.calib.io import camera_from_dict, load_yaml
from tello_calibration.session import IdentityError, SessionError, resolve_drone, \
    verified_link, video_stream
from tello_calibration.tags.detect import TagDetector, load_tag_config
from tello_calibration.tags.mapping import MapBuilder, TagMap
from tello_calibration.utils.config import CONFIG_DIR

GREEN, YELLOW, RED = (0, 220, 0), (0, 220, 220), (0, 0, 255)


def quick_connect_drone(override: str | None) -> str:
    if override:
        return override
    fleet = load_yaml(CONFIG_DIR / "fleet.yaml")
    qc = (fleet or {}).get("quick_connect") or []
    if not qc:
        raise SystemExit("no quick_connect drones in config/fleet.yaml")
    return str(qc[0])


def dongle_iface(drone: str) -> str | None:
    reg = json.loads((CONFIG_DIR / "dongles.json").read_text())
    for d in reg.get("dongles", []):
        if d.get("drone") == drone:
            return d.get("iface")
    return None


def bring_up(ssid: str, iface: str | None) -> None:
    args = ["nmcli", "con", "up", ssid] + (["ifname", iface] if iface else [])
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    print(f"nmcli up {ssid}: {'ok' if result.returncode == 0 else (result.stderr or '').strip()}",
          flush=True)


def map_loop(video, detector: TagDetector, origin_tag: int, designated: set[int]) -> TagMap:
    builder = MapBuilder(origin_tag)
    tag_map = builder.build()
    last_id, last_build, last_print = 0, 0.0, 0.0
    cv2.namedWindow("map", cv2.WINDOW_NORMAL)
    print("mapping... press q / s / Esc in the window, OR Ctrl+C here, to save and stop.", flush=True)
    try:
        while True:
            frame, frame_id, _ = video.get_latest()
            if frame is not None and frame_id != last_id:
                last_id = frame_id
                observations = [o for o in detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                                if o.tag_id in designated]
                builder.add_frame(observations)
                now = time.monotonic()
                if now - last_build > 0.5:
                    tag_map, last_build = builder.build(), now
                if now - last_print > 1.0:  # terminal progress (works even if the window has no focus)
                    print(f"  mapped {tag_map.mapped()} / want {sorted(designated)}", flush=True)
                    last_print = now
                image = frame.copy()
                for o in observations:
                    placed = o.tag_id in tag_map.poses
                    color = GREEN if (o.accepted and placed) else YELLOW if o.accepted else RED
                    pts = o.corners.round().astype(np.int32)
                    cv2.polylines(image, [pts], True, color, 2)
                    where = "mapped" if placed else "seen" if o.accepted else o.reasons[0]
                    x, y = pts.min(axis=0)
                    for col, th in (((0, 0, 0), 4), (color, 2)):
                        cv2.putText(image, f"id {o.tag_id} {where}", (int(x), max(int(y) - 8, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th)
                cv2.putText(image, f"mapped {tag_map.mapped()} / want {sorted(designated)}  q/Ctrl+C save",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("map", image)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("s"), 27):
                break
            if cv2.getWindowProperty("map", cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\nCtrl+C -> stopping and saving the map", flush=True)
    return builder.build()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", help="ssid like TELLO-3; default = first fleet.yaml quick_connect")
    parser.add_argument("--tags-list", default="10,5,6,7,8,9", help="tag ids to map (origin first)")
    parser.add_argument("--tags", type=Path, default=CONFIG_DIR / "tags.yaml")
    parser.add_argument("--camera", type=Path, help="default: config/camera/<drone>.yaml")
    parser.add_argument("--out", type=Path, default=CONFIG_DIR / "map.yaml")
    args = parser.parse_args()

    ssid = quick_connect_drone(args.drone)
    key = ssid.lower().replace("-", "_")                 # TELLO-3 -> tello_3 (drones/camera keys)
    designated = {int(x) for x in args.tags_list.split(",") if x.strip()}
    print(f"mapping {sorted(designated)} on {ssid} ({key}); origin = tag 10.", flush=True)

    bring_up(ssid, dongle_iface(ssid))
    camera_path = args.camera or CONFIG_DIR / "camera" / f"{key}.yaml"
    if not camera_path.exists():
        print(f"no camera calibration at {camera_path}; run 03_calibrate.py first", flush=True)
        return 1
    K, dist = camera_from_dict(load_yaml(camera_path))
    tags = load_tag_config(args.tags)
    if tags.origin_tag not in designated:
        designated.add(tags.origin_tag)
    detector = TagDetector(tags, K, dist)

    try:
        drone, local_ip = resolve_drone(key)
        with verified_link(drone, local_ip) as link, video_stream(link, local_ip) as video:
            try:
                tag_map = map_loop(video, detector, tags.origin_tag, designated)
            finally:
                cv2.destroyAllWindows()
    except IdentityError:
        return 2
    except SessionError:
        return 1
    except KeyboardInterrupt:
        print("Ctrl+C", flush=True)
        return 0

    if tags.origin_tag not in tag_map.poses:
        print(f"origin tag {tags.origin_tag} never linked; no map written.", flush=True)
        return 1
    tag_map.save(args.out)
    for tag_id in tag_map.mapped():
        t = tag_map.poses[tag_id][:3, 3]
        print(f"tag {tag_id:2d}  x {t[0]:+.3f}  y {t[1]:+.3f}  z {t[2]:+.3f} m", flush=True)
    print(f"wrote {args.out} ({len(tag_map.mapped())} tags in tag-{tags.origin_tag} frame)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

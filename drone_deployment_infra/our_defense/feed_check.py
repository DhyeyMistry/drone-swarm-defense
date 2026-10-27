#!/usr/bin/env python3
"""Check that we read the attackers' reported positions correctly. No flying.

Listens to the attacker feed and prints, for every drone in it, the raw `pose` and our
map-frame position under each tag-frame convention (feeds.CONVENTIONS). Put an attacker
drone (or hold it) at a spot you measured with a tape -- e.g. 1 m to the RIGHT of tag 10
(facing the tag), 2.5 m out from the wall, below the tag -- and see which column matches:

    TELLO-8  raw tag 10 xyz [-1.01, -0.38, -2.49]  age 0.05
        gui    -> x +1.01  y -0.38  z +2.49   (x right of tag 10, y above its centre, z out from the wall)
        readme -> x -1.01  y +0.38  z +2.49

Then run defend.py with --feed_frame <that one>. Use a spot that is off to one side AND
not at the tag's height: straight in front at tag height both conventions agree.
--raw also dumps whole datagrams (to send to us if the format is different).

  python feed_check.py                       # all drones on :9100
  python feed_check.py --port 9100 --only TELLO-8,TELLO-9 --raw
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feeds import CONVENTIONS, entry_to_map  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--map", default=None, help="map.yaml (default config/map.yaml)")
    ap.add_argument("--only", default="", help="comma list of drone names to show")
    ap.add_argument("--raw", action="store_true", help="also print each whole datagram")
    ap.add_argument("--every_s", type=float, default=1.0, help="print at most this often")
    args = ap.parse_args()

    from tello_calibration.tags.mapping import TagMap
    from tello_calibration.utils.config import CONFIG_DIR
    map_path = Path(args.map) if args.map else CONFIG_DIR / "map.yaml"
    poses = {int(k): np.asarray(v) for k, v in TagMap.load(map_path).poses.items()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    print(f"listening on :{args.port}, map {map_path} (tags {sorted(poses)}). Ctrl-C to stop.", flush=True)
    last = 0.0
    while True:
        data, src = sock.recvfrom(65535)
        if time.time() - last < args.every_s:
            continue
        last = time.time()
        try:
            msg = json.loads(data)
        except ValueError:
            print(f"from {src}: not JSON ({len(data)} B): {data[:120]!r}", flush=True)
            continue
        if args.raw:
            print(json.dumps(msg), flush=True)
        if msg.get("v") != 1:
            print(f"from {src}: not repo telemetry v1 (keys {sorted(msg)[:10]}) -> feeds.entry_to_map "
                  f"needs an adapter; run with --raw and keep a sample", flush=True)
            continue
        print(f"--- seq {msg.get('seq')} src {msg.get('src')} from {src[0]}", flush=True)
        for e in msg.get("drones", []):
            if only and e.get("name") not in only:
                continue
            pose = e.get("pose") or {}
            print(f"  {e.get('name')}  team {e.get('team')!r}  fly {e.get('fly')}  h {e.get('h')}  "
                  f"raw tag {pose.get('tag')} xyz {pose.get('xyz')}  age {pose.get('age')}", flush=True)
            if pose.get("tag") is not None and int(pose["tag"]) not in poses:
                print(f"      tag {pose['tag']} is NOT in our map -> can't place this drone; remap with it",
                      flush=True)
                continue
            if e.get("h") is not None:
                print(f"      height from h: {float(e['h']):.2f} m above the floor "
                      f"(defend.py uses this by default; y below is from the tag)", flush=True)
            for conv in CONVENTIONS:
                p = entry_to_map(e, poses, conv)
                if p is not None:
                    print(f"      {conv:6s} -> x {p[0]:+.2f}  y {p[1]:+.2f}  z {p[2]:+.2f}   "
                          f"(x right of tag 10, y above its centre, z out from the wall)", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass

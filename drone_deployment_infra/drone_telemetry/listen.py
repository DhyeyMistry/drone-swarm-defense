#!/usr/bin/env python3
"""Print the fleet telemetry drone_gui publishes -- and the template for a policy that reads it.

  ../../.venv/bin/python listen.py                    # everything on :9100
  ../../.venv/bin/python listen.py --team attack      # only attackers (what a defender reads)
  ../../.venv/bin/python listen.py --port 9200 --raw  # raw JSON

Only one process can listen on a port. For several consumers, add one destination
per consumer (different ports) in the GUI's telemetry box.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

STALE_S = 0.5     # a pose older than this is last-known, not current


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--team", choices=("attack", "defense"), help="only show this team")
    ap.add_argument("--raw", action="store_true", help="print each datagram as JSON")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"listening on {args.host}:{args.port}", flush=True)
    last_seq, lost, t_prev = None, 0, None
    while True:
        data, _ = sock.recvfrom(65535)
        msg = json.loads(data)
        if msg.get("v") != 1:
            print(f"unknown telemetry version {msg.get('v')}; skipping")
            continue
        if last_seq is not None and msg["seq"] > last_seq + 1:
            lost += msg["seq"] - last_seq - 1
        last_seq = msg["seq"]
        now = time.time()
        rate = f"{1 / (now - t_prev):5.1f} Hz" if t_prev else "  --   "
        t_prev = now
        if args.raw:
            print(json.dumps(msg), flush=True)
            continue
        drones = [d for d in msg["drones"] if not args.team or d["team"] == args.team]
        print(f"seq {msg['seq']:6d}  {rate}  lost {lost}  [{msg['src']}]", flush=True)
        for d in drones:
            p = d.get("pose")
            if p is None:
                where = "pose: never seen a tag"
            else:
                x, y, z = p["xyz"]
                stale = "STALE " if p["age"] > STALE_S else ""
                where = (f"{stale}tag {p['tag']}  x {x:+.2f}  y {y:+.2f}  z {z:+.2f} m"
                         f"  yaw {p['yaw']:+.0f}  age {p['age']:.2f}s")
            fly = "FLY" if d["fly"] else "gnd"
            print(f"   {d['slot']} {d['name']:<13} {d['team']:<7} {fly} {d['mode']:<7} "
                  f"bat {d['bat'] if d['bat'] is not None else '--':>3}  {where}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

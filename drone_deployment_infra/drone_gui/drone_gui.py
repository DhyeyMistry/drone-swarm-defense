#!/usr/bin/env python3
"""Browser GUI for flying several Tellos at once, one per USB WiFi dongle.

  ../../.venv/bin/python -u drone_gui.py            # then open http://127.0.0.1:8780/
  ../../.venv/bin/python -u drone_gui.py --port 8781
  ../../.venv/bin/python -u drone_gui.py --sim 8    # 8 simulated drones, no hardware

Shows every plugged-in dongle and its drone (link, battery, height), each drone's
live camera with AprilTags outlined, and per-drone Connect / Take off / Land /
manual sticks / Auto. Auto = fly to the assigned tag and land STANDOFF metres in
front of it. "Auto all" does that for every connected drone at the same time.

Telemetry: every drone's team (attack/defense), pose, battery... is published as UDP
JSON at a rate set live in the page (see ../drone_telemetry/README.md).

Don't run this together with core/tello_link.py or tello_dual_video.py: they bind
the same per-dongle ports and would steal each other's packets.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tello_unit import DD_ROOT, Drone, load_tag_sizes

sys.path.insert(0, str(DD_ROOT / "core"))
sys.path.insert(0, str(DD_ROOT / "drone_telemetry"))
from tello_dongle_setup import REGISTRY_PATH, load_registry, present_ifaces  # noqa: E402
from telemetry import TEAMS, Publisher, load_roles  # noqa: E402

HERE = Path(__file__).resolve().parent
# Default tag each drone flies to in Auto; change it live in the GUI.
DEFAULT_TAGS = {"TELLO-2": 8, "TELLO-FE2950": 9}
TAKEOFF_STAGGER_S = 2.0   # "Auto all" launches drones this far apart


class Fleet:
    def __init__(self, sim: int = 0) -> None:
        self.drones: dict[str, Drone] = {}     # by iface
        self.tag_sizes = load_tag_sizes()
        self.lock = threading.Lock()
        self.sim = sim
        self.roles = load_roles()
        self.team_override: dict[str, str] = dict(self.roles["drones"])   # name -> team, set from the GUI
        self.publisher = Publisher(self.telemetry, "sim" if sim else "real", self.roles["destinations"],
                                   self.roles["hz"], self.roles["enabled"])
        if sim:
            # Simulated drones only: sim i looks at (and defaults to) tag i (tags 1-9 repeat).
            from sim_drone import SimDrone
            for i in range(1, sim + 1):
                tag = (i - 1) % 9 + 1
                drone = SimDrone(i, tag, self.tag_sizes)
                drone.target_tag = tag
                self.drones[drone.iface] = drone
            return
        threading.Thread(target=self._discover_loop, daemon=True).start()

    def _discover_loop(self) -> None:
        while True:
            try:
                self.discover()
            except Exception as exc:  # noqa: BLE001
                print(f"discover error: {exc}", flush=True)
            time.sleep(3.0)

    def discover(self) -> None:
        here = present_ifaces()
        reg = {d["iface"]: d for d in load_registry(REGISTRY_PATH)["dongles"]}
        with self.lock:
            for iface in sorted(here - set(self.drones)):
                entry = reg.get(iface, {})
                number = entry.get("number")
                port = entry.get("video_port") or (17100 + len(self.drones))
                drone = Drone(iface, number, entry.get("drone", ""), port, self.tag_sizes)
                drone.target_tag = DEFAULT_TAGS.get(drone.drone)
                self.drones[iface] = drone
                print(f"dongle {iface} (#{number}) -> {drone.drone or 'unassigned'}", flush=True)
            for iface in set(self.drones) - here:
                print(f"dongle {iface} unplugged", flush=True)
                self.drones.pop(iface).close()
            drones = list(self.drones.values())
        for drone in drones:
            drone.refresh_wifi()

    def get(self, iface: str) -> Drone | None:
        return self.drones.get(iface)

    def all(self) -> list[Drone]:
        return sorted(self.drones.values(), key=lambda d: (d.number is None, d.number or 0, d.iface))

    def team(self, drone: Drone, slot: int) -> str:
        """GUI override, then roles.yaml by name, then roles.yaml by grid slot; '' = no team."""
        if drone.name in self.team_override:
            return self.team_override[drone.name]
        return self.roles["slots"].get(slot, "")

    def telemetry(self) -> list[dict]:
        return [d.telemetry(i, self.team(d, i)) for i, d in enumerate(self.all(), 1)]

    def states(self) -> list[dict]:
        out = []
        for i, d in enumerate(self.all(), 1):
            out.append(dict(d.status(), slot=i, team=self.team(d, i)))
        return out

    def auto_all(self) -> None:
        ready = [d for d in self.all() if d.link and d.target_tag is not None]
        def run():
            for i, drone in enumerate(ready):
                if i and not drone.flying:
                    time.sleep(TAKEOFF_STAGGER_S)
                drone.start_auto()
        threading.Thread(target=run, daemon=True).start()

    def land_all(self) -> None:
        for drone in self.all():
            drone.land("land all")

    def close(self) -> None:
        self.publisher.close()
        threads = [threading.Thread(target=d.close) for d in self.all()]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)


def make_handler(fleet: Fleet):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (HERE / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json({"drones": fleet.states(), "telemetry": fleet.publisher.status(), "t": time.time()})
            elif self.path.startswith("/video/"):
                self._mjpeg(self.path.split("/")[2])
            elif self.path.startswith("/snap/"):
                # One JPEG per request. The page uses this for its grid: browsers allow
                # only ~6 connections per host, so 8 endless MJPEG streams would starve.
                drone = fleet.get(self.path.split("/")[2].split("?")[0])
                jpg = drone.jpeg() if drone else None
                if jpg is None:
                    self._send(404, b"{}")
                else:
                    self._send(200, jpg, "image/jpeg")
            else:
                self._send(404, b"{}")

        def _mjpeg(self, iface: str) -> None:
            drone = fleet.get(iface)
            if drone is None:
                self._send(404, b"{}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while fleet.get(iface) is drone:
                    jpg = drone.jpeg()
                    if jpg is None or jpg is last:
                        time.sleep(0.03)
                        continue
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            parts = self.path.strip("/").split("/")   # api/<action> or api/<iface>/<action>
            if parts[:1] != ["api"]:
                return self._send(404, b"{}")
            if len(parts) == 2:
                action = parts[1]
                if action == "telemetry":
                    try:
                        fleet.publisher.configure(body.get("enabled"), body.get("hz"), body.get("destinations"))
                    except (ValueError, TypeError) as exc:
                        return self._json({"error": str(exc)}, 400)
                elif action == "connect_all":
                    for d in fleet.all():
                        if not (d.link and d.streaming) and (d.drone or d.ssid):
                            d.connect()
                elif action == "land_all":
                    fleet.land_all()
                elif action == "auto_all":
                    fleet.auto_all()
                elif action == "emergency_all":
                    for d in fleet.all():
                        d.emergency()
                elif action == "shutdown" and fleet.sim:
                    # Sim only: a page button that kills a real fleet's server is too easy to hit.
                    print("stop requested from the page", flush=True)
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    return self._json({"error": f"unknown {action}"}, 400)
                return self._json({"ok": True})
            drone = fleet.get(parts[1]) if len(parts) == 3 else None
            if drone is None:
                return self._json({"error": "no such dongle"}, 404)
            action = parts[2]
            if action == "connect":
                drone.connect(body.get("ssid"))
            elif action == "disconnect":
                threading.Thread(target=drone.disconnect, daemon=True).start()
            elif action == "takeoff":
                drone.takeoff()
            elif action == "land":
                drone.land("button")
            elif action == "hover":
                drone.hover()
            elif action == "auto":
                drone.start_auto()
            elif action == "rc":
                drone.set_manual(body.get("sticks", (0, 0, 0, 0)))
            elif action == "config":
                if "target_tag" in body:
                    tag = body["target_tag"]
                    drone.target_tag = int(tag) if tag not in (None, "") else None
                if "standoff" in body:
                    drone.standoff = max(0.5, min(3.0, float(body["standoff"])))
                if "team" in body:
                    if body["team"] in TEAMS or body["team"] == "none":
                        fleet.team_override[drone.name] = "" if body["team"] == "none" else body["team"]
                    else:   # "auto": back to roles.yaml
                        fleet.team_override.pop(drone.name, None)
            elif action == "emergency":
                drone.emergency()
            elif action == "reset" and getattr(drone, "is_sim", False):
                drone.reset()
            else:
                return self._json({"error": f"unknown {action}"}, 400)
            return self._json({"ok": True})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--sim", type=int, default=0, metavar="N",
                        help="run N simulated drones instead of real dongles (GUI testing)")
    args = parser.parse_args()

    fleet = Fleet(sim=args.sim)
    # Localhost only: anyone who can reach this page can fly the drones.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(fleet))
    server.daemon_threads = True

    def shutdown(*_):
        print("shutting down: landing anything airborne", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"drone GUI on http://127.0.0.1:{args.port}/  (Ctrl-C lands everything and quits)", flush=True)
    server.serve_forever()
    fleet.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

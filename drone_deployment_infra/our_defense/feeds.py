"""Where are the attackers? Read the reported positions and turn them into map-frame Tracks.

The reported feed is assumed to be the repo's fleet telemetry (drone_telemetry/README.md,
`v: 1`): UDP JSON, one datagram per tick, `drones: [{name, team, fly, link, h, pose}]`.
`pose.xyz` there is the camera in ONE tag's frame -- not the room frame -- so
entry_to_map() converts it with that tag's pose from config/map.yaml (map tag frame:
x right, y UP, z OUT of the face).

WHICH tag frame (CONVENTIONS): drone_gui runs pupil-apriltags + tello_policy.cam_in_tag.
With the OFFICIAL tag36h11 prints (the real tags; drone_gui's own sim draws those) that
gives what drone_telemetry/README.md says -- x right, y down, z into the wall -- checked
against drone_gui --sim's ground truth (0.08 m). Tags drawn by cv2.aruco are rotated
180 deg from the official ones and flip x and y ("gui" below). Default "readme".
Confirm on the real feed with feed_check.py.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np

from defense_core import Track

VEL_ALPHA = 0.4          # EMA weight of a new velocity sample
VEL_RESET_S = 1.0        # a gap longer than this restarts the velocity estimate
GROUND_H_M = 0.2         # reported height below this = on the ground

# telemetry tag-frame xyz -> map tag frame (x right, y up, z out), per convention
CONVENTIONS = {
    "readme": (1.0, -1.0, -1.0),   # drone_gui with official tag prints (default)
    "gui": (-1.0, 1.0, -1.0),      # drone_gui with cv2.aruco-drawn tag prints
}


def entry_to_map(entry: dict, tag_poses: dict, convention: str = "readme") -> np.ndarray | None:
    """One telemetry drone entry -> its position in the map (tag-10) frame, or None."""
    if entry.get("map_xyz") is not None:
        return np.asarray(entry["map_xyz"], float)
    pose = entry.get("pose")
    if not pose or pose.get("xyz") is None:
        return None
    xyz = np.asarray(pose["xyz"], float)
    if pose.get("frame") == "map":
        return xyz
    tag = pose.get("tag")
    M = tag_poses.get(int(tag)) if tag is not None else None
    if M is None:
        return None                                   # tag not in our map: can't place it
    M = np.asarray(M, float)
    p_tag = xyz * np.asarray(CONVENTIONS[convention])  # telemetry tag frame -> map tag frame
    return M[:3, :3] @ p_tag + M[:3, 3]


class TelemetryFeed:
    """Background UDP listener; tracks() returns the latest Track per attacker name."""

    def __init__(self, port: int, attackers: list[str] | None, tag_poses: dict, host: str = "0.0.0.0",
                 convention: str = "readme", height_from_h: bool = True, tag10_height: float = 1.22,
                 exclude: list[str] | None = None) -> None:
        # attackers=None: AUTO -- every drone in the feed that is not in `exclude` (our
        # defenders) and not labelled team "defense" is treated as an attacker.
        self.auto = attackers is None
        self.attackers = [] if attackers is None else [display_name(drone_number(a) or 0) if drone_number(a)
                                                        else a for a in attackers]
        self.exclude = set(exclude or [])
        self.convention = convention
        # drone_gui's pose y (and yaw) can flip sign on small/far tags (telemetry README,
        # "Known limitation"); x, z and range stay reliable. The Tello's own height `h`
        # (above its takeoff point = the floor) doesn't flip, so use it for height.
        self.height_from_h = height_from_h
        self.tag10_height = tag10_height
        self.tag_poses = tag_poses
        self.lock = threading.Lock()
        self.state: dict[str, dict] = {}           # name -> {pos, vel, t_meas, alive, raw}
        self.datagrams = 0
        self.last_rx = 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if msg.get("v") != 1:
                continue
            now = time.time()
            self.datagrams += 1
            self.last_rx = now
            for e in msg.get("drones", []):
                if self._is_attacker(e):
                    self._update(e, now)

    def _is_attacker(self, e: dict) -> bool:
        name = e.get("name")
        if not self.auto:
            return name in self.attackers
        if not name or name in self.exclude or e.get("team") == "defense":
            return False
        if name not in self.attackers:
            with self.lock:
                self.attackers.append(name)
            print(f"feed: tracking attacker {name} (team {e.get('team')!r})", flush=True)
        return True

    def _update(self, e: dict, now: float) -> None:
        pos = entry_to_map(e, self.tag_poses, self.convention)
        age = float((e.get("pose") or {}).get("age", 0.0) or 0.0)
        h = e.get("h")
        alive = bool(e.get("fly", True)) and bool(e.get("link", True)) and \
            (h is None or h >= GROUND_H_M)
        with self.lock:
            s = self.state.setdefault(e["name"], {"pos": None, "vel": np.zeros(3), "t_meas": 0.0})
            s["alive"] = alive
            s["raw"] = e
            if pos is None:
                return
            if self.height_from_h and h is not None and h >= GROUND_H_M:
                pos = pos.copy()
                pos[1] = float(h) - self.tag10_height       # map y = height above tag 10's centre
            t_meas = now - age
            if s["pos"] is not None and t_meas > s["t_meas"] + 1e-3:
                dt = t_meas - s["t_meas"]
                if dt < VEL_RESET_S:
                    v = (pos - s["pos"]) / dt
                    s["vel"] = (1 - VEL_ALPHA) * s["vel"] + VEL_ALPHA * v
                else:
                    s["vel"] = np.zeros(3)
            if t_meas >= s["t_meas"]:
                s["pos"], s["t_meas"] = pos, t_meas

    def tracks(self) -> list[Track]:
        now = time.time()
        out = []
        with self.lock:
            for name in self.attackers:
                s = self.state.get(name)
                if s is None or s["pos"] is None:
                    continue
                out.append(Track(name, s["pos"].copy(), s["vel"].copy(), now - s["t_meas"], s["alive"]))
        return out

    def close(self) -> None:
        self.stop.set()
        self.sock.close()


# ---- names -------------------------------------------------------------------------------
def _dongle_numbers() -> dict[str, int]:
    """config/dongles.json: drone name ("TELLO-8") -> stick number (8)."""
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "config" / "dongles.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    rows = data.get("dongles", []) if isinstance(data, dict) else data
    return {r["drone"]: int(r["number"]) for r in rows if r.get("drone") and r.get("number") is not None}


def drone_number(name, table: dict[str, int] | None = None) -> int | None:
    """"TELLO-8", "drone8", "8" or 8 -> 8 (via dongles.json for named drones)."""
    if isinstance(name, int):
        return name
    s = str(name).strip()
    table = _dongle_numbers() if table is None else table
    if s in table:
        return table[s]
    for prefix in ("drone", "TELLO-", "tello_", "tello-"):
        if s.lower().startswith(prefix.lower()) and s[len(prefix):].isdigit():
            return int(s[len(prefix):])
    return int(s) if s.isdigit() else None


def display_name(num: int, table: dict[str, int] | None = None) -> str:
    table = _dongle_numbers() if table is None else table
    for name, n in table.items():
        if n == num:
            return name
    return f"drone{num}"


# ---- swarm_policies.py telemetry bus ----------------------------------------------------
def find_swarm_hub() -> tuple[str, int] | None:
    """Address of a running swarm_policies.py's TelemetryHub (a 127.0.0.1 UDP socket on a
    random port, normally only passed to its own child processes via $SWARM_TELEM)."""
    import os
    spec = os.environ.get("SWARM_TELEM")
    if spec:
        host, _, port = spec.rpartition(":")
        return host, int(port)
    inodes = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
        except OSError:
            continue
        if b"swarm_policies.py" not in cmd:
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
                if link.startswith("socket:["):
                    inodes.add(link[8:-1])
        except OSError:
            continue
    if not inodes:
        return None
    try:
        lines = open("/proc/net/udp").read().splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        f = line.split()
        local, inode = f[1], f[9]
        ip_hex, port_hex = local.split(":")
        if inode in inodes and ip_hex == "0100007F":          # 127.0.0.1 (little-endian hex)
            return "127.0.0.1", int(port_hex, 16)
    return None


class SwarmBusFeed:
    """Attackers from swarm_policies.py's telemetry: "formation/drone<N>" carries `p`, the
    drone's position already in the MAP frame (x right, y up, z out of tag 10, metres) --
    published by swarm_formation.py. We join the hub as one more client (it relays every
    publish to every client that has said hello). The hub is re-searched if it goes quiet."""

    def __init__(self, attackers: list[str] | None, exclude: list[str] | None = None,
                 hub: tuple[str, int] | None = None) -> None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent.parent / "swarm_ops"))   # last: never shadow our modules
        from policies.telemetry import TelemetryClient
        self._Client = TelemetryClient
        self.table = _dongle_numbers()
        self.auto = attackers is None
        self.att_nums = set() if attackers is None else {drone_number(a, self.table) for a in attackers}
        self.exclude_nums = {drone_number(e, self.table) for e in (exclude or [])}
        self.fixed_hub = hub
        self.hub = None
        self.client = None
        self.lock = threading.Lock()
        self.state: dict[int, dict] = {}
        self.datagrams = 0
        self.last_rx = 0.0
        self._seen: dict[str, float] = {}
        self.stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def _connect(self) -> None:
        self._joined_t = time.time()
        hub = self.fixed_hub or find_swarm_hub()
        if hub and (hub != self.hub or self.client is None):
            if self.client is not None:
                self.client.close()
            self.hub = hub
            self.client = self._Client(hub)
            print(f"feed: joined swarm_policies telemetry hub {hub[0]}:{hub[1]}", flush=True)

    def _watch(self) -> None:
        while not self.stop.is_set():
            if time.time() - max(self.last_rx, getattr(self, "_joined_t", 0.0)) > 3.0:
                self._connect()                 # first time, or the hub went quiet / restarted
            self._poll()
            self.stop.wait(0.05)

    def _is_attacker(self, num: int) -> bool:
        if num is None:
            return False
        if not self.auto:
            return num in self.att_nums
        return num not in self.exclude_nums

    def _poll(self) -> None:
        if self.client is None:
            return
        now = time.time()
        for topic, (data, age) in self.client.since("formation/").items():
            stamp = now - age
            if self._seen.get(topic) == stamp:
                continue
            self._seen[topic] = stamp
            self.datagrams += 1
            self.last_rx = now
            num = drone_number(topic.split("/", 1)[1], self.table)
            if not self._is_attacker(num) or not isinstance(data, dict):
                continue
            h_cm = data.get("height_cm")
            alive = h_cm is None or h_cm >= GROUND_H_M * 100
            p = data.get("p")
            with self.lock:
                s = self.state.setdefault(num, {"pos": None, "vel": np.zeros(3), "t_meas": 0.0, "alive": True})
                if s["pos"] is None and p is not None and self.auto:
                    print(f"feed: tracking attacker {display_name(num, self.table)} (swarm formation)", flush=True)
                s["alive"] = alive
                if p is None:
                    continue
                pos = np.asarray(p, float)
                t_meas = stamp - float(data.get("fix_age") or 0.0)
                if s["pos"] is not None and t_meas > s["t_meas"] + 1e-3:
                    dt = t_meas - s["t_meas"]
                    s["vel"] = ((1 - VEL_ALPHA) * s["vel"] + VEL_ALPHA * (pos - s["pos"]) / dt
                                if dt < VEL_RESET_S else np.zeros(3))
                if t_meas >= s["t_meas"]:
                    s["pos"], s["t_meas"] = pos, t_meas

    def tracks(self) -> list[Track]:
        now = time.time()
        with self.lock:
            return [Track(display_name(n, self.table), s["pos"].copy(), s["vel"].copy(), now - s["t_meas"], s["alive"])
                    for n, s in self.state.items() if s["pos"] is not None]

    def close(self) -> None:
        self.stop.set()
        if self.client is not None:
            self.client.close()


class CombinedFeed:
    """Several feeds at once; per attacker (by stick number) the freshest report wins."""

    def __init__(self, feeds: list) -> None:
        self.feeds = feeds
        self.table = _dongle_numbers()

    @property
    def datagrams(self) -> int:
        return sum(f.datagrams for f in self.feeds)

    @property
    def last_rx(self) -> float:
        return max((f.last_rx for f in self.feeds), default=0.0)

    def tracks(self) -> list[Track]:
        best: dict = {}
        for f in self.feeds:
            for t in f.tracks():
                key = drone_number(t.name, self.table) or t.name
                if key not in best or t.age < best[key].age:
                    best[key] = t
        return list(best.values())

    def close(self) -> None:
        for f in self.feeds:
            f.close()

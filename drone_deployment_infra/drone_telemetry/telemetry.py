"""Publish fleet telemetry (team, pose, battery, ...) as UDP JSON at an adjustable rate.

One datagram per tick holds EVERY drone, so a consumer always gets a consistent
snapshot; consumers filter on each drone's "team" (a defender policy reads the
"attack" entries). Datagrams go to a list of host:port destinations, editable live
along with the rate. Wire format: see README.md in this folder.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Callable

import yaml

HERE = Path(__file__).resolve().parent
ROLES_PATH = HERE / "roles.yaml"
TEAMS = ("attack", "defense")
MIN_HZ, MAX_HZ = 0.5, 50.0
MTU_SAFE = 1400        # bytes; bigger datagrams fragment: fine on localhost / wired, lossy over WiFi


def load_roles(path: Path = ROLES_PATH) -> dict:
    """roles.yaml -> {"slots": {slot: team}, "drones": {name: team}, "destinations", "hz", "enabled"}."""
    cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = cfg or {}
    slots = {}
    for team, nums in (cfg.get("slots") or {}).items():
        for n in nums or []:
            slots[int(n)] = team
    return {
        "slots": slots,
        "drones": {str(k): v for k, v in (cfg.get("drones") or {}).items()},
        "destinations": list(cfg.get("destinations") or ["127.0.0.1:9100"]),
        "hz": float(cfg.get("hz", 10.0)),
        "enabled": bool(cfg.get("enabled", True)),
    }


def parse_destinations(text: str | list) -> list[tuple[str, int]]:
    """'127.0.0.1:9100, 10.0.0.5:9100' -> [(host, port), ...]; raises ValueError on junk."""
    items = text if isinstance(text, list) else [p for p in text.replace(";", ",").split(",")]
    out = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        host, sep, port = item.rpartition(":")
        if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
            raise ValueError(f"bad destination {item!r}: want host:port")
        out.append((host, int(port)))
    return out


class Publisher:
    """Background sender. `snapshot()` returns the list of per-drone dicts for one tick."""

    def __init__(self, snapshot: Callable[[], list[dict]], src: str, destinations, hz: float, enabled: bool):
        self.snapshot = snapshot
        self.src = src
        self.destinations = parse_destinations(destinations)
        self.hz = min(MAX_HZ, max(MIN_HZ, float(hz)))
        self.enabled = enabled
        self.seq = 0
        self.sent = 0
        self.errors = 0
        self.last_bytes = 0
        self.last_error = ""                  # last send failure; cleared by the next clean tick
        self.warning = ""                     # datagram too big to go unfragmented
        self.rate = 0.0                       # measured datagrams per second
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stop = threading.Event()
        self.wake = threading.Event()         # set when hz/enabled change, so a slow tick doesn't lag
        threading.Thread(target=self._loop, daemon=True).start()

    def configure(self, enabled=None, hz=None, destinations=None) -> None:
        if destinations is not None:
            self.destinations = parse_destinations(destinations)
        if hz is not None:
            self.hz = min(MAX_HZ, max(MIN_HZ, float(hz)))
        if enabled is not None:
            self.enabled = bool(enabled)
        self.wake.set()

    def message(self) -> bytes:
        msg = {"v": 1, "src": self.src, "seq": self.seq, "t": round(time.time(), 3),
               "hz": self.hz, "drones": self.snapshot()}
        return json.dumps(msg, separators=(",", ":")).encode()

    def _loop(self) -> None:
        window = []
        while not self.stop.is_set():
            t0 = time.time()
            if self.enabled and self.destinations:
                payload = self.message()
                self.last_bytes = len(payload)
                self.warning = (f"datagram {len(payload)} B > {MTU_SAFE} B: fragments (ok on localhost/wired,"
                                " lossy over WiFi)" if len(payload) > MTU_SAFE else "")
                error = ""
                for dest in self.destinations:
                    try:
                        self.sock.sendto(payload, dest)
                        self.sent += 1
                    except OSError as exc:
                        self.errors += 1
                        error = f"{dest[0]}:{dest[1]}: {exc}"
                self.last_error = error
                self.seq += 1
                window = [t for t in window if t0 - t < 2.0] + [t0]
                self.rate = (len(window) - 1) / (window[-1] - window[0]) if len(window) > 1 else 0.0
            else:
                window, self.rate = [], 0.0
            self.wake.wait(max(0.0, 1.0 / self.hz - (time.time() - t0)))
            self.wake.clear()

    def status(self) -> dict:
        return {"enabled": self.enabled, "hz": self.hz, "rate": round(self.rate, 1),
                "destinations": [f"{h}:{p}" for h, p in self.destinations], "seq": self.seq,
                "sent": self.sent, "errors": self.errors, "bytes": self.last_bytes, "error": self.last_error,
                "warning": self.warning}

    def close(self) -> None:
        self.stop.set()
        self.wake.set()

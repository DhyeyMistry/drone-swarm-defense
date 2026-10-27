"""Shared connection subroutine for drone_deployment_tests.

Which drones a test flies comes from ONE place: config/fleet.yaml `quick_connect`.
We keep only the quick-connect drones whose USB stick is plugged in right now
(config/dongles.json maps drone -> dongle), then bring each up on its own dongle with
SO_BINDTODEVICE sockets and start its video + state. Two quick-connect drones present
-> connect both; one -> connect one. No test hardcodes a drone name or interface.

connect_all() WAITS for every present quick-connect drone to answer before it returns,
so a test can take off knowing the whole group is up -- the "wait for all to connect,
then fly" behavior the real flight engine (tello_dual_video) already has.

It reuses the engine's per-dongle plumbing, so a test gets the same stick-pinned
sockets the swarm uses. djitellopy can't pin an interface (SO_BINDTODEVICE), which is
why single-drone djitellopy fails on a dongle that isn't the default route.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from fleet import quick_connect_drones
from tello_dongle_setup import REGISTRY_PATH, load_registry, present_ifaces
from tello_dual_video import (  # engine plumbing
    decode, dev_socket, forward, has_tello_lan, listen_state, next_session, sdk,
)

DD_ROOT = Path(__file__).resolve().parent.parent
CONFIG = DD_ROOT / "config"


@dataclass
class DroneLink:
    """One connected drone: its command socket plus the shared live-state dicts.

    frames/heights/batteries/motion/stats are the SAME dict objects across every link
    (keyed by drone name), mirroring the engine; link={"t": <last state packet time>}
    is per drone so each test can tell on its own when a drone's control link went stale.
    """
    name: str
    iface: str
    video_port: int
    cmd: socket.socket
    frames: dict
    heights: dict
    batteries: dict
    motion: dict
    link: dict
    stats: dict
    session: str
    decoder: threading.Thread | None = None
    _video_t: float = 0.0

    def connected(self, now: float | None = None) -> bool:
        """True while the stick holds the Tello LAN and state packets are fresh."""
        now = time.time() if now is None else now
        return has_tello_lan(self.iface) and (now - self.link["t"] < 1.5)

    def ensure_video(self, stop: threading.Event) -> None:
        """(Re)start the video decoder if it isn't running.

        streamon on this fleet is sometimes rejected with 'unkown command!' on the first
        try (a transient command-link desync -- the same thing that hits land), so we
        insist on an 'ok' before opening the decoder. Opening it against a stream that
        never started is exactly what produced the ~8 s 'decode open failed' timeout.
        The real flight engine does the same: it only starts decode when streamon == ok.
        Throttled; call it each tick while frames are missing, a no-op once video is up.
        """
        if self.decoder is not None and self.decoder.is_alive():
            return
        now = time.time()
        if now - self._video_t < 3.0:
            return
        self._video_t = now
        reply = insist(self.cmd, "streamon", self.name, attempts=4, gap=0.4)
        if reply != "ok":
            return   # don't open a decoder against a stream that didn't start
        self.decoder = threading.Thread(
            target=decode, args=(self.name, self.video_port, self.frames, self.stats, stop),
            daemon=True)
        self.decoder.start()


def insist(cmd: socket.socket, command: str, who: str = "", attempts: int = 6,
           gap: float = 1.0, timeout: float = 5.0) -> str:
    """Send an SDK command, retrying until the drone acks 'ok'.

    This fleet intermittently rejects the first command after a state change with
    'unkown command!' (seen on both streamon and land); a retry gets 'ok'. Each attempt
    and the raw reply are printed so a desync is visible in the log. Returns the last reply.
    """
    tag = f"{who} " if who else ""
    reply = ""
    for i in range(attempts):
        reply = sdk(cmd, command, timeout)
        if reply == "ok":
            if i:
                print(f"{tag}{command} -> ok (after {i} retr{'y' if i == 1 else 'ies'})", flush=True)
            return reply
        print(f"{tag}{command} try {i + 1}/{attempts} -> {reply!r}", flush=True)
        time.sleep(gap)
    return reply


def land(cmd: socket.socket) -> str:
    """Land, retrying until 'ok'. A drone that doesn't land is the worst failure."""
    return insist(cmd, "land", attempts=6, gap=1.0)


def present_quick_connect() -> list[dict]:
    """dongles.json entries for quick_connect drones whose stick is plugged in now."""
    want = quick_connect_drones()
    here = present_ifaces()
    by_drone = {d.get("drone"): d for d in load_registry(REGISTRY_PATH)["dongles"] if d.get("drone")}
    present = []
    for name in want:
        d = by_drone.get(name)
        if d is None:
            print(f"skip {name}: no dongle in config/dongles.json (run core/tello_dongle_setup.py)",
                  flush=True)
        elif d["iface"] not in here:
            print(f"skip {name}: stick {d['iface']} not plugged in", flush=True)
        else:
            present.append(d)
    return present


def bring_up(ssid: str, iface: str, timeout: float = 20.0) -> None:
    """Activate the drone's Wi-Fi profile on its stick. Non-fatal: if the AP is slow
    or not up yet nmcli blocks, so we cap it and let the connect loop retry instead of
    letting a TimeoutExpired kill the whole run."""
    try:
        subprocess.run(["nmcli", "con", "up", ssid, "ifname", iface],
                       capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    except Exception as exc:  # noqa: BLE001  (nmcli missing, profile gone, ...)
        print(f"  nmcli up {ssid} on {iface}: {exc}", flush=True)


def connect_all(stop: threading.Event, connect_timeout: float = 40.0) -> list[DroneLink]:
    """Connect every present quick-connect drone; return their links once ALL are up.

    Raises SystemExit if none are present, or if some never answer within
    connect_timeout -- a test should not fly a partial group.
    """
    present = present_quick_connect()
    if not present:
        raise SystemExit(
            "no quick_connect drones available: set config/fleet.yaml `quick_connect` "
            "and plug in the matching sticks (config/dongles.json maps them)."
        )

    session = next_session()
    frames: dict = {}
    heights: dict = {}
    batteries: dict = {}
    motion: dict = {}
    stats: dict = {}
    links: list[DroneLink] = []

    last_up: dict = {}
    for d in present:
        name, iface, port = d["drone"], d["iface"], d["video_port"]
        bring_up(name, iface)
        last_up[name] = time.time()
        cmd = dev_socket(iface, 8889)
        video = dev_socket(iface, 11111)
        state = dev_socket(iface, 8890)
        link = {"t": 0.0}
        stats[name] = {"session": session, "frames": 0, "drops": 0, "overflow": 0,
                       "tag_hit": 0, "tag_miss": 0, "file": ""}
        threading.Thread(target=forward, args=(video, port, stop), daemon=True).start()
        threading.Thread(target=listen_state,
                         args=(state, name, heights, batteries, motion, link, stop),
                         daemon=True).start()
        links.append(DroneLink(name, iface, port, cmd, frames, heights, batteries,
                               motion, link, stats, session))

    print(f"connecting quick_connect drones: {[dl.name for dl in links]}", flush=True)
    deadline = time.time() + connect_timeout
    pending = list(links)
    while pending and time.time() < deadline and not stop.is_set():
        for dl in list(pending):
            if not has_tello_lan(dl.iface):
                if time.time() - last_up[dl.name] > 6.0:   # re-nudge a slow association
                    bring_up(dl.name, dl.iface)
                    last_up[dl.name] = time.time()
                continue
            if sdk(dl.cmd, "command", 1.0) == "ok":
                dl.ensure_video(stop)   # streamon + start the decoder (retried in the fly loop)
                print(f"{dl.name} connected", flush=True)
                pending.remove(dl)
        if pending:
            stop.wait(1.0)

    if pending:
        raise SystemExit(
            f"gave up after {connect_timeout:.0f} s: these quick_connect drones never "
            f"answered 'command': {[dl.name for dl in pending]}. Are they powered on?"
        )
    print(f"all {len(links)} connected -> ready to fly", flush=True)
    return links

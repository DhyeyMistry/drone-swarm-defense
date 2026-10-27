"""One Tello behind one USB WiFi dongle: link, state, video, tag detection, control.

Every Tello is 192.168.10.1, so each Drone pins its sockets to its own dongle with
SO_BINDTODEVICE (same trick as core/tello_dual_video.py). That is what lets the GUI
drive several drones at once from one laptop.

Threads per drone:
  state    reads the 8890 state broadcast (battery, height, link alive)
  forward  copies the dongle's 11111 video to 127.0.0.1:<video_port>
  decode   H.264 -> BGR frames (PyAV), reopened whenever the stream stalls
  detect   AprilTags on the newest frame -> tag pose in the camera frame
  control  20 Hz rc loop while airborne: manual sticks, or the auto approach

Auto steers on the tag's image bearing and its range, not on the tag's estimated
orientation: for a 13 cm tag a few metres away the single-tag pose is ambiguous
(it flips between two mirror-image tilts), so "align with the tag normal" chases
noise. Bearing and range don't flip, and that is all "land in front of it" needs.
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector

DD_ROOT = Path(__file__).resolve().parent.parent
for _p in (DD_ROOT / "core", DD_ROOT / "drone_deployment_tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tello_dongle_setup import device_info  # noqa: E402
from tello_policy import DIST_PID, LAND_TOL_M, MAX_FORWARD, Pid, cam_in_tag, yaw_error_deg  # noqa: E402

CONFIG = DD_ROOT / "config"
TELLO = "192.168.10.1"
FALLBACK_CAMERA = CONFIG / "camera" / "tello_3.yaml"

# Safety limits. Kept deliberately conservative for flying indoors.
MIN_TAKEOFF_BATTERY = 20   # %, refuse takeoff below this
LAND_BATTERY = 10          # %, auto-land in flight below this
MAX_FLIGHT_S = 60.0        # auto mode lands after this long airborne
TAG_FRESH_S = 0.4          # a tag sighting older than this counts as lost
SEARCH_AFTER_S = 1.0       # lost this long -> slow yaw sweep in place
SEARCH_YAW = 20            # rc yaw while sweeping
SEARCH_LEG_S = 2.5         # seconds per left/right sweep leg
LOST_LAND_S = 10.0         # lost this long in auto -> land where it is
MANUAL_FRESH_S = 0.35      # manual sticks expire (deadman) after this
MANUAL_MAX = 40            # cap on manual rc magnitude
RC_HZ = 20.0
STREAM_WIDTH = 640           # px, width of the MJPEG sent to the browser
AIM_DEG = 6.0              # tag within this bearing counts as "facing it"
PIVOT_DEG = 15.0           # tag farther off than this: turn in place, don't advance
ELEV_DEADBAND_DEG = 8.0    # vertical image angle tolerated before climbing/sinking


def make_pids() -> dict[str, Pid]:
    return {"yaw": Pid(1.5, 0.0, 0.1, out_abs=30), "dist": Pid(*DIST_PID, out_abs=MAX_FORWARD),
            "up": Pid(1.2, 0.0, 0.0, out_abs=20)}


def approach_rc(obs: dict, standoff: float, pids: dict[str, Pid], dt: float) -> tuple[tuple[int, int, int, int], str]:
    """((right, forward, up, yaw), step) to face the tag and stop `standoff` m from it."""
    bearing, elev = obs["bearing"], obs["elev"]
    error = obs["range"] - standoff            # + too far, - too near
    yaw = int(pids["yaw"].step(bearing, dt))
    up = int(pids["up"].step(-elev, dt)) if abs(elev) > ELEV_DEADBAND_DEG else 0
    if abs(error) <= LAND_TOL_M and abs(bearing) < AIM_DEG:
        return (0, 0, 0, 0), "land"
    if abs(bearing) > PIVOT_DEG:
        pids["dist"].reset()
        return (0, 0, up, yaw), "aim"
    forward = int(pids["dist"].step(error, dt))
    if error > LAND_TOL_M and 0 <= forward < 8:     # beat the rc deadband
        forward = 8
    elif error < -LAND_TOL_M and -8 < forward <= 0:
        forward = -8
    return (0, forward, up, yaw), "close in" if forward > 0 else "back off"


def dev_socket(iface: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode())
    sock.bind(("0.0.0.0", port))
    return sock


def load_tag_sizes() -> dict[int, float]:
    """tag id -> black-square side in metres, from config/tags.yaml."""
    cfg = yaml.safe_load((CONFIG / "tags.yaml").read_text()) or {}
    return {int(k): float(v) / 1000.0 for k, v in (cfg.get("sizes_mm") or {}).items() if v}


def load_camera(drone: str) -> tuple[np.ndarray, bool]:
    """(K at 960x720, provisional?) for config/camera/<tello_x>.yaml, else tello_3's."""
    path = CONFIG / "camera" / f"{drone.lower().replace('-', '_')}.yaml"
    provisional = not path.exists()
    if provisional:
        path = FALLBACK_CAMERA
    text = path.read_text()
    data = yaml.safe_load(text)
    provisional = provisional or "PROVISIONAL" in text
    return np.asarray(data["camera_matrix"], dtype=float), provisional


class Drone:
    def __init__(self, iface: str, number: int | None, drone: str, video_port: int, tag_sizes: dict[int, float]):
        self.iface = iface
        self.number = number
        self.drone = drone                    # registry name, '' when unassigned
        self.video_port = video_port
        self.tag_sizes = tag_sizes
        self.K, self.provisional_cam = load_camera(drone or "unknown")

        # Link / telemetry
        self.ssid = ""
        self.ip = ""
        self.state_t = 0.0
        self.battery: int | None = None
        self.height: float | None = None
        self.temp: int | None = None
        self.tof: float | None = None          # downward range sensor, m
        self.vel: list[float] | None = None    # vgx/vgy/vgz, m/s (Tello's own axes)
        self.msg = "idle"

        # Video / detection
        self.streaming = False
        self.frame: np.ndarray | None = None
        self.frame_t = 0.0
        self.frames = 0
        self.tags: list[dict] = []
        self.tags_t = 0.0
        self._jpeg: tuple[float, bytes] | None = None
        self.pose: dict | None = None         # last camera pose in a tag frame, kept while stale

        # Flight
        self.flying = False
        self.busy = ""                        # "", "connecting", "takeoff", "landing"
        self.mode = "ground"                  # ground | hover | manual | auto | search
        self.target_tag: int | None = None
        self.standoff = 1.3
        self.manual = (0, 0, 0, 0)
        self.manual_t = 0.0
        self.takeoff_t = 0.0
        self.lost_since: float | None = None
        self.link_lost_in_flight = False

        self.cmd_lock = threading.Lock()
        self.stop = threading.Event()
        self._open()

    def _open(self) -> None:
        """Bind this dongle's sockets and start the per-drone threads."""
        self.cmd = dev_socket(self.iface, 8889)
        self.state_sock = dev_socket(self.iface, 8890)
        self.video_sock = dev_socket(self.iface, 11111)
        for target in (self._state_loop, self._forward_loop, self._decode_loop, self._detect_loop, self._control_loop):
            threading.Thread(target=target, daemon=True).start()

    # ------------------------------------------------------------------ helpers
    @property
    def name(self) -> str:
        return self.drone or self.ssid or self.iface

    @property
    def link(self) -> bool:
        return time.time() - self.state_t < 1.0

    def refresh_wifi(self) -> None:
        self.ssid, self.ip = device_info(self.iface)

    def sdk(self, cmd: str, timeout: float = 3.0) -> str:
        with self.cmd_lock:
            self.cmd.setblocking(False)
            try:
                while True:
                    self.cmd.recvfrom(256)
            except (BlockingIOError, OSError):
                pass
            self.cmd.settimeout(timeout)
            try:
                self.cmd.sendto(cmd.encode(), (TELLO, 8889))
                data, _ = self.cmd.recvfrom(256)
                return data.decode(errors="replace").strip()
            except (socket.timeout, OSError):
                return "timeout"

    def send_rc(self, sticks: tuple[int, int, int, int]) -> None:
        r, f, u, y = (int(v) for v in sticks)
        with self.cmd_lock:
            try:
                self.cmd.sendto(f"rc {r} {f} {u} {y}".encode(), (TELLO, 8889))
            except OSError:
                pass

    def _spawn(self, target, *args) -> None:
        threading.Thread(target=target, args=args, daemon=True).start()

    # ------------------------------------------------------------------ actions
    def connect(self, ssid: str | None = None) -> None:
        if self.busy:
            return
        self._spawn(self._connect, ssid or self.drone or self.ssid)

    def _connect(self, ssid: str) -> None:
        self.busy = "connecting"
        try:
            self.refresh_wifi()
            if ssid and self.ssid != ssid:
                # Each dongle keeps its own scan list; a freshly powered drone isn't
                # in it until this dongle rescans.
                self.msg = f"scanning for {ssid}"
                subprocess.run(["nmcli", "dev", "wifi", "rescan", "ifname", self.iface],
                               capture_output=True, timeout=15)
                for _ in range(10):
                    time.sleep(1)
                    scan = subprocess.run(["nmcli", "-t", "-f", "SSID", "dev", "wifi", "list", "ifname", self.iface,
                                           "--rescan", "no"], capture_output=True, text=True, timeout=15)
                    if ssid in scan.stdout.splitlines():
                        break
                self.msg = f"joining {ssid} WiFi"
                res = subprocess.run(["nmcli", "dev", "wifi", "connect", ssid, "ifname", self.iface],
                                     capture_output=True, text=True, timeout=40)
                if res.returncode != 0:
                    # A failed first try is common right after a rescan; try once more.
                    time.sleep(2)
                    res = subprocess.run(["nmcli", "dev", "wifi", "connect", ssid, "ifname", self.iface],
                                         capture_output=True, text=True, timeout=40)
                    if res.returncode != 0:
                        self.msg = f"WiFi join failed: {(res.stderr or res.stdout).strip()[:120]}"
                        return
            self.refresh_wifi()
            self.msg = "SDK handshake"
            for _ in range(3):
                if self.sdk("command") == "ok":
                    break
            else:
                self.streaming = False
                self.msg = "drone did not answer 'command' (is it on?)"
                return
            reply = self.sdk("streamon")
            self.streaming = reply == "ok"
            self.msg = "connected" if self.streaming else f"streamon: {reply}"
        except Exception as exc:  # noqa: BLE001 - surface anything in the GUI
            self.msg = f"connect error: {exc}"
        finally:
            self.busy = ""

    def disconnect(self) -> None:
        if self.flying:
            self.msg = "land before disconnecting"
            return
        self.sdk("streamoff", timeout=1.0)
        self.streaming = False
        subprocess.run(["nmcli", "dev", "disconnect", self.iface], capture_output=True, timeout=15)
        self.refresh_wifi()
        self.msg = "disconnected"

    def takeoff(self, then_auto: bool = False) -> None:
        if self.busy or self.flying:
            return
        if not self.link:
            self.msg = "no link: connect first"
            return
        if self.battery is not None and self.battery < MIN_TAKEOFF_BATTERY:
            self.msg = f"battery {self.battery}% < {MIN_TAKEOFF_BATTERY}%: charge it"
            return
        if then_auto and self.target_tag is None:
            self.msg = "pick a target tag first"
            return
        self._spawn(self._takeoff, then_auto)

    def _takeoff(self, then_auto: bool) -> None:
        self.busy = "takeoff"
        self.msg = "taking off"
        reply = self.sdk("takeoff", timeout=20.0)
        self.busy = ""
        if reply != "ok" and (self.height or 0) < 0.3:
            self.msg = f"takeoff failed: {reply}"
            return
        self.flying = True
        self.takeoff_t = time.time()
        self.link_lost_in_flight = False
        self.mode = "auto" if then_auto else "hover"
        self.msg = "auto: approaching tag" if then_auto else "hovering"

    def start_auto(self) -> None:
        if self.target_tag is None:
            self.msg = "pick a target tag first"
            return
        if not self.flying:
            self.takeoff(then_auto=True)
            return
        self.lost_since = None
        self.mode = "auto"
        self.msg = "auto: approaching tag"

    def hover(self) -> None:
        if self.flying:
            self.mode = "hover"
            self.msg = "hovering"

    def set_manual(self, sticks) -> None:
        if not self.flying:
            return
        self.manual = tuple(max(-MANUAL_MAX, min(MANUAL_MAX, int(v))) for v in sticks)
        self.manual_t = time.time()
        if any(self.manual):
            self.mode = "manual"

    def land(self, why: str = "land") -> None:
        if not self.flying or self.busy == "landing":
            return
        self._spawn(self._land, why)

    def _land(self, why: str) -> None:
        self.busy = "landing"
        self.msg = f"landing ({why})"
        self.send_rc((0, 0, 0, 0))
        reply = self.sdk("land", timeout=10.0)
        if reply != "ok":
            reply = self.sdk("land", timeout=10.0)
        self.busy = ""
        if reply == "ok" or (self.height is not None and self.height < 0.15):
            self.flying = False
            self.mode = "ground"
            self.msg = f"landed ({why})"
        else:
            self.msg = f"land reply: {reply} - still flying?"

    def emergency(self) -> None:
        """Cut the motors immediately. The drone falls."""
        with self.cmd_lock:
            try:
                self.cmd.sendto(b"emergency", (TELLO, 8889))
            except OSError:
                pass
        self.flying = False
        self.mode = "ground"
        self.msg = "EMERGENCY motor stop sent"

    def close(self) -> None:
        if self.flying:
            self._land("shutdown")
        if self.streaming:
            self.sdk("streamoff", timeout=1.0)
        self.stop.set()

    # ------------------------------------------------------------------ threads
    def _state_loop(self) -> None:
        self.state_sock.settimeout(0.5)
        while not self.stop.is_set():
            try:
                data, _ = self.state_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                time.sleep(0.5)
                continue
            self.state_t = time.time()
            fields = dict(p.split(":", 1) for p in data.decode(errors="replace").split(";") if ":" in p)
            try:
                self.battery = int(fields["bat"])
                self.height = int(fields["h"]) / 100.0
                self.temp = int(fields.get("temph", 0))
                self.tof = int(fields["tof"]) / 100.0 if "tof" in fields else None
                self.vel = [int(fields[k]) / 10.0 for k in ("vgx", "vgy", "vgz")]   # dm/s -> m/s
            except (KeyError, ValueError):
                pass

    def _forward_loop(self) -> None:
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_sock.settimeout(0.5)
        while not self.stop.is_set():
            try:
                data, _ = self.video_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                time.sleep(0.5)
                continue
            out.sendto(data, ("127.0.0.1", self.video_port))

    def _decode_loop(self) -> None:
        url = f"udp://@127.0.0.1:{self.video_port}?overrun_nonfatal=1&fifo_size=5000000"
        while not self.stop.is_set():
            if not self.streaming:
                time.sleep(0.3)
                continue
            try:
                container = av.open(url, timeout=(6, 3))
            except av.FFmpegError:
                time.sleep(0.5)
                continue
            try:
                for frame in container.decode(video=0):
                    if self.stop.is_set() or not self.streaming:
                        break
                    self.frame = frame.to_ndarray(format="bgr24")
                    self.frame_t = time.time()
                    self.frames += 1
            except av.FFmpegError:
                pass   # stream stalled; reopen
            finally:
                container.close()

    def _detect_loop(self) -> None:
        det = Detector(families="tag36h11", nthreads=2, quad_decimate=2.0)
        seen = None
        while not self.stop.is_set():
            frame = self.frame
            if frame is None or frame is seen:
                time.sleep(0.02)
                continue
            seen = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            s = w / 960.0
            fx, fy, cx, cy = self.K[0, 0] * s, self.K[1, 1] * s, self.K[0, 2] * s, self.K[1, 2] * s
            tags = []
            for t in det.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=1.0):
                if t.decision_margin < 10 or t.pose_t is None:
                    continue
                size = self.tag_sizes.get(int(t.tag_id))
                R = np.asarray(t.pose_R, dtype=float)
                entry = {"id": int(t.tag_id), "corners": t.corners.tolist(), "center": t.center.tolist(),
                         "bearing": math.degrees(math.atan2(t.center[0] - cx, fx)),   # + = tag right of centre
                         "elev": math.degrees(math.atan2(t.center[1] - cy, fy))}      # + = tag below centre
                if size:
                    pose_t = np.asarray(t.pose_t, dtype=float).reshape(3) * size
                    cam = cam_in_tag(R.tolist(), pose_t.tolist())
                    entry.update(range=float(np.linalg.norm(pose_t)), ahead=float(-cam[2]),
                                 cam=cam, yaw_err=yaw_error_deg(R.tolist()))
                tags.append(entry)
            self.tags = tags
            self.tags_t = time.time()
            posed = [t for t in tags if "cam" in t]
            if posed:
                best = next((t for t in posed if t["id"] == self.target_tag), min(posed, key=lambda t: t["range"]))
                x, y, z = best["cam"]
                self.pose = {"tag": best["id"], "xyz": [round(x, 2), round(y, 2), round(z, 2)],
                             "yaw": round(-best["yaw_err"], 1), "stamp": self.tags_t}

    def target_obs(self) -> dict | None:
        if self.target_tag is None or time.time() - self.tags_t > TAG_FRESH_S:
            return None
        for t in self.tags:
            if t["id"] == self.target_tag and "cam" in t:
                return t
        return None

    def _control_loop(self) -> None:
        pids = make_pids()
        last = time.time()
        while not self.stop.is_set():
            time.sleep(1.0 / RC_HZ)
            now = time.time()
            dt, last = now - last, now
            if not self.flying or self.busy:
                pids = make_pids()
                continue

            # Link watchdog: if the link dropped mid-flight, land as soon as it is back.
            if not self.link:
                self.link_lost_in_flight = True
                continue
            if self.link_lost_in_flight:
                self.land("link was lost")
                continue
            if self.battery is not None and self.battery < LAND_BATTERY:
                self.land(f"battery {self.battery}%")
                continue

            sticks = (0, 0, 0, 0)
            if self.mode == "manual":
                if now - self.manual_t < MANUAL_FRESH_S:
                    sticks = self.manual
                else:
                    self.mode = "hover"
            elif self.mode in ("auto", "search"):
                if now - self.takeoff_t > MAX_FLIGHT_S:
                    self.land(f"{MAX_FLIGHT_S:.0f} s cap")
                    continue
                obs = self.target_obs()
                if obs is not None:
                    self.lost_since = None
                    self.mode = "auto"
                    sticks, step = approach_rc(obs, self.standoff, pids, dt)
                    self.msg = f"auto: {step}, tag {obs['id']} {obs['range']:.2f} m, {obs['bearing']:+.0f} deg"
                    if step == "land":
                        self.land(f"arrived at tag {obs['id']}")
                        continue
                else:
                    for pid in pids.values():
                        pid.reset()
                    self.lost_since = self.lost_since or now
                    lost = now - self.lost_since
                    if lost > LOST_LAND_S:
                        self.land(f"tag {self.target_tag} lost {LOST_LAND_S:.0f} s")
                        continue
                    if lost > SEARCH_AFTER_S:
                        self.mode = "search"
                        leg = int((lost - SEARCH_AFTER_S) // SEARCH_LEG_S)
                        sticks = (0, 0, 0, SEARCH_YAW if leg % 2 == 0 else -SEARCH_YAW)
                        self.msg = f"search: looking for tag {self.target_tag} ({lost:.0f} s)"
            self.send_rc(sticks)   # also the keepalive that stops the Tello's 15 s auto-land

    # ------------------------------------------------------------------ views
    def jpeg(self) -> bytes | None:
        frame, t = self.frame, self.frame_t
        if frame is None:
            return None
        if self._jpeg and self._jpeg[0] == t:
            return self._jpeg[1]
        img = frame.copy()
        fresh = time.time() - self.tags_t < TAG_FRESH_S
        for tag in self.tags if fresh else []:
            pts = np.asarray(tag["corners"], dtype=np.int32)
            color = (0, 220, 0) if tag["id"] == self.target_tag else (0, 200, 255)
            cv2.polylines(img, [pts], True, color, 3)
            label = f"#{tag['id']}"
            if "range" in tag:
                label += f" {tag['range']:.2f}m"
            cx, cy = (int(v) for v in tag["center"])
            cv2.putText(img, label, (cx - 40, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(img, label, (cx - 40, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if img.shape[1] > STREAM_WIDTH:   # the grid tiles are small; keep 8 streams cheap
            img = cv2.resize(img, (STREAM_WIDTH, img.shape[0] * STREAM_WIDTH // img.shape[1]), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        self._jpeg = (t, buf.tobytes())
        return self._jpeg[1]

    def telemetry(self, slot: int, team: str) -> dict:
        """This drone's entry in a telemetry datagram (format: drone_telemetry/README.md)."""
        pose = None
        if self.pose:
            pose = {k: v for k, v in self.pose.items() if k != "stamp"}
            pose["age"] = round(time.time() - self.pose["stamp"], 2)
        return {"name": self.name, "slot": slot, "team": team, "link": self.link, "fly": self.flying,
                "mode": self.busy or self.mode, "bat": self.battery, "h": self.height,
                "tof": self.tof, "vel": self.vel, "pose": pose}

    def status(self) -> dict:
        now = time.time()
        fresh = now - self.tags_t < TAG_FRESH_S
        if self.streaming and not self.link and not self.busy and self.state_t:
            hint = []
            if self.battery is not None and self.battery < 15:
                hint.append(f"battery {self.battery}%")
            if self.temp is not None and self.temp >= 85:
                hint.append(f"{self.temp} C hot")
            self.msg = "drone stopped answering" + (f" (last seen: {', '.join(hint)})" if hint else "") \
                + " - power-cycle it, then Connect"
        return {
            "name": self.name, "iface": self.iface, "number": self.number, "drone": self.drone,
            "ssid": self.ssid, "ip": self.ip, "link": self.link, "battery": self.battery,
            "height": self.height, "temp": self.temp, "streaming": self.streaming,
            "video": now - self.frame_t < 1.5, "flying": self.flying, "busy": self.busy,
            "mode": self.mode, "msg": self.msg, "target_tag": self.target_tag, "standoff": self.standoff,
            "provisional_cam": self.provisional_cam, "sim": getattr(self, "is_sim", False),
            "tags": [{"id": t["id"], "range": round(t.get("range", 0), 2) or None} for t in self.tags] if fresh else [],
            "airborne_s": round(now - self.takeoff_t) if self.flying else None,
        }

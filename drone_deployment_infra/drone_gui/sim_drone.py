"""A fake Tello for testing the GUI and the multi-drone logic without hardware.

SimDrone replaces only the radio side of Drone: no sockets, no nmcli. It keeps a
simple kinematic model (rc sticks -> velocity, same scale as tello_policy.predict_tag),
renders what the camera would see -- its AprilTag on a wall, projected through the
real camera matrix -- and feeds those frames to the SAME detection and control
loops a real drone uses. So "Auto" in sim mode exercises the real approach-and-land
code end to end.
"""

from __future__ import annotations

import math
import random
import threading
import time

import cv2
import numpy as np

from tello_unit import Drone

SIM_FPS = 12.0
TAG_HEIGHT_M = 1.0          # tag centre above the floor
TAKEOFF_HEIGHT_M = 0.8
WALL = (196, 200, 204)      # BGR


def tag_texture(tag_id: int, px: int = 240) -> np.ndarray:
    """tag36h11 marker (black square edge to edge) padded with a one-cell white border."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    cell = px // 10
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, cell * 8)
    # OpenCV draws its AprilTag markers rotated 180 deg from the official tag36h11 images
    # (the ones printed for the real tags): pupil_apriltags then puts corner 0 top-right
    # instead of bottom-left and every pose comes out with x and y negated. Undo that.
    marker = cv2.rotate(marker, cv2.ROTATE_180)
    return cv2.copyMakeBorder(marker, cell, cell, cell, cell, cv2.BORDER_CONSTANT, value=255)


class SimDrone(Drone):
    is_sim = True

    def __init__(self, index: int, tag_id: int, tag_sizes: dict[int, float]):
        self.sim_tag = tag_id
        self.texture = tag_texture(tag_id)
        self.sim_rc = (0, 0, 0, 0)
        self.sim_lock = threading.Lock()
        self.reset_pose()
        super().__init__(f"sim{index}", index, f"SIM-{index}", 0, tag_sizes)
        self.provisional_cam = False           # the sim renders with this same K, so it is exact
        self.battery = random.randint(70, 100)
        self.temp = 50

    def reset_pose(self) -> None:
        """Somewhere 2-3 m in front of the tag, a little off-axis, on the floor."""
        self.x = random.uniform(-0.6, 0.6)     # right of the tag, m (as seen facing it)
        self.z = random.uniform(2.2, 3.2)      # out from the wall, m
        self.psi = random.uniform(-20, 20)     # yaw, deg, + = turned right
        self.h = 0.1                           # camera height above the floor, m

    # ---------------------------------------------------------------- radio stubs
    def _open(self) -> None:
        for target in (self._sim_loop, self._detect_loop, self._control_loop):
            threading.Thread(target=target, daemon=True).start()

    def refresh_wifi(self) -> None:
        pass

    def _connect(self, ssid: str) -> None:
        self.busy = "connecting"
        self.msg = "joining (sim)"
        time.sleep(0.8)
        self.ssid, self.ip = self.drone, "sim"
        self.streaming = True
        self.msg = "connected (sim)"
        self.busy = ""

    def disconnect(self) -> None:
        if self.flying:
            self.msg = "land before disconnecting"
            return
        self.streaming = False
        self.ssid = self.ip = ""
        self.state_t = 0.0
        self.msg = "disconnected"

    def sdk(self, cmd: str, timeout: float = 3.0) -> str:
        if cmd == "takeoff":
            for _ in range(25):
                self.h = min(TAKEOFF_HEIGHT_M, self.h + 0.04)
                time.sleep(0.1)
            return "ok"
        if cmd == "land":
            while self.h > 0.1:
                self.h = max(0.1, self.h - 0.05)
                time.sleep(0.1)
            return "ok"
        return "ok"

    def send_rc(self, sticks) -> None:
        with self.sim_lock:
            self.sim_rc = tuple(int(v) for v in sticks)

    def emergency(self) -> None:
        self.h = 0.1
        self.sim_rc = (0, 0, 0, 0)
        self.flying = False
        self.mode = "ground"
        self.msg = "EMERGENCY motor stop (sim)"

    def reset(self) -> None:
        if not self.flying and not self.busy:
            self.reset_pose()
            self.msg = "pose reset (sim)"

    def close(self) -> None:
        self.stop.set()

    def telemetry(self, slot: int, team: str) -> dict:
        """Adds ground truth in the same convention as `pose` (camera in the tag frame)."""
        entry = super().telemetry(slot, team)
        entry["truth"] = {"tag": self.sim_tag, "xyz": [round(self.x, 2), round(TAG_HEIGHT_M - self.h, 2),
                                                        round(-self.z, 2)], "yaw": round(self.psi, 1)}
        return entry

    # ---------------------------------------------------------------- simulation
    def _sim_loop(self) -> None:
        last = time.time()
        drain = 0.0
        while not self.stop.is_set():
            time.sleep(1.0 / SIM_FPS)
            now = time.time()
            dt, last = now - last, now
            airborne = self.flying and not self.busy
            if airborne:
                with self.sim_lock:
                    right, fwd, up, yaw = self.sim_rc
                self.psi += (yaw / 100.0) * 60.0 * dt
                p = math.radians(self.psi)
                f = (math.sin(p), -math.cos(p))            # forward, in (x, z)
                r = (math.cos(p), math.sin(p))             # right, in (x, z)
                v_f = (fwd / 100.0) * 0.8 + random.gauss(0, 0.02)
                v_r = (right / 100.0) * 0.8 + random.gauss(0, 0.02)
                self.x += (f[0] * v_f + r[0] * v_r) * dt
                self.z = max(0.25, self.z + (f[1] * v_f + r[1] * v_r) * dt)
                self.h = min(2.5, max(0.1, self.h + (up / 100.0) * 0.6 * dt))
                self.vel = [round(v_f, 2), round(v_r, 2), round(-(up / 100.0) * 0.6, 2)]   # Tello: x fwd, y right, z down
            else:
                self.vel = [0.0, 0.0, 0.0]
            drain += dt / (20.0 if airborne else 90.0)
            if drain >= 1.0:
                drain -= 1.0
                self.battery = max(0, self.battery - 1)
            if self.streaming:
                self.state_t = now                          # "state packets" arrive while linked
                self.height = max(0.0, self.h - 0.1)
                self.tof = self.h
                self.frame = self._render()
                self.frame_t = now
                self.frames += 1

    def _render(self) -> np.ndarray:
        img = np.empty((720, 960, 3), np.uint8)
        img[:] = WALL
        # Floor below the wall's base line, which drops in the image as the drone climbs.
        base = int(self.K[1, 2] + self.K[1, 1] * self.h / max(self.z, 0.25))
        img[max(0, min(720, base)):] = (120, 135, 150)

        size = self.tag_sizes.get(self.sim_tag, 0.132)
        b = size * 10 / 16                                  # half-width of tag + white border
        p = math.radians(self.psi)
        f = np.array([math.sin(p), 0.0, -math.cos(p)])
        r = np.array([math.cos(p), 0.0, math.sin(p)])
        down = np.array([0.0, -1.0, 0.0])
        cam = np.array([self.x, self.h, self.z])
        pts = []
        for u, v in ((-b, b), (b, b), (b, -b), (-b, -b)):  # TL TR BR BL as seen facing the tag
            d = np.array([u, TAG_HEIGHT_M + v, 0.0]) - cam
            X, Y, Z = d @ r, d @ down, d @ f
            if Z < 0.1:
                return img                                  # tag behind the camera
            pts.append((self.K[0, 0] * X / Z + self.K[0, 2], self.K[1, 1] * Y / Z + self.K[1, 2]))
        n = self.texture.shape[0] - 1
        src = np.float32([(0, 0), (n, 0), (n, n), (0, n)])
        H = cv2.getPerspectiveTransform(src, np.float32(pts))
        warped = cv2.warpPerspective(self.texture, H, (960, 720), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(self.texture, 255), H, (960, 720), borderValue=0)
        img[mask > 127] = warped[mask > 127][:, None]
        return img

#!/usr/bin/env python3
"""Swarm formation in front of AprilTag 10, using tag 10 plus (optionally) a tag MAP.

Per drone:
  1. CLIMB to --start-height after takeoff (no turning)
  2. SEARCH: look 2 s, then turn 20 deg + pause, up to a full turn (-> that drone lands)
  3. LOCKED: sees ANY trusted mapped tag -> knows its position relative to tag 10.
     Keeps that tag centered, goes to tag-10 height, IN PLACE. Waits for the others.
  4. slots: non-crossing arc in front of tag 10, assigned once, kept (re-matched only
     when a drone joins/leaves)
  5. APPROACH: flies to its slot (sideways/backwards as needed) while its camera stays
     on its REFERENCE tag, so it never loses its position. Hands off to a better tag
     when one appears; tag 10 is always preferred. Keeps out of other drones' view of
     their tags, and keeps spacing.
  6. FACE10: at the slot, turns in place straight to tag 10 (direction known from the
     pose), then HOLDING: locked on tag 10, keeps its spot (moves only to make room).
  7. all HOLDING in their slots SETTLE_S -> hold --hold s -> all land.
Tag lost: freeze. Another drone known to be blocking the view -> wait (BLOCKED, up to
15 s). Otherwise recover in place (turn back toward the tag, back to its height), and
land in place after RECOVER_S. Never moves sideways without a tag in view.
Map check: whenever tag 10 and another tag are seen together, that tag's observed
position is compared with the map; a tag that disagrees is switched off for the flight.

DEFAULT IS DRY-RUN (no takeoff / no motion). --fly to fly.
Reads drone_deployment_infra/config/ via core/paths.py: dongles.json, camera/tello_<n>.yaml
(default.yaml if uncalibrated), map.yaml + tags.yaml (--no-map: tag 10 only). Libs: numpy opencv-python pupil-apriltags av pyyaml

  python swarm_formation.py --only 3                 # dry-run
  python swarm_formation.py --only 1,3,5 --fly
  python swarm_formation.py --only 1,3,5 --fly --no-map

Keys (camera windows): q = land all, SPACE = EMERGENCY motor stop all. Ctrl+C = land all.
"""
import argparse
import csv
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from itertools import permutations

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from paths import CAMERA_DIR, DONGLES_PATH, MAP_PATH, TAGS_PATH, camera_file  # noqa: E402  (config/ lives in core)

# ======================= SETTINGS =======================
TAG_ID = 10
TAG_SIZE_M = 0.5715          # tag 10 black square (22.5 in), used when no tags.yaml

START_HEIGHT_CM = 170
CLIMB_TIMEOUT_S = 8.0
LOOK_S = 2.0
SEARCH_STEP_DEG = 20
SEARCH_LOOK_S = 0.8
SEARCH_YAW_RC = 25
SEARCH_MAX_DEG = 380
GLIMPSE_HOLD_S = 1.5

ARC_RADIUS_M = 1.3           # tag 10 (57 cm) must fit the camera view from every slot
ARC_RADIUS_MAX_M = 2.0
MIN_SPACING_M = 0.7
MAX_VIEW_ANGLE_DEG = 55
HEIGHT_OFFSET_M = 0.0        # slot height vs tag 10 center (+ = higher), --slot-offset

KP_POS = 40.0
KP_YAW = 1.0
KP_HEIGHT = 0.5
MAX_RC_XY = 15
MAX_RC_Z = 20
MAX_RC_YAW = 25
DEADBAND_POS_M = 0.05
DEADBAND_YAW_DEG = 3.0
MIN_RC = 8
MIN_RC_XY = 14               # near the slot: smaller horizontal rc doesn't move a Tello
KI_POS = 15.0                # rc per (m * s): builds up if the drone stays off its slot
I_ZONE_M = 0.5               # integral only active this close to the slot
I_MAX_RC = 10                # integral contribution cap
POSE_ALPHA = 0.35

ALIGN_H_M = 0.15
ALIGN_TIMEOUT_S = 6.0
WAIT_ALL_S = 6.0

D_SAFE_M = 0.6
D_STOP_M = 0.4
K_REP = 60.0
LOST_DRIFT_MPS = 0.15
MIN_TAG_DIST_M = 0.8
DRONE_RADIUS_M = 0.12        # for view-cone checks
K_CONE = 50.0                # push out of another drone's view cone (rc per m)

TOL_POS_M = 0.10
TOL_YAW_DEG = 6.0
SETTLE_S = 2.0

FIX_FRESH_S = 0.5
RECOVER_S = 4.0
BLOCKED_MAX_S = 15.0
INPLACE_NOFIX_S = 10.0       # LOCKED/FACE10 may turn/climb in place without a tag this long
RECOVER_YAW_RC = 12
JUMP_MAX_M = 0.5
JUMP_ACCEPT_AFTER_S = 1.0
MIN_DECISION_MARGIN = 15
MIN_TAG_SIDE_PX = 15
MAX_SINGLE_ERR_PX = 3.0
JOINT_MAX_RMS_PX = 3.0
MAPCHECK_M = 0.15            # observed vs mapped tag position disagreement limit
MAPCHECK_MIN_BAD = 5

# --- --fast profile: overrides applied when --fast is given -------------------
FAST = dict(                 # tuned from logs 2026-10-17 (normal mode: 52-70 s to formation)
    TAKEOFF_STAGGER_S=0.2,   # was 0.5
    START_HEIGHT_CM=155,     # was 170   formation settled at ~156 cm (tag 10 center)
    LOOK_S=0.8,              # was 2.0   first look before turning
    SEARCH_LOOK_S=0.4,       # was 0.8   pause after each search turn
    SEARCH_STEP_DEG=30,      # was 20    camera sees ~54 deg, steps still overlap
    SEARCH_YAW_RC=70,        # was 25    measured: rc 25 turned only ~10 deg/s
    WAIT_ALL_S=0.0,          # was 6     approach as soon as localized (re-matched on join)
    ALIGN_TIMEOUT_S=0.0,     # was 6     height is fixed while approaching instead
    MAX_RC_XY=35,            # was 15    far: fast; near the slot speed = KP_POS*dist (unchanged)
    MAX_RC_Z=50,             # was 20    measured: rc 20 climbed only ~7.5 cm/s
    MAX_RC_YAW=60,           # was 25    big turns fast; small ones proportional (unchanged)
    SETTLE_S=1.0,            # was 2
)
FLIP_CMD = "flip b"           # back flip = away from tag 10 and from the neighbours
FLIP_MIN_BATTERY = 55        # Tello refuses flips below 50 %
FLIP_GRACE_S = 3.0           # after a flip, losing the tags is expected (no freeze/land)
FLIP_RETURN_TIMEOUT_S = 8.0  # next drone flips once this one is back in its slot (or after this)
CAUTIOUS_RC_XY = 15          # after a tag loss while moving: this cap for SLOW_AFTER_LOSS_S
SLOW_AFTER_LOSS_S = 3.0
SPEED_BUBBLE_M = 0.20        # extra spacing around a drone moving at full fast speed
TAKEOFF_STAGGER_S = 0.5

MIN_BATTERY_FLY = 40
LAND_BATTERY = 15
LOOP_HZ = 20
RES = (960, 720)
# ========================================================

DRONE_ADDR = ("192.168.10.1", 8889)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
NOFLY = {"command", "streamon", "streamoff", "battery?", "height?"}
RC_RE = re.compile(r"^rc (-?\d+) (-?\d+) (-?\d+) (-?\d+)$")
C_FLIP = np.diag([1.0, -1.0, -1.0])
# Internal frame (tag 10 = origin): x right, y DOWN, z INTO the tag; in front = -z.


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def obj_points(size):
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def rotz90(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def make_slots(n):
    r = ARC_RADIUS_M
    while r <= ARC_RADIUS_MAX_M + 1e-9:
        if n == 1:
            phis = [0.0]
        else:
            dphi = 2 * math.asin(min(1.0, MIN_SPACING_M / (2 * r)))
            if (n - 1) * dphi / 2 > math.radians(MAX_VIEW_ANGLE_DEG):
                r += 0.1
                continue
            phis = [(i - (n - 1) / 2) * dphi for i in range(n)]
        return [np.array([r * math.sin(f), -HEIGHT_OFFSET_M, -r * math.cos(f)]) for f in phis], r
    raise ValueError(f"{n} drones don't fit (spacing {MIN_SPACING_M} m, angle "
                     f"{MAX_VIEW_ANGLE_DEG} deg, radius <= {ARC_RADIUS_MAX_M} m)")


def match_slots(positions, slots):
    """Min-total-length matching: straight paths never cross."""
    nums = list(positions)
    best, cost_best = {}, float("inf")
    for perm in permutations(range(len(slots)), len(nums)):
        cost = sum(math.hypot(slots[k][0] - positions[n][0], slots[k][2] - positions[n][2])
                   for n, k in zip(nums, perm))
        if cost < cost_best:
            best, cost_best = dict(zip(nums, perm)), cost
    return best


def kick(v):
    """Non-zero rc below MIN_RC is raised to MIN_RC (Tello ignores tiny commands)."""
    v = int(v)
    return 0 if v == 0 else (max(v, MIN_RC) if v > 0 else min(v, -MIN_RC))


def face_tag_yaw(p):
    return math.degrees(math.atan2(-p[0], -p[2]))


def control(p, psi, target, others, cap=None, integ=None):
    """rc toward target, facing tag 10, with spacing. others: [(p, extra_radius)].
    cap: horizontal speed limit (default MAX_RC_XY). Near the target the speed is
    KP_POS * distance, so precision close in does not depend on the cap."""
    cap = MAX_RC_XY if cap is None else cap
    e = target - p
    e_yaw = wrap_deg(face_tag_yaw(p) - psi)
    if np.linalg.norm(e) < DEADBAND_POS_M:
        e = np.zeros(3)
    v = KP_POS * e
    if integ is not None:
        v = v + KI_POS * integ
    scale, rep = 1.0, np.zeros(3)
    for q, extra in others:
        dv = np.array([p[0] - q[0], 0.0, p[2] - q[2]])
        dist = float(np.linalg.norm(dv))
        ds, dstop = D_SAFE_M + extra, D_STOP_M + extra
        if dist < ds:
            u = dv / dist if dist > 1e-6 else np.array([1.0, 0, 0])
            rep += K_REP * (ds - dist) * u
            scale = min(scale, max(0.0, (dist - dstop) / (ds - dstop)))
    v = v * scale + rep
    dh = math.hypot(p[0], p[2])
    if 1e-6 < dh < MIN_TAG_DIST_M:
        u = np.array([p[0], 0.0, p[2]]) / dh
        inward = -float(v @ u)
        if inward > 0:
            v = v + inward * u
        v = v + 10.0 * u
    nxz = math.hypot(v[0], v[2])
    if nxz > cap:
        v[0], v[2] = v[0] * cap / nxz, v[2] * cap / nxz
    v[1] = float(np.clip(v[1], -MAX_RC_Z, MAX_RC_Z))
    s, c = math.sin(math.radians(psi)), math.cos(math.radians(psi))
    fwd, right, up = v[0] * s + v[2] * c, v[0] * c - v[2] * s, -v[1]
    mag = math.hypot(fwd, right)
    if 0.5 < mag < MIN_RC_XY:                    # too small to move a Tello: scale up
        fwd, right = fwd * MIN_RC_XY / mag, right * MIN_RC_XY / mag
    yaw = 0.0 if abs(e_yaw) < DEADBAND_YAW_DEG else float(np.clip(KP_YAW * e_yaw, -MAX_RC_YAW, MAX_RC_YAW))
    return tuple(int(round(x)) for x in (right, fwd, up, yaw)), float(np.linalg.norm(target - p)), e_yaw


def iface_ip(iface):
    out = subprocess.run(["ip", "-j", "-4", "addr", "show", iface], capture_output=True, text=True).stdout or "[]"
    for dev in json.loads(out):
        for a in dev.get("addr_info", []):
            if a.get("local", "").startswith("192.168.10."):
                return a["local"]
    return None


def active_ssid(iface):
    out = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi", "list", "ifname", iface,
                          "--rescan", "no"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("yes:"):
            return line[4:].replace("\\:", ":")
    return None


def load_camera(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    K = c.get("K", c.get("camera_matrix"))
    D = c.get("dist", c.get("dist_coeffs", c.get("D")))
    K = K.get("data") if isinstance(K, dict) else K
    D = D.get("data") if isinstance(D, dict) else D
    if K is None or D is None:
        raise ValueError(f"{path}: no K/dist (keys {list(c)})")
    return np.array(K, np.float64).reshape(3, 3), np.array(D, np.float64).reshape(-1)


def parallel(drones, fn):
    res, ths = {}, []
    for d in drones:
        def w(d=d):
            try:
                res[d.num] = fn(d)
            except Exception as ex:
                d.log(f"error: {ex}")
                res[d.num] = False
        t = threading.Thread(target=w, daemon=True)
        t.start()
        ths.append(t)
    for t in ths:
        while t.is_alive():
            t.join(0.2)
    return res


def preflight(d, fly):
    if not iface_ip(d.iface):
        d.log(f"FAIL: {d.iface} has no 192.168.10.x (not connected?)")
        return False
    ssid = active_ssid(d.iface)
    if ssid != d.ssid:
        d.log(f"FAIL: {d.iface} on {ssid!r}, expected {d.ssid!r}")
        return False
    threading.Thread(target=d.state_loop, daemon=True).start()
    for _ in range(5):
        if d.send("command", 3) == "ok":
            break
        time.sleep(1)
    else:
        d.log("FAIL: no reply to command")
        return False
    b = d.send("battery?", 5)
    d.battery = int(b) if b.isdigit() else None
    if fly and (d.battery is None or d.battery < MIN_BATTERY_FLY):
        d.log(f"FAIL: battery {b}% < {MIN_BATTERY_FLY}%")
        return False
    threading.Thread(target=d.video_loop, daemon=True).start()
    for _ in range(3):
        r = d.send("streamon", 5)
        if r == "ok":
            break
        if "unknown" in r or "unkown" in r:
            d.log("FAIL: drone doesn't know streamon -> update firmware")
            return False
        time.sleep(1)
    else:
        d.log("FAIL: streamon not confirmed")
        return False
    t0, resent = time.time(), False
    while d.frames < 5 and time.time() - t0 < 15:
        if not resent and time.time() - t0 > 6:
            d.send("streamon", 5)
            resent = True
        time.sleep(0.1)
    if d.frames < 5:
        d.log("FAIL: no video" if d.packets == 0 else f"FAIL: {d.packets} packets, {d.frames} frames")
        return False
    if d.frame.shape[1::-1] != RES:
        d.log(f"FAIL: frame {d.frame.shape[1]}x{d.frame.shape[0]}, expected {RES}")
        return False
    threading.Thread(target=d.perception_loop, daemon=True).start()
    threading.Thread(target=d.battery_loop, daemon=True).start()
    t0 = time.time()
    while not d.st and time.time() - t0 < 3:
        time.sleep(0.1)
    if not d.st:
        d.log("WARNING: no status packets -> no height sensor / IMU (climb + recovery limited)")
    d.log(f"PASS (battery {d.battery}%, video ok, status {'ok' if d.st else 'missing'})")
    return True


def land_async(d, why, fly):
    if not fly:
        if not d.dry_noted:
            d.log(f"{why} -> would LAND in place (dry-run)")
            d.dry_noted = True
        return
    if d.landing or d.landed:
        return
    d.landing = True
    d.log(f"{why} -> LANDING in place")
    d.send_noreply("rc 0 0 0 0")
    d.land_thread = threading.Thread(target=d.land, daemon=True)
    d.land_thread.start()


def load_tag_files(map_path, tags_path):
    """Returns (origin_id, {id: (R_10k, t_10k)} in internal frames, {id: size_m})."""
    with open(tags_path) as f:
        tcfg = yaml.safe_load(f)
    sizes = {int(k): float(v) / 1000.0 for k, v in (tcfg.get("sizes_mm") or {}).items()
             if v is not None}
    with open(map_path) as f:
        m = yaml.safe_load(f)
    origin = int(m.get("origin_tag", tcfg.get("origin_tag", 10)))
    frame = str(m.get("frame", ""))
    if "y up" not in frame or "out of" not in frame:
        print(f"WARNING: map frame says {frame!r}; expected 'x right, y up, z out of the "
              f"tag face'. Check the conversion.")
    poses = {}
    for k, v in (m.get("tags") or {}).items():
        k = int(k)
        if k not in sizes:
            print(f"WARNING: tag {k} is in the map but has no size in {tags_path}; skipped")
            continue
        R = np.array(v["R"], np.float64).reshape(3, 3)
        U, _, Vt = np.linalg.svd(R)                       # re-orthonormalize
        R = U @ Vt
        t = np.array(v["t"], np.float64).reshape(3)
        poses[k] = (C_FLIP @ R @ C_FLIP, C_FLIP @ t)      # map frame -> internal frame
    if origin not in poses:
        raise SystemExit(f"origin tag {origin} missing from {map_path}")
    return origin, poses, sizes


def drone_in_frame(R, t):
    """Camera pose given camera<-frame (R, t): position p and yaw psi (deg) in that
    frame. psi=0 faces into the tag (+z); positive = turned clockwise seen from above."""
    p = -R.T @ t
    f = R.T @ np.array([0.0, 0.0, 1.0])
    return p, math.degrees(math.atan2(f[0], f[2]))


def chain_to_origin(R_ck, t_ck, R_0k, t_0k):
    """camera<-tag_k and tag_k pose in origin -> camera<-origin."""
    R_c0 = R_ck @ R_0k.T
    return R_c0, t_ck - R_c0 @ t_0k


def tag_candidates(corners, K, D, size):
    """All plausible camera<-tag poses for one tag, independent of the detector's
    corner order: 4 start corners x 2 windings x 2 IPPE solutions; keep front-face
    views with low error; fix in-plane 90-deg ambiguity (tag upright: y ~ down).
    Returns [(err, R, t)] sorted by error, near-duplicates removed."""
    obj = obj_points(size)
    out = []
    for wind in (corners, corners[::-1]):
        for k in range(4):
            pts = np.roll(wind, k, axis=0).reshape(-1, 1, 2).astype(np.float64)
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, pts, K, D,
                                                        flags=cv2.SOLVEPNP_IPPE)
            errs = np.array(errs).reshape(-1)
            for i in range(n):
                if not np.isfinite(errs[i]) or errs[i] > MAX_SINGLE_ERR_PX:
                    continue
                R, _ = cv2.Rodrigues(rvecs[i])
                t = tvecs[i].reshape(3)
                if R[2, 2] <= 0 or t[2] <= 0:
                    continue
                best = max(range(4), key=lambda j: (R @ rotz90(j))[1, 1])
                out.append((float(errs[i]), R @ rotz90(best), t))
    out.sort(key=lambda c: c[0])
    uniq = []
    for c in out:
        if all(np.linalg.norm(c[2] - u[2]) > 0.02 or
               math.degrees(math.acos(max(-1, min(1, (np.trace(c[1].T @ u[1]) - 1) / 2)))) > 3
               for u in uniq):
            uniq.append(c)
    return uniq


def match_corners(corners, R, t, K, D, size):
    """Reorder detected corners to match obj_points(size) for pose (R, t)."""
    proj, _ = cv2.projectPoints(obj_points(size), cv2.Rodrigues(R)[0], t, K, D)
    proj = proj.reshape(-1, 2)
    idx = [int(np.argmin(np.linalg.norm(corners - p, axis=1))) for p in proj]
    if len(set(idx)) != 4:
        return None
    return corners[idx]


def localize(dets, K, D, tag_map, sizes, prior):
    """dets: [(id, corners)] of trusted mapped tags. Returns (p, psi, info, per_tag) in the
    tag-10 frame, or (None, None, reason, {}). per_tag: {id: (t_cam, area, R_ck)}."""
    per = []
    for tid, c in dets:
        cands = tag_candidates(c, K, D, sizes[tid])
        if not cands:
            continue
        R0k, t0k = tag_map[tid]
        e0 = cands[0][0]
        poses = []
        for err, R, t in [cc for cc in cands if cc[0] < 2.0 * e0 + 0.5]:
            Rc0, tc0 = chain_to_origin(R, t, R0k, t0k)
            p, psi = drone_in_frame(Rc0, tc0)
            poses.append((err, R, t, p, psi, Rc0))
        if prior is not None:
            pp, ppsi = prior
            pick = min(poses, key=lambda q: np.linalg.norm(q[3] - pp) + abs(wrap_deg(q[4] - ppsi)) / 30.0)
        else:
            pick = min(poses, key=lambda q: abs(q[1][1, 2]))
        per.append((tid, c, pick, cv2.contourArea(c.astype(np.float32))))
    if not per:
        return None, None, "no valid tag pose", {}
    per_tag = {tid: (pk[2], area, pk[1]) for tid, c, pk, area in per}
    best = max(per, key=lambda q: (q[0] == TAG_ID, q[3]))   # tag 10 first, then biggest
    err, R, t, p, psi, Rc0 = best[2]
    info = {"tags": sorted(q[0] for q in per), "mode": "single", "rms": err}
    if len(per) >= 2:
        obj_all, img_all = [], []
        for tid, c, pk, _ in per:
            mc = match_corners(c, pk[1], pk[2], K, D, sizes[tid])
            if mc is None:
                continue
            R0k, t0k = tag_map[tid]
            obj_all.append((R0k @ obj_points(sizes[tid]).T).T + t0k)
            img_all.append(mc)
        if len(obj_all) >= 2:
            obj_all, img_all = np.vstack(obj_all), np.vstack(img_all)
            _, tc0 = chain_to_origin(R, t, *tag_map[best[0]])
            ok, rv, tv = cv2.solvePnP(obj_all, img_all, K, D, cv2.Rodrigues(Rc0)[0],
                                      tc0.reshape(3, 1).copy(), useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                pr, _ = cv2.projectPoints(obj_all, rv, tv, K, D)
                rms = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - img_all) ** 2, 1))))
                if rms <= JOINT_MAX_RMS_PX:
                    p, psi = drone_in_frame(cv2.Rodrigues(rv)[0], tv.reshape(3))
                    info.update(mode="joint", rms=rms)
                else:
                    info.update(mode="single (map disagrees)", rms=rms)
    return p, psi, info, per_tag


class MapCheck:
    """Shared across drones: compares tags seen together with tag 10 against the map."""
    def __init__(self, tag_map):
        self.tag_map, self.lock = tag_map, threading.Lock()
        self.good, self.bad, self.off = {}, {}, set()

    def observe(self, per_tag):
        if TAG_ID not in per_tag:
            return
        t10, _, R10 = per_tag[TAG_ID]
        for tid, (tk, _, _) in per_tag.items():
            if tid == TAG_ID or tid in self.off:
                continue
            obs = R10.T @ (tk - t10)                     # tag k center in tag-10 frame
            dev = float(np.linalg.norm(obs - self.tag_map[tid][1]))
            with self.lock:
                if dev > MAPCHECK_M:
                    self.bad[tid] = self.bad.get(tid, 0) + 1
                else:
                    self.good[tid] = self.good.get(tid, 0) + 1
                b, g = self.bad.get(tid, 0), self.good.get(tid, 0)
                if b >= MAPCHECK_MIN_BAD and b > 2 * g:
                    self.off.add(tid)
                    print(f"!!! MAP CHECK: tag {tid} is {dev:.2f} m from its mapped spot "
                          f"(seen with tag 10) -> IGNORED for this flight. Re-map it.", flush=True)

    def trusted(self, tid):
        return tid not in self.off


def cone_push(p, others_cones):
    """Push (tag frame, x/z) out of other drones' view cones. others_cones: [(q, T, half)]
    with q = other drone, T = the tag it watches, half = that tag's half width."""
    push = np.zeros(3)
    a = np.array([p[0], p[2]])
    for q, T, half in others_cones:
        q2, T2 = np.array([q[0], q[2]]), np.array([T[0], T[2]])
        seg = T2 - q2
        L = float(seg @ seg)
        if L < 1e-6:
            continue
        s = float((a - q2) @ seg) / L                    # 0 at that drone, 1 at its tag
        if not 0.05 < s < 0.98:
            continue
        foot = q2 + s * seg
        lat = a - foot
        dist = float(np.linalg.norm(lat))
        width = s * half + DRONE_RADIUS_M
        if dist < width:
            u = lat / dist if dist > 1e-6 else np.array([-seg[1], seg[0]]) / math.sqrt(L)
            push += K_CONE * (width - dist) * np.array([u[0], 0.0, u[1]])
    return push


def in_cone(p, q, T, half):
    """Is drone at q blocking my (p) view of tag at T?"""
    return float(np.linalg.norm(cone_push(q, [(p, T, half)]))) > 0


class Drone:
    def __init__(self, num, iface, ssid, K, D, fly, tag_map, sizes, mapcheck):
        self.num, self.iface, self.ssid, self.K, self.D = num, iface, ssid, K, D
        self.name, self.fly = f"drone{num}", fly
        self.tag_map, self.sizes, self.mapcheck = tag_map, sizes, mapcheck
        self.sock = self._sock(9000 + num)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame, self.frame_t, self.frame_id, self.frames, self.packets = None, 0.0, 0, 0, 0
        self.overlay = None
        self.battery = None
        self.airborne = self.landed = self.landing = False
        self.land_thread = None
        self.st, self.st_t = {}, 0.0
        self.last_yaw, self.yaw_acc = None, 0.0
        self.fix = None              # (p, psi, t, imu_yaw)
        self.fix_info = {}
        self.seen_t = 0.0            # last time a trusted tag was detected well
        self.seen_h = None
        self.ref_id = None           # tag the camera stays on
        self.ref_tcam = None         # that tag's center in camera frame
        self.tags_now = []
        self.reason = "no frames"
        self.glimpse_t = 0.0
        self.jump_since = None
        self.imu_sign, self.imu_ev, self.imu_learned = 1.0, 0.0, False
        self.state, self.sub, self.sub_t = "CLIMB", None, None
        self.turn_start, self.searched = 0.0, 0.0
        self.lock_t = self.face_t = None
        self.face_h = None           # FACE10: height (cm, sensor) of tag 10's center + offset
        self.aligned = False
        self.lost_t = None
        self.was_fix = False
        self.dry_noted = False
        self.last_status = 0.0
        self.last_err = (None, None)
        self.slow_until = 0.0        # cautious speed after a tag loss
        self.flipping = False
        self.flip_done = False
        self.flip_end = 0.0
        self.grace_until = 0.0       # tag loss tolerated until this time (after a flip)
        self.integ = np.zeros(3)     # integral of slot error (tag frame), near the slot
        self.integ_slot = None
        self.integ_t = None
        self.last_rc = (0, 0, 0, 0)

    def _sock(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, self.iface.encode())
        s.bind(("", port))
        return s

    def log(self, msg):
        print(f"[{self.name}] {msg}", flush=True)

    def _check(self, cmd):
        if cmd in NOFLY:
            return
        if self.fly:
            if cmd in ("takeoff", "land", "emergency", FLIP_CMD):
                return
            m = RC_RE.match(cmd)
            if m and all(abs(int(x)) <= 100 for x in m.groups()):
                return
        raise PermissionError(f"blocked command {cmd!r} (fly={self.fly})")

    def send(self, cmd, timeout=10, quiet=False):
        self._check(cmd)
        with self.lock:
            self.sock.setblocking(False)
            try:
                while True:
                    self.sock.recv(1024)
            except (BlockingIOError, OSError):
                pass
            self.sock.settimeout(timeout)
            self.sock.sendto(cmd.encode(), DRONE_ADDR)
            try:
                r = self.sock.recv(1024).decode(errors="ignore").strip().lower()
            except socket.timeout:
                r = "timeout"
        if not quiet:
            self.log(f"{cmd} -> {r}")
        return r

    def send_noreply(self, cmd):
        self._check(cmd)
        self.sock.sendto(cmd.encode(), DRONE_ADDR)

    # ---- status stream ----
    def state_loop(self):
        ss = self._sock(8890)
        ss.settimeout(1.0)
        while not self.stop.is_set():
            try:
                data = ss.recv(2048).decode(errors="ignore")
            except socket.timeout:
                continue
            vals = dict(kv.split(":", 1) for kv in data.strip().split(";") if ":" in kv)
            self.st, self.st_t = vals, time.time()
            try:
                y = float(vals["yaw"])
                if self.last_yaw is not None:
                    self.yaw_acc += abs(wrap_deg(y - self.last_yaw))
                self.last_yaw = y
            except (KeyError, ValueError):
                pass
        ss.close()

    def _stv(self, key):
        if time.time() - self.st_t > 1.0:
            return None
        try:
            return float(self.st[key])
        except (KeyError, ValueError):
            return None

    def imu_yaw(self):
        return self._stv("yaw")

    def height_cm(self):
        tof = self._stv("tof")
        if tof is not None and 10 <= tof <= 400:
            return tof
        h = self._stv("h")
        return h if h is not None and 0 < h <= 400 else None

    def on_ground(self):
        h = self._stv("h")
        return None if h is None else h <= 5

    def land(self):
        if not self.airborne or self.landed:
            return
        try:
            self.send_noreply("rc 0 0 0 0")
            for _ in range(3):
                self.send_noreply("land")
                t0 = time.time()
                while time.time() - t0 < 8:
                    if self.on_ground():
                        self.landed = True
                        self.log("landed")
                        return
                    time.sleep(0.2)
                if self.on_ground() is None and self.send("land", 10) == "ok":
                    self.landed = True
                    return
            self.log("WARNING: land NOT confirmed - catch this drone!")
        except OSError as ex:
            self.log(f"land error: {ex}")

    # ---- pose ----
    def has_fix(self, now):
        """A trusted tag is in view right now."""
        return self.fix is not None and now - self.seen_t < FIX_FRESH_S

    def sees10(self, now):
        return self.has_fix(now) and self.ref_id == TAG_ID

    def ref_bearing(self):
        """(deg to center the reference tag, + = right ; its height vs camera, m, + = below)"""
        x, y, z = self.ref_tcam
        return math.degrees(math.atan2(x, z)), y

    def est(self, now):
        if self.fix is None:
            return None
        p, psi, t, y0 = self.fix
        y = self.imu_yaw()
        if self.imu_learned and y is not None and y0 is not None:
            psi = wrap_deg(psi + self.imu_sign * wrap_deg(y - y0))
        return p, psi, now - t

    def video_loop(self):
        import av
        vs = self._sock(11111)
        vs.settimeout(1.0)
        codec = av.CodecContext.create("h264", "r")
        while not self.stop.is_set():
            try:
                data = vs.recv(65536)
            except socket.timeout:
                continue
            self.packets += 1
            try:
                for pkt in codec.parse(data):
                    for fr in codec.decode(pkt):
                        self.frame, self.frame_t = fr.to_ndarray(format="bgr24"), time.time()
                        self.frame_id += 1
                        self.frames += 1
            except Exception:
                pass
        vs.close()

    def perception_loop(self):
        from pupil_apriltags import Detector
        det = Detector(families="tag36h11", nthreads=1, quad_decimate=2.0, refine_edges=True,
                       decode_sharpening=0.25)
        last = 0
        while not self.stop.is_set():
            if self.frame_id == last or self.frame is None:
                time.sleep(0.005)
                continue
            last, img, t = self.frame_id, self.frame.copy(), self.frame_t
            good, reason = [], "no trusted tag in view"
            for d in det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)):
                if d.tag_id not in self.tag_map:
                    continue
                c = np.array(d.corners, np.float64)
                side = min(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4))
                trusted = self.mapcheck.trusted(d.tag_id)
                ok = trusted and d.decision_margin >= MIN_DECISION_MARGIN and side >= MIN_TAG_SIDE_PX
                self.glimpse_t = t if trusted else self.glimpse_t
                col = (0, 255, 0) if ok else ((0, 0, 255) if trusted else (128, 128, 128))
                cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, col, 3)
                cv2.putText(img, str(d.tag_id), tuple(int(v) for v in c.mean(0)), 0, 0.9, col, 2)
                if ok:
                    good.append((d.tag_id, c))
                elif trusted:
                    reason = f"tag {d.tag_id} weak (margin {d.decision_margin:.0f}, {side:.0f}px)"
            if good:
                prior = (self.fix[0], self.fix[1]) if self.fix is not None and t - self.fix[2] < 2.0 else None
                p, psi, info, per_tag = localize(good, self.K, self.D, self.tag_map, self.sizes, prior)
                if p is not None:
                    self.mapcheck.observe(per_tag)
                    per_tag = {k: v for k, v in per_tag.items() if self.mapcheck.trusted(k)}
                if p is not None and per_tag:
                    # reference tag: tag 10 if visible; else keep current unless much smaller
                    big = max(per_tag, key=lambda k: per_tag[k][1])
                    if TAG_ID in per_tag:
                        ref = TAG_ID
                    elif self.ref_id in per_tag and per_tag[self.ref_id][1] > 0.5 * per_tag[big][1]:
                        ref = self.ref_id
                    else:
                        ref = big
                    if ref != self.ref_id and self.ref_id is not None:
                        self.log(f"reference tag {self.ref_id} -> {ref}")
                    self.ref_id, self.ref_tcam = ref, per_tag[ref][0]
                    self.seen_t, self.seen_h, self.tags_now = t, self.height_cm(), sorted(per_tag)
                    self._accept(p, psi, info, t)
                else:
                    self.reason = info if p is None else "only untrusted tags in view"
            else:
                self.reason = reason
            self.overlay = img

    def _accept(self, p, psi, info, t):
        lim = max(JUMP_MAX_M, 0.25 * float(np.hypot(p[0], p[2])))
        if self.fix is not None and t - self.fix[2] < 1.0 and np.linalg.norm(p - self.fix[0]) > lim:
            self.jump_since = self.jump_since or t
            if t - self.jump_since < JUMP_ACCEPT_AFTER_S:
                self.reason = f"tags {self.tags_now} seen; position jump ignored"
                return
        self.jump_since = None
        y = self.imu_yaw()
        if self.fix is not None and y is not None and self.fix[3] is not None and t - self.fix[2] < 2.0:
            d_tag, d_imu = wrap_deg(psi - self.fix[1]), wrap_deg(y - self.fix[3])
            if abs(d_imu) > 3:
                self.imu_ev += d_tag * d_imu
                if abs(self.imu_ev) > 300:
                    self.imu_sign, self.imu_learned = (1.0 if self.imu_ev > 0 else -1.0), True
        if self.fix is not None and t - self.fix[2] < FIX_FRESH_S:
            p = POSE_ALPHA * p + (1 - POSE_ALPHA) * self.fix[0]
            psi = wrap_deg(self.fix[1] + POSE_ALPHA * wrap_deg(psi - self.fix[1]))
        self.fix, self.fix_info = (p, psi, t, y), info
        self.reason = f"tags {info['tags']} {info['mode']} {info['rms']:.1f}px  ref {self.ref_id}"

    def battery_loop(self):
        while not self.stop.wait(5):
            b = self.send("battery?", 3, quiet=True)
            if b.isdigit():
                self.battery = int(b)

    def close(self):
        self.stop.set()
        try:
            self.send("streamoff", 3, quiet=True)
        except Exception:
            pass
        self.sock.close()



# ======================= SETUP =======================
def load_drones(cfg, only, cam_dir, fly, tag_map, sizes, mapcheck):
    with open(cfg) as f:
        data = json.load(f)
    data = data.get("dongles", []) if isinstance(data, dict) else data
    entries = [e for e in data if e.get("ssid")]
    if only:
        entries = [e for e in entries if int(e["number"]) in only]
        missing = only - {int(e["number"]) for e in entries}
        if missing:
            sys.exit(f"No config/ssid for drone(s) {sorted(missing)} in {cfg}")
    if not entries:
        sys.exit("No drones selected.")
    drones, bad = [], []
    for e in entries:
        n = int(e["number"])
        try:
            K, D = load_camera(str(camera_file(n, cam_dir)))
        except Exception as ex:
            bad.append(f"drone{n}: {ex}")
            continue
        drones.append(Drone(n, e["iface"], e["ssid"], K, D, fly, tag_map, sizes, mapcheck))
    if bad:
        sys.exit("Cannot run:\n  " + "\n  ".join(bad))
    return drones


def draw_top(drones, slots, assign, tag_map, mapcheck, now):
    S, W, H = 120, 680, 560
    img = np.full((H, W, 3), 30, np.uint8)
    ox, oy = W // 2, 60

    def px(p):
        return int(ox + p[0] * S), int(oy - p[2] * S)
    for tid, (R, t) in tag_map.items():
        xa = R[:, 0] * (0.29 if tid == TAG_ID else 0.08)
        col = (255, 255, 255) if mapcheck.trusted(tid) else (80, 80, 80)
        cv2.line(img, px(t - xa), px(t + xa), col, 5 if tid == TAG_ID else 3)
        cv2.putText(img, str(tid) + ("" if mapcheck.trusted(tid) else " OFF"), (px(t)[0] + 6, px(t)[1] - 6),
                    0, 0.45, col, 1)
    for i, sp in enumerate(slots):
        cv2.circle(img, px(sp), 6, (0, 200, 255), 1)
        cv2.putText(img, f"s{i}", (px(sp)[0] + 8, px(sp)[1]), 0, 0.4, (0, 200, 255), 1)
    for d in drones:
        e = d.est(now)
        if e is None or d.landed:
            continue
        p, psi, age = e
        fresh = d.has_fix(now)
        col = (0, 255, 0) if fresh else (0, 200, 255)
        if not fresh:
            cv2.circle(img, px(p), int((D_SAFE_M + LOST_DRIFT_MPS * (now - d.seen_t)) * S), (0, 0, 140), 1)
        cv2.circle(img, px(p), 8, col, -1)
        hx = p + 0.25 * np.array([math.sin(math.radians(psi)), 0, math.cos(math.radians(psi))])
        cv2.line(img, px(p), px(hx), col, 2)
        if d.ref_id in tag_map:                         # line to the tag it is watching
            cv2.line(img, px(p), px(tag_map[d.ref_id][1]), (255, 255, 0), 1)
        cv2.putText(img, f"{d.num} {d.state} ref{d.ref_id}", (px(p)[0] + 10, px(p)[1] - 10), 0, 0.42, col, 1)
        if d.num in assign:
            cv2.line(img, px(p), px(slots[assign[d.num]]), (90, 90, 90), 1)
    cv2.putText(img, "green = sees a tag   yellow = frozen   cyan line = tag it watches",
                (10, H - 12), 0, 0.42, (200, 200, 200), 1)
    return img


def yaw_to(err_deg):
    return 0 if abs(err_deg) < DEADBAND_YAW_DEG else kick(np.clip(KP_YAW * err_deg, -MAX_RC_YAW, MAX_RC_YAW))


def climb_err(d, p, ty):
    """Meters to CLIMB, straight from the image (ty > 0 = tag below camera = too high).
    Watching tag 10: go to its center height + offset. Watching another tag: keep THAT
    tag vertically centered so it never slides out of view (tag-10 height comes later,
    once tag 10 is the reference at the slot)."""
    return (HEIGHT_OFFSET_M if d.ref_id == TAG_ID else 0.0) - ty


def face10_vertical(d, p, ty):
    """FACE10 vertical rc: image-based once tag 10 is the reference, otherwise hold the
    height sensor at the tag-10 height computed from the pose when the slot was reached."""
    if d.ref_id == TAG_ID and d.ref_tcam is not None:
        return vert_to(climb_err(d, p, ty))
    h = d.height_cm()
    if d.face_h is None or h is None:
        return 0
    return vert_to((d.face_h - h) / 100.0)


def vert_to(err_m):
    """err_m > 0 -> must climb."""
    return 0 if abs(err_m) < 0.05 else kick(np.clip(KP_POS * err_m, -MAX_RC_Z, MAX_RC_Z))


def main():
    global HEIGHT_OFFSET_M, ARC_RADIUS_M
    g = globals()
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DONGLES_PATH))
    ap.add_argument("--camera-dir", default=str(CAMERA_DIR))
    ap.add_argument("--map", default=str(MAP_PATH))
    ap.add_argument("--tags", default=str(TAGS_PATH))
    ap.add_argument("--no-map", action="store_true", help="use tag 10 only")
    ap.add_argument("--ignore-tags", default="", help="mapped tags to ignore, e.g. 6,7")
    ap.add_argument("--only", default="", help="e.g. 3 or 1,3,5")
    ap.add_argument("--start-height", type=int, default=None,
                    help=f"climb to this after takeoff, cm (default {START_HEIGHT_CM}, fast 155)")
    ap.add_argument("--slot-offset", type=int, default=0, help="slot height vs tag 10 center, cm")
    ap.add_argument("--radius", type=float, default=ARC_RADIUS_M, help="slot distance from tag 10, m")
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--fast", action="store_true", help="fast profile (see FAST in settings)")
    ap.add_argument("--flip", action="store_true",
                    help="once in formation, each drone does a back flip (one at a time), then holds")
    ap.add_argument("--fly", action="store_true", help="actually fly (default dry-run)")
    args = ap.parse_args()
    ARC_RADIUS_M = max(0.9, args.radius)
    HEIGHT_OFFSET_M = args.slot_offset / 100.0
    if args.fast:
        g.update(FAST)
        print("FAST profile: " + ", ".join(f"{k}={v}" for k, v in FAST.items()))
    fly = args.fly
    start_h = max(60, min(250, args.start_height if args.start_height is not None
                          else globals()["START_HEIGHT_CM"]))
    only = {int(x) for x in args.only.split(",") if x.strip()}
    print(f"MODE: {'FLY' if fly else 'DRY-RUN (no takeoff, no motion)'}")

    if not args.no_map and os.path.exists(args.map) and os.path.exists(args.tags):
        origin, tag_map, sizes = load_tag_files(args.map, args.tags)
        if origin != TAG_ID:
            sys.exit(f"map origin is tag {origin}, this script expects tag {TAG_ID}")
        for k in {int(x) for x in args.ignore_tags.split(",") if x.strip()}:
            if k != TAG_ID:
                tag_map.pop(k, None)
        print(f"Map: tags {sorted(tag_map)}")
    else:
        tag_map, sizes = {TAG_ID: (np.eye(3), np.zeros(3))}, {TAG_ID: TAG_SIZE_M}
        print("No map: tag 10 only")
    mapcheck = MapCheck(tag_map)

    drones = load_drones(args.config, only, args.camera_dir, fly, tag_map, sizes, mapcheck)
    slots, radius = make_slots(len(drones))
    print(f"Drones {[d.name for d in drones]} -> slots {radius:.2f} m from tag 10 at "
          f"{[round(math.degrees(math.atan2(s[0], -s[2]))) for s in slots]} deg")
    if not all(parallel(drones, lambda d: preflight(d, fly)).values()):
        for d in drones:
            d.close()
        sys.exit("Preflight FAILED -> nothing takes off.")

    os.makedirs(os.path.join(here, "logs"), exist_ok=True)
    logf = open(os.path.join(here, time.strftime("logs/formation_%Y%m%d_%H%M%S.csv")), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["t", "drone", "state", "sees", "ref", "tags", "px", "py", "pz", "psi",
                  "pos_err", "a", "b", "c", "d", "height_cm", "battery", "reason"])

    assign, members = {}, set()
    phase, settle_t, hold_t, first_lock_t = "RUN", None, None, None
    try:
        if fly:
            if input("All PASS. Space clear, drones 1 m+ apart? Type FLY: ").strip() != "FLY":
                sys.exit("Aborted, no takeoff.")
            for d in drones:
                d.airborne = True

            def to(d):
                time.sleep(TAKEOFF_STAGGER_S * drones.index(d))
                d.to_state = "pending"
                ok = False
                for _ in range(2):
                    if d.send("takeoff", 20) == "ok":
                        ok = True
                        break
                    time.sleep(1.0)
                    h = re.sub(r"[^0-9]", "", d.send("height?", 5))
                    if h and int(h) >= 3:
                        d.log(f"takeoff reply lost but airborne ({h} dm) -> ok")
                        ok = True
                        break
                if d.to_state != True:
                    d.to_state = ok
            # each drone starts its mission the moment ITS takeoff is done (no waiting)
            for d in drones:
                d.to_state, d.h_ok_since = None, None
                threading.Thread(target=to, args=(d,), daemon=True).start()
        else:
            for d in drones:
                d.to_state = True
        t0 = time.time()
        for d in drones:
            d.sub_t, d.yaw_acc = t0, 0.0

        period = 1.0 / LOOP_HZ
        while True:
            t_loop = now = time.time()
            active = [d for d in drones if not d.landed and not d.landing]
            if not active:
                print("No drones left in the formation.")
                return
            cmds, all_in = {}, True

            def others_of(d):
                out, cones = [], []
                for o in drones:
                    if o is d or o.landed:
                        continue
                    e = o.est(now)
                    if e is None or now - o.seen_t > 15:
                        continue
                    age = now - o.seen_t
                    spd = math.hypot(o.last_rc[0], o.last_rc[1])
                    fast_extra = SPEED_BUBBLE_M * min(1.0, max(0.0, (spd - CAUTIOUS_RC_XY) / 20.0))
                    out.append((e[0], (0.0 if age < FIX_FRESH_S else LOST_DRIFT_MPS * age) + fast_extra))
                    if o.ref_id in tag_map:
                        cones.append((e[0], tag_map[o.ref_id][1], sizes[o.ref_id] / 2))
                return out, cones

            def status(d, msg):
                if now - d.last_status > 2.0:
                    d.last_status = now
                    d.log(msg)

            if any(d.to_state is False for d in drones):
                print("A takeoff failed -> landing all")
                return
            for d in active:
                if d.to_state is not True:
                    hh = d._stv("h")                      # airborne already? (status stream)
                    if d.to_state == "pending" and hh is not None and hh >= 60:
                        d.h_ok_since = d.h_ok_since or now
                        if now - d.h_ok_since > 0.5:
                            d.to_state = True
                    if d.to_state is True:
                        d.sub_t, d.yaw_acc = now, 0.0
                        d.log("airborne -> climbing/searching now")
                    else:
                        cmds[d.num] = None                # still taking off: send nothing
                        all_in = False
                        continue
                if d.flipping:                             # mid-flip: hands off
                    cmds[d.num] = None
                    all_in = False
                    continue
                if d.battery is not None and d.battery < LAND_BATTERY:
                    land_async(d, f"battery {d.battery}%", fly)
                fx = d.has_fix(now)
                if d.was_fix and not fx:
                    d.log(f"lost tags in {d.state}: {d.reason}")
                    if d.state == "APPROACH":
                        d.slow_until = now + SLOW_AFTER_LOSS_S
                d.was_fix = fx
                h = d.height_cm()
                e = d.est(now)

                # ---------- CLIMB ----------
                if d.state == "CLIMB" and fx and args.fast:
                    d.state, d.lock_t = "LOCKED", now
                    first_lock_t = first_lock_t or now
                    d.log(f"LOCALIZED during climb from tags {d.tags_now} (ref {d.ref_id})")
                if d.state == "CLIMB":
                    all_in = False
                    c = 0 if h is None else int(np.clip(KP_HEIGHT * (start_h - h), -MAX_RC_Z, MAX_RC_Z))
                    if h is None or abs(start_h - h) < 10 or now - d.sub_t > CLIMB_TIMEOUT_S:
                        d.state, d.sub, d.sub_t = "SEARCH", "LOOK", now
                    cmds[d.num] = (0, 0, c, 0)
                    continue

                # ---------- SEARCH ----------
                if d.state == "SEARCH":
                    all_in = False
                    if fx:
                        d.state, d.lock_t = "LOCKED", now
                        first_lock_t = first_lock_t or now
                        p, psi, _ = e
                        d.log(f"LOCALIZED from tags {d.tags_now} (ref {d.ref_id}): "
                              f"{math.hypot(p[0], p[2]):.2f} m from tag 10, "
                              f"{math.degrees(math.atan2(p[0], -p[2])):+.0f} deg off its centerline, "
                              f"{-p[1] * 100:+.0f} cm vs its center")
                    else:
                        c = 0 if h is None else int(np.clip(KP_HEIGHT * (start_h - h), -MAX_RC_Z, MAX_RC_Z))
                        yaw = 0
                        glimpse = now - d.glimpse_t < GLIMPSE_HOLD_S
                        if d.sub == "LOOK":
                            look = LOOK_S if d.searched == 0 else SEARCH_LOOK_S
                            if now - d.sub_t > look and (not glimpse or now - d.sub_t > GLIMPSE_HOLD_S + look):
                                d.sub, d.sub_t, d.turn_start = "TURN", now, d.yaw_acc
                        else:
                            turned = (d.yaw_acc - d.turn_start) if d.st else (now - d.sub_t) * 40.0
                            if d.glimpse_t > d.sub_t + 0.3 or turned >= SEARCH_STEP_DEG:
                                d.searched += max(0.0, turned)
                                d.sub, d.sub_t = "LOOK", now
                            else:
                                yaw = SEARCH_YAW_RC
                        cmds[d.num] = (0, 0, c, yaw)
                        if d.searched > SEARCH_MAX_DEG:
                            land_async(d, f"full turn ({d.searched:.0f} deg) without a trusted tag", fly)
                        continue

                # ---------- no tag in view (after localizing) ----------
                if not fx and now < d.grace_until:          # just flipped: hover, reacquire
                    all_in = False
                    cmds[d.num] = (0, 0, 0, 0)
                    continue
                if not fx:
                    all_in = False
                    d.lost_t = d.lost_t or now
                    lost = now - max(d.seen_t, d.flip_end + FLIP_GRACE_S if d.flip_end else 0)
                    p, psi, _ = e
                    yaw, c = 0, 0
                    if d.state in ("LOCKED", "FACE10"):            # in-place turning allowed
                        if d.state == "FACE10":
                            yaw = yaw_to(wrap_deg(face_tag_yaw(p) - psi)) if d.imu_learned else \
                                (RECOVER_YAW_RC if wrap_deg(face_tag_yaw(p) - psi) > 0 else -RECOVER_YAW_RC)
                            c = face10_vertical(d, p, 0.0) if d.ref_id != TAG_ID else 0
                        if lost > INPLACE_NOFIX_S:
                            land_async(d, f"no tag for {lost:.1f}s while turning in place", fly)
                        cmds[d.num] = (0, 0, c, yaw)
                        continue
                    T_ref = tag_map[d.ref_id][1] if d.ref_id in tag_map else np.zeros(3)
                    blocker = next((o for o in active if o is not d and o.est(now) is not None
                                    and in_cone(p, o.est(now)[0], T_ref, sizes.get(d.ref_id, 0.1) / 2)), None)
                    if blocker is not None:                   # known blockage: just wait
                        d.reason = f"BLOCKED by {blocker.name} -> waiting"
                        if lost > BLOCKED_MAX_S:
                            land_async(d, f"blocked by {blocker.name} for {lost:.0f}s", fly)
                    else:                                     # recover in place
                        if lost > 0.4:
                            if h is not None and d.seen_h is not None:
                                c = int(np.clip(KP_HEIGHT * (d.seen_h - h), -MAX_RC_Z, MAX_RC_Z))
                            brg = d.ref_bearing()[0] if d.ref_tcam is not None else 0.0
                            y = d.imu_yaw()
                            if d.imu_learned and y is not None and d.fix[3] is not None:
                                rem = brg - d.imu_sign * wrap_deg(y - d.fix[3])
                                yaw = 0 if abs(rem) < 3 else int(np.clip(KP_YAW * rem, -RECOVER_YAW_RC, RECOVER_YAW_RC))
                            elif abs(brg) > 10:
                                yaw = RECOVER_YAW_RC if brg > 0 else -RECOVER_YAW_RC
                        if lost > RECOVER_S:
                            land_async(d, f"tags lost {lost:.1f}s, recovery failed", fly)
                    cmds[d.num] = (0, 0, c, yaw)              # never sideways without a tag
                    continue
                if d.lost_t is not None:
                    d.log(f"tags back after {now - d.lost_t:.1f}s (ref {d.ref_id})")
                    d.lost_t = None
                d.dry_noted = False
                p, psi, _ = e
                eb, ty = d.ref_bearing()

                # ---------- LOCKED: keep ref tag centered, go to tag-10 height, in place ----------
                if d.state == "LOCKED":
                    all_in = False
                    up = climb_err(d, p, ty)
                    c, yaw = vert_to(up), yaw_to(eb)
                    herr = -up                                         # + = drone too high
                    d.aligned = abs(herr) < ALIGN_H_M or now - d.lock_t > ALIGN_TIMEOUT_S
                    cmds[d.num] = (0, 0, c, yaw)
                    status(d, f"LOCKED (ref {d.ref_id}): height err {herr * 100:+.0f} cm, "
                              f"rc c{c:+d} d{yaw:+d}" + ("  -> ready" if d.aligned else ""))
                    continue

                # ---------- APPROACH / FACE10 / HOLDING need a slot ----------
                if d.num not in assign:
                    cmds[d.num] = (0, 0, 0, 0)
                    all_in = False
                    continue
                target = slots[assign[d.num]]
                others, cones = others_of(d)
                cap = CAUTIOUS_RC_XY if now < d.slow_until else MAX_RC_XY
                e_h = np.array([target[0] - p[0], 0.0, target[2] - p[2]])
                dist_h = float(np.linalg.norm(e_h))
                if d.integ_slot != assign[d.num] or dist_h > I_ZONE_M:
                    d.integ, d.integ_t = np.zeros(3), None            # reset
                    d.integ_slot = assign[d.num]
                elif dist_h > DEADBAND_POS_M:
                    dt_i = 0.0 if d.integ_t is None else min(0.2, now - d.integ_t)
                    d.integ = d.integ + e_h * dt_i
                    lim = I_MAX_RC / KI_POS
                    n_i = float(np.linalg.norm(d.integ))
                    if n_i > lim:
                        d.integ *= lim / n_i
                d.integ_t = now
                rc, _, _ = control(p, psi, target, others, cap, d.integ)
                pe = float(math.hypot(target[0] - p[0], target[2] - p[2]))   # horizontal only
                push = cone_push(p, cones)                         # stay out of others' view
                if np.linalg.norm(push) > 0:
                    s_, c_ = math.sin(math.radians(psi)), math.cos(math.radians(psi))
                    a = rc[0] + push[0] * c_ - push[2] * s_
                    b = rc[1] + push[0] * s_ + push[2] * c_
                    n = math.hypot(a, b)
                    if n > MAX_RC_XY:
                        a, b = a * MAX_RC_XY / n, b * MAX_RC_XY / n
                    rc = (int(round(a)), int(round(b)), rc[2], rc[3])
                # height: from the image when watching tag 10, else from the pose
                cz = vert_to(climb_err(d, p, ty))
                d.last_err = (pe, None)

                if d.state == "APPROACH":
                    all_in = False
                    rc = (rc[0], rc[1], cz, yaw_to(eb))            # camera stays on the ref tag
                    if pe < TOL_POS_M:
                        d.state, d.face_t = "FACE10", now
                        h0 = d.height_cm()             # target: tag-10 center height (+offset)
                        d.face_h = None if h0 is None else h0 + 100.0 * (HEIGHT_OFFSET_M + float(p[1]))
                        d.log(f"at slot s{assign[d.num]} -> turning to tag 10")
                    status(d, f"APPROACH s{assign[d.num]} (ref {d.ref_id}): {pe:.2f} m to go, "
                              f"rc a{rc[0]:+d} b{rc[1]:+d} c{rc[2]:+d} d{rc[3]:+d}")
                elif d.state == "FACE10":
                    all_in = False
                    ey = eb if d.ref_id == TAG_ID else wrap_deg(face_tag_yaw(p) - psi)
                    rc = (rc[0], rc[1], face10_vertical(d, p, ty), yaw_to(ey))
                    if d.ref_id == TAG_ID and abs(eb) < TOL_YAW_DEG:
                        d.state = "HOLDING"
                        d.log(f"locked on tag 10 in slot s{assign[d.num]}")
                    elif now - d.face_t > 20:
                        land_async(d, "could not find tag 10 from its slot", fly)
                else:                                              # HOLDING
                    rc = (rc[0], rc[1], cz, yaw_to(eb))
                    if pe > TOL_POS_M or abs(eb) > TOL_YAW_DEG or d.ref_id != TAG_ID:
                        all_in = False
                cmds[d.num] = rc

            # ---- start approaching: all localized & ready, or waited long enough ----
            ready = all(d.state not in ("CLIMB", "SEARCH") for d in active) or \
                (first_lock_t is not None and now - first_lock_t > WAIT_ALL_S)
            if ready:
                for d in active:
                    if d.state == "LOCKED" and d.aligned:
                        d.state = "APPROACH"
            new_members = {d.num for d in active if d.state in ("APPROACH", "FACE10", "HOLDING")}
            if new_members != members:
                pos = {d.num: d.est(now)[0] for d in active if d.num in new_members}
                old = dict(assign)
                assign.clear()
                assign.update(match_slots(pos, slots) if pos else {})
                for d in active:                 # a drone given a new slot must re-approach it
                    if d.num in old and assign.get(d.num) != old[d.num] and d.state in ("FACE10", "HOLDING"):
                        d.state = "APPROACH"
                members = new_members
                print(f">>> slots: {assign}")

            if phase == "RUN":
                ok = all_in and members and all(d.state == "HOLDING" for d in active)
                settle_t = (settle_t or now) if ok else None
                if settle_t and now - settle_t > SETTLE_S:
                    if args.flip:
                        phase = "FLIP"
                        flip_queue = sorted(active, key=lambda x: assign.get(x.num, 0))  # left -> right
                        flip_cur = None
                        print(">>> formation reached -> flip wave")
                    else:
                        phase, hold_t = "HOLD", now
                        print(f">>> formation reached, holding {args.hold:.0f}s")
            elif phase == "FLIP":
                if flip_cur is not None and not flip_cur.flipping:     # wait until it's back
                    back = flip_cur.has_fix(now) and flip_cur.ref_id == TAG_ID and \
                        flip_cur.last_err[0] is not None and flip_cur.last_err[0] < TOL_POS_M
                    if back or now - flip_cur.flip_end > FLIP_RETURN_TIMEOUT_S or flip_cur.landing:
                        flip_cur = None
                if flip_cur is None:
                    flip_queue = [x for x in flip_queue if not x.landed and not x.landing]
                    if not flip_queue:
                        phase, hold_t = "HOLD", now
                        print(f">>> flips done, holding {args.hold:.0f}s")
                    else:
                        flip_cur = flip_queue.pop(0)
                        if not fly:
                            flip_cur.log("would FLIP (dry-run)")
                            flip_cur.flip_end = now
                        elif flip_cur.battery is not None and flip_cur.battery < FLIP_MIN_BATTERY:
                            flip_cur.log(f"battery {flip_cur.battery}% < {FLIP_MIN_BATTERY}% -> skipping flip")
                            flip_cur.flip_end = now - FLIP_RETURN_TIMEOUT_S
                        else:
                            def do_flip(x=flip_cur):
                                x.flipping = True
                                x.send_noreply("rc 0 0 0 0")
                                time.sleep(0.2)
                                r = x.send(FLIP_CMD, 10)
                                x.flip_end = time.time()
                                x.grace_until = x.flip_end + FLIP_GRACE_S
                                x.flipping = False
                                x.flip_done = r == "ok"
                            threading.Thread(target=do_flip, daemon=True).start()
            elif now - hold_t > args.hold:
                print(">>> hold done -> landing all")
                return

            # ---- send + show ----
            for d in active:
                rc = cmds.get(d.num, (0, 0, 0, 0))
                if rc is None:                            # taking off: don't interfere
                    rc = (0, 0, 0, 0)
                elif fly:
                    d.send_noreply("rc %d %d %d %d" % rc)
                d.last_rc = rc
                e = d.est(now)
                h = d.height_cm()
                p = e[0] if e else (np.nan,) * 3
                pe = d.last_err[0] if d.state in ("APPROACH", "FACE10", "HOLDING") else None
                log.writerow([f"{now:.3f}", d.num, d.state, d.has_fix(now), d.ref_id, d.tags_now,
                              *[f"{v:.3f}" for v in p], f"{e[1]:.1f}" if e else "",
                              "" if pe is None else f"{pe:.3f}", *rc, h, d.battery, d.reason])
                img = (d.overlay if d.overlay is not None else np.zeros((RES[1], RES[0], 3), np.uint8)).copy()
                lines = [f"{d.name} {d.state}" + (f"/{d.sub}" if d.state == "SEARCH" else "")
                         + (" HOLD" if phase == "HOLD" else "") + ("  [SEE]" if d.has_fix(now) else "  [LOST]")
                         + (f"  ref {d.ref_id}" if d.ref_id is not None else ""),
                         d.reason[:58],
                         f"rc a{rc[0]:+d} b{rc[1]:+d} c{rc[2]:+d} d{rc[3]:+d}" + ("" if fly else " (DRY)"),
                         f"bat {d.battery}%  height " + (f"{h:.0f}cm" if h is not None else "?")]
                if e:
                    lines.append(f"tag10: {math.hypot(p[0], p[2]):.2f} m, "
                                 f"{math.degrees(math.atan2(p[0], -p[2])):+.0f} deg, {-p[1] * 100:+.0f} cm")
                if pe is not None:
                    lines.append(f"slot s{assign.get(d.num)}: {pe:.2f} m to go")
                for i, s_ in enumerate(lines):
                    cv2.putText(img, s_, (10, 30 + 30 * i), 0, 0.7, (0, 0, 0), 4)
                    cv2.putText(img, s_, (10, 30 + 30 * i), 0, 0.7, (0, 255, 255), 2)
                cv2.imshow(d.name, cv2.resize(img, (480, 360)))
            cv2.imshow("top view", draw_top(drones, slots, assign, tag_map, mapcheck, now))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print("q -> landing all")
                return
            if k == ord(" ") and fly:
                print("!!! EMERGENCY: motors off all")
                for d in drones:
                    d.send_noreply("emergency")
                    d.landed = True
                return
            time.sleep(max(0.0, period - (time.time() - t_loop)))
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing all")
    finally:
        if fly:
            for d in drones:
                if d.land_thread is not None:
                    d.land_thread.join(20)
            parallel([d for d in drones if d.airborne and not d.landed], lambda d: d.land())
        for d in drones:
            d.close()
        logf.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
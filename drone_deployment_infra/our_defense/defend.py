#!/usr/bin/env python3
"""Defend the table: read the reported attacker positions, assign defenders, ram the attackers.

COMPETITION: just run `python defend.py` (any Python, any directory). Everything comes
from defend_config.yaml next to this file: our defenders, the venue map and tag sizes, the
table position, flight mode (fly: true) and limits. Attackers default to "auto": every drone
in the telemetry feed that is not one of our defenders and not labelled team "defense".
It relaunches itself with the repo's .venv Python if the current one lacks the packages,
and Ctrl-C or SIGTERM (a stop button / kill) lands every defender.

Any command-line flag overrides the config, so roles can still be set per run:

  --defenders TELLO-2,TELLO-3,TELLO-5   --attackers TELLO-8,TELLO-9   (or --attackers auto)

Three ways to run (same brain, same controller):
  --sim       no hardware: kinematic defenders fly in software. Pair with hil_gui.py
              (mouse-driven attackers) to watch the strategy react.
  --dry_run   real drones: connects, streams video, localizes from tags, PRINTS the rc it
              would send, no takeoff. Carry a drone by hand. (Default when the config says
              fly: false.)
  --fly       real flight (the config's fly: true does the same).
              KEEP THE AREA CLEAR AND BE READY TO CATCH THEM.

Attackers (feeds.py, --feed auto): swarm_policies.py's telemetry bus (formation/drone<N>,
already in map coordinates) and/or drone_gui telemetry on UDP --feed_port (per-tag poses,
converted with map.yaml); matched by drone name. Defenders (defender_link.py): each takes
off the moment it connects, climbs --climb_cm, advances --advance_cm toward tag 10, then
localizes from the mapped AprilTags it sees (upright_positional.py: print-orientation
proof, drops tags that disagree with tag 10), holds its heading toward --face and
translates sideways/forward/up without turning, so its camera keeps the tags in view.
Strategy: defense_core.py (assign, ram at the attacker's current position, goalkeeper,
--formation home|v|triangle|line when no attacker is live).

Safety: rc capped by --speed_cap and MAX_RC; targets clamped to a geofence from the map and
a height band around tag 10; no tag -> hover at tag height with a widening yaw search, land
after --first_fix_s / --lost_timeout_s; battery < LAND_BATTERY -> that drone lands; link
drop -> lands on reconnect; Ctrl-C / SIGTERM / --seconds -> all land; one instance only.

  python defend.py --sim --defenders D1,D2,D3 --attackers TELLO-8,TELLO-9      # + hil_gui.py
  python defend.py --defenders TELLO-2 --attackers TELLO-8,TELLO-9              # dry run
  python defend.py --defenders TELLO-2,TELLO-3,TELLO-5 --attackers TELLO-8,TELLO-9 --fly
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path


def _ensure_venv() -> None:
    """Re-exec with the repo's .venv Python if this interpreter can't import what we need."""
    try:
        import av  # noqa: F401
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import pupil_apriltags  # noqa: F401
        import tello_calibration  # noqa: F401
        import yaml  # noqa: F401
    except ImportError as exc:
        venv = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
        if venv.exists() and Path(sys.executable).resolve() != venv.resolve() \
                and not os.environ.get("DEFEND_REEXEC"):
            print(f"defend.py: {exc}; relaunching with {venv}", flush=True)
            os.environ["DEFEND_REEXEC"] = "1"
            os.execv(str(venv), [str(venv), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit(f"defend.py: {exc}. Install the repo venv (see HANDOFF / FLYING.md).")


if __name__ == "__main__":
    _ensure_venv()

import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
DD_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DD_ROOT / "core"))

from defense_core import BrainCfg, DefenseBrain, Plan, default_fence  # noqa: E402
from feeds import CombinedFeed, SwarmBusFeed, TelemetryFeed  # noqa: E402

RC_PER_MPS = 125.0          # rc 100 ~ 0.8 m/s (drone_deployment_tests/tello_policy.predict_tag)
MAX_RC = 45
MAX_YAW = 25
K_YAW = 1.5                 # rc per degree of heading error
K_POS = 1.2                 # 1/s: desired speed per metre of position error (guard / hold)
HOLD_DEADBAND_M = 0.12
FIX_MAX_AGE_S = 0.4
LOST_HOVER_S = 1.5          # lost tags this long -> start a yaw search
LAND_BATTERY = 15
TAKEOFF_BATTERY = 30
CAPTURE_M = 0.35            # a defender this close to a live attacker counts as contact
TAG10_HEIGHT_M = 1.22
# Tags the venue moved after map.yaml was made (swarm_formation MAP CHECK, 24 Oct: 7 -2.51 m,
# 12 -1.99 m, 11 -1.87 m, 23 -1.55 m, 2 -1.50 m). Never localize from them; others that
# disagree with tag 10 are dropped at runtime too (upright_positional.MAP_CHECK_M).
DEFAULT_IGNORE_TAGS = "2,7,11,12,23"
SEARCH_YAW, SEARCH_LEG_S = 25, 2.0


def heading_error_deg(R: np.ndarray, face: np.ndarray) -> float:
    """Degrees to yaw right (SDK +) so the camera looks along `face` (map frame)."""
    d = R.T @ face
    return math.degrees(math.atan2(d[0], d[2]))


def facing_R(face: np.ndarray) -> np.ndarray:
    """Camera -> map rotation for a level camera looking along `face`."""
    z = face / np.linalg.norm(face)
    down = np.array([0.0, -1.0, 0.0])
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, down, z])


def velocity_cmd(plan: Plan, pos: np.ndarray, speed_cap: float) -> np.ndarray:
    """Desired map-frame velocity for this plan."""
    err = plan.target - pos
    dist = float(np.linalg.norm(err))
    if plan.mode == "ram":                    # full speed through the meeting point
        return err / dist * speed_cap if dist > 1e-6 else np.zeros(3)
    if dist < HOLD_DEADBAND_M:
        return np.zeros(3)
    v = K_POS * err
    n = float(np.linalg.norm(v))
    return v if n <= speed_cap else v / n * speed_cap


def rc_from(v_map: np.ndarray, R: np.ndarray, face: np.ndarray, rc_cap: int) -> tuple[int, int, int, int]:
    """Map-frame velocity -> Tello rc (right, forward, up, yaw) for a drone with camera->map R."""
    v_cam = R.T @ v_map                       # x right, y down, z forward
    clamp = lambda v, lim: int(max(-lim, min(lim, round(v))))  # noqa: E731
    yaw = clamp(K_YAW * heading_error_deg(R, face), MAX_YAW)
    return (clamp(v_cam[0] * RC_PER_MPS, rc_cap), clamp(v_cam[2] * RC_PER_MPS, rc_cap),
            clamp(-v_cam[1] * RC_PER_MPS, rc_cap), yaw)


# ---- defenders ---------------------------------------------------------------------
class SimDefender:
    """Kinematic stand-in: rc -> velocity with a 0.3 s lag, perfect localization."""

    def __init__(self, name: str, pos: np.ndarray, face: np.ndarray) -> None:
        self.name, self.pos, self.face = name, pos.astype(float), face
        self.R = facing_R(face)
        self.vel = np.zeros(3)
        self.rc = (0, 0, 0, 0)
        self.note = "sim"

    def position(self):
        return self.pos.copy(), 0.0

    def command(self, rc, dt: float) -> None:
        self.rc = rc
        v_cam = np.array([rc[0], -rc[2], rc[1]]) / RC_PER_MPS
        v_cmd = self.R @ v_cam
        self.vel += (v_cmd - self.vel) * min(1.0, dt / 0.3)
        self.pos += self.vel * dt


class RealDefender:
    """One real Tello: localizes from tags each new frame, sends (or prints) rc."""

    def __init__(self, dl, positioner, face, args, stop) -> None:
        self.dl, self.positioner, self.face = dl, positioner, face
        self.name, self.args, self.stop = dl.name, args, stop
        self.loc = None
        self.loc_t = 0.0
        self.last_frames = -1
        self.R = facing_R(face)
        self.rc = (0, 0, 0, 0)
        self.airborne = False
        self.abort = False
        self.landed = False
        self.note = "dry" if not args.fly else "fly"
        self.last_print = 0.0
        self.lost_since = None
        self.upside_down = False    # the tag says our camera is upside down: tag mounted wrong
        self.last_warn = 0.0

    def update_fix(self) -> None:
        frame = self.dl.frames.get(self.name)
        if frame is None:
            return                      # video is (re)started by connect.py's thread, not here
            return
        stats = self.dl.stats.get(self.name, {})
        if stats.get("frames", 0) == self.last_frames:
            return
        self.last_frames = stats.get("frames", 0)
        loc = self.positioner.locate(frame)
        if loc is not None:
            self.loc, self.loc_t = loc, stats.get("view_t", time.time())
            self.R = np.asarray(loc["R"], float)
            # A level camera's "down" axis points down the map's y. If the tag says it points
            # UP, the tag is mounted upside down (or rotated): left/right and up/down would be
            # mirrored and the drone flies the wrong way. Refuse to steer on it.
            self.upside_down = self.R[1, 1] > 0.0
            if self.upside_down and time.time() - self.last_warn > 2.0:
                print(f"{self.name} WARNING: tag(s) {loc['tags']} give a mirrored pose "
                      f"(stale map / bad print?). Ignoring that fix, holding.", flush=True)
                self.last_warn = time.time()

    def position(self):
        if self.loc is None:
            return None, None
        age = time.time() - self.loc_t
        return (np.asarray(self.loc["xyz"], float) if age <= FIX_MAX_AGE_S else None), age

    def send(self, rc) -> None:
        self.rc = tuple(int(v) for v in rc)
        if not self.args.fly:
            now = time.time()
            if now - self.last_print > 0.25:
                print(f"{self.name} DRY rc {self.rc}  {self.note}", flush=True)
                self.last_print = now
            return
        from tello_dual_video import send_rc
        try:
            send_rc(self.dl.cmd, self.rc)
        except OSError:
            pass

    def land(self, why: str) -> None:
        if self.landed:
            return
        self.landed = True
        print(f"{self.name} LAND ({why})", flush=True)
        if self.args.fly and self.airborne:
            from test_link import land
            try:
                self.send((0, 0, 0, 0))
            except OSError:
                pass
            lock = getattr(self.dl, "sdk_lock", None)
            if lock is not None:
                lock.acquire(timeout=3.0)        # let a streamon in progress finish (<= 2 s)
            try:
                print(f"{self.name} land -> {land(self.dl.cmd)}", flush=True)
            finally:
                if lock is not None and lock.locked():
                    lock.release()


# ---- main --------------------------------------------------------------------------
def parse_xyz(text: str, n: int) -> np.ndarray:
    vals = [float(v) for v in text.split(",")]
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f"want {n} comma-separated numbers, got {text!r}")
    return np.array(vals)


CONFIG_KEYS_PATHS = ("map", "tags")       # config values resolved relative to the config file


def load_config(argv: list[str]) -> tuple[Path, dict]:
    """--config PATH (default defend_config.yaml next to this file) -> its settings."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(HERE / "defend_config.yaml"))
    path = Path(pre.parse_known_args(argv)[0].config).expanduser().resolve()
    if not path.exists():
        return path, {}
    cfg = yaml.safe_load(path.read_text()) or {}
    for k in CONFIG_KEYS_PATHS:
        if cfg.get(k):
            p = Path(str(cfg[k])).expanduser()
            cfg[k] = str(p if p.is_absolute() else (path.parent / p).resolve())
    for k, v in list(cfg.items()):
        if isinstance(v, list):                 # yaml lists -> "a,b,c" like the CLI
            cfg[k] = ",".join(str(x) for x in v)
    return path, cfg


SINGLE_INSTANCE_PORT = 47391            # a second defend.py would fight the first over the same drones


def single_instance() -> socket.socket:
    """Hold a fixed localhost port for the whole run; a second copy can't bind it and stops."""
    lock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        lock.bind(("127.0.0.1", SINGLE_INSTANCE_PORT))
    except OSError:
        raise SystemExit("another defend.py is already running on this machine -- stop it first "
                         "(Ctrl-C in its terminal, or: pkill -f defend.py)")
    return lock


def main() -> int:
    _lock = single_instance()                 # noqa: F841  (kept open until we exit)
    cfg_path, file_cfg = load_config(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(cfg_path), help="settings file (default defend_config.yaml)")
    ap.add_argument("--defenders", default=None, help="comma list, e.g. TELLO-2,TELLO-3,TELLO-5")
    ap.add_argument("--attackers", default="auto",
                    help="comma list, e.g. TELLO-8,TELLO-9, or auto = every non-defender in the feed")
    ap.add_argument("--sim", action="store_true", help="software defenders, no hardware")
    ap.add_argument("--fly", action="store_true", help="real flight")
    ap.add_argument("--dry_run", action="store_true", help="no takeoff, even if the config says fly: true")
    ap.add_argument("--swarm_external", action="store_true",
                    help="launched by swarm_policies.py (via swarm_run.py): report PASS per drone, print "
                         "its 'Type FLY:' prompt and take off when it answers FLY")
    ap.add_argument("--feed", choices=("auto", "gui", "swarm"), default="auto",
                    help="where attacker positions come from: gui = drone_gui telemetry on --feed_port, "
                         "swarm = swarm_policies.py's telemetry bus (formation/drone<N>, map frame), "
                         "auto = both, freshest report wins")
    ap.add_argument("--swarm_hub", default="auto",
                    help="swarm_policies.py telemetry hub host:port, or auto = find the running process")
    ap.add_argument("--feed_port", type=int, default=9100, help="drone_gui telemetry UDP port")
    ap.add_argument("--feed_frame", choices=("readme", "gui"), default="readme",
                    help="tag-frame convention of the feed's pose.xyz (see feeds.py; check with feed_check.py)")
    ap.add_argument("--feed_height", choices=("h", "tag"), default="h",
                    help="attacker height from its Tello height sensor h (robust) or the tag pose y (can flip)")
    ap.add_argument("--status", default="127.0.0.1:9101", help="send our status here (hil_gui), '' = off")
    ap.add_argument("--map", default=None, help="map.yaml (default config/map.yaml)")
    ap.add_argument("--tags", default=None, help="tags.yaml (default config/tags.yaml)")
    ap.add_argument("--tag10_height_m", default="auto",
                    help="tag 10 centre above the floor in m, or auto = from the floor tags in the map "
                         "(tags lying face-up), falling back to 1.22")
    ap.add_argument("--ignore_tags", default=DEFAULT_IGNORE_TAGS,
                    help="comma list of tag ids never to localize from (moved since mapping); tags that "
                         "disagree with tag 10 by > 0.5 m are also dropped automatically")
    ap.add_argument("--fence_margin", type=float, default=0.4,
                    help="the flight box is the area spanned by all tags in map.yaml, pulled in by this "
                         "much (m) on each side; 0 = right at the outermost tags, negative = beyond them")
    ap.add_argument("--room", default=None,
                    help="OPTIONAL manual override of the map-derived box: x_min,x_max,z_max in metres "
                         "(write it with =: --room=-1.5,1.5,3.5). Normally leave unset.")
    ap.add_argument("--site", default="auto",
                    help="the table we protect: auto = just behind our defenders' starting line (they face "
                         "tag 10, the table is at their backs), or x,z in the map frame (m)")
    ap.add_argument("--site_behind_m", type=float, default=0.3,
                    help="site auto: how far behind the rearmost defender's starting spot the table is")
    ap.add_argument("--face", choices=("base", "front"), default="base",
                    help="heading the defenders hold (base = look at tag 10)")
    ap.add_argument("--speed_cap", type=float, default=0.35, help="m/s, at most 0.5")
    ap.add_argument("--max_lead_s", type=float, default=0.0,
                    help="how far ahead of the attacker to aim, s: 0 = its current position (default), "
                         "e.g. 1.0 = up to 1 s of predicted motion (intercept point)")
    ap.add_argument("--latency_s", type=float, default=0.0,
                    help="extra lag compensation added to the aim, s (0 = none)")
    ap.add_argument("--guard_radius", type=float, default=1.2)
    ap.add_argument("--hold_height_m", type=float, default=1.2, help="idle height above the floor")
    ap.add_argument("--v_tag", type=int, default=23, help="v: the V sits in front of this mapped tag")
    ap.add_argument("--v_front_m", type=float, default=0.8,
                    help="v: the V's tip is this far in front of --v_tag (toward tag 10)")
    ap.add_argument("--v_width_m", type=float, default=1.2, help="v: distance between the two outer defenders")
    ap.add_argument("--v_depth_m", type=float, default=0.6, help="v: how far the arms trail behind the tip")
    ap.add_argument("--formation", choices=("home", "v", "triangle", "line"), default="home",
                    help="where defenders hold when no attacker is live: home = each where it first hovered "
                         "after takeoff, v = a V in front of --v_tag (tip toward tag 10), triangle = line in "
                         "front of the table with the middle one higher, line = that line at one height")
    ap.add_argument("--hold_spread_m", type=float, default=0.8, help="sideways gap between held defenders")
    ap.add_argument("--tri_rise_m", type=float, default=0.4, help="triangle: middle defender this much higher")
    ap.add_argument("--band_m", type=float, default=0.5,
                    help="defenders stay within +-this of tag 10's height so the (fixed, level) "
                         "camera keeps the tag in view; 0 = no limit")
    ap.add_argument("--first_fix_s", type=float, default=30.0,
                    help="after takeoff, a defender gets this long to see its first tag (video start-up)")
    ap.add_argument("--lost_timeout_s", type=float, default=30.0,
                    help="tag lost this long (hovering + searching meanwhile) -> that defender lands")
    ap.add_argument("--seconds", type=float, default=120.0, help="land everything after this")
    ap.add_argument("--advance_cm", type=int, default=50,
                    help="after takeoff + climb, fly this far forward toward tag 10 (SDK `forward`, 20-500; "
                         "0 = off): closer to the wall = bigger tags = better position fixes")
    ap.add_argument("--log_every_s", type=float, default=1.0,
                    help="print every defender's and attacker's state this often (0 = only events)")
    ap.add_argument("--climb_cm", type=int, default=30,
                    help="climb this much right after takeoff (SDK `up`, 20-500; 0 = off) so the tag "
                         "comes into view")
    known = {a.dest for a in ap._actions}
    unknown = sorted(set(file_cfg) - known)
    if unknown:
        ap.error(f"unknown setting(s) in {cfg_path.name}: {unknown}")
    ap.set_defaults(**file_cfg)                   # config file < command line
    args = ap.parse_args()
    if args.dry_run or args.sim:
        args.fly = False
    if not args.defenders:
        ap.error(f"no defenders: set `defenders:` in {cfg_path} or pass --defenders")
    if not 0 < args.speed_cap <= 0.5:
        ap.error("--speed_cap must be in (0, 0.5]")
    if args.advance_cm and not 20 <= args.advance_cm <= 500:
        ap.error("--advance_cm must be 0 or 20..500")
    if args.climb_cm and not 20 <= args.climb_cm <= 500:
        ap.error("--climb_cm must be 0 or 20..500")
    defenders = [s.strip() for s in str(args.defenders).split(",") if s.strip()]
    auto = str(args.attackers).strip().lower() == "auto"
    attackers = [] if auto else [s.strip() for s in str(args.attackers).split(",") if s.strip()]
    if set(defenders) & set(attackers):
        ap.error(f"a drone can't be on both teams: {sorted(set(defenders) & set(attackers))}")

    from tello_calibration.tags.mapping import TagMap
    from tello_calibration.utils.config import CONFIG_DIR
    map_path = Path(args.map) if args.map else CONFIG_DIR / "map.yaml"
    if not map_path.exists():
        raise SystemExit(f"map {map_path} not found: build it with swarm_ops/map_tags.py --out {map_path}")
    tag_map = TagMap.load(map_path)
    ignore_tags = {int(t) for t in str(args.ignore_tags or "").replace(" ", "").split(",") if t}
    tag_poses = {int(k): np.asarray(v) for k, v in tag_map.poses.items()}
    if str(args.tag10_height_m).lower() == "auto":
        # Tags lying face-up on the floor have their normal (z axis) along map +y; tag 10's
        # height is how far below its centre they lie.
        floor = [-float(M[1, 3]) for M in tag_poses.values() if M[1, 2] > 0.9]
        args.tag10_height_m = round(float(np.median(floor)), 3) if floor else TAG10_HEIGHT_M
        print(f"tag 10 height {args.tag10_height_m} m "
              f"({'from ' + str(len(floor)) + ' floor tags in the map' if floor else 'default: no floor tags in the map'})",
              flush=True)
    else:
        args.tag10_height_m = float(args.tag10_height_m)
    site_auto = str(args.site).strip().lower() == "auto"
    if site_auto:
        # Placeholder until the defenders have localized; replaced by their starting line.
        site = np.array([0.0, 0.0, 3.0])
    else:
        sx, sz = parse_xyz(args.site, 2)
        site = np.array([sx, 0.0, sz])
    face = np.array([0.0, 0.0, -1.0]) if args.face == "base" else np.array([0.0, 0.0, 1.0])
    fence = default_fence(tag_poses, args.tag10_height_m, margin=args.fence_margin)
    if args.band_m > 0:                       # map y = 0 is the tag centre height
        fence.lo[1] = max(fence.lo[1], -args.band_m)
        fence.hi[1] = min(fence.hi[1], args.band_m)
    if args.room:
        x0, x1, z1 = (float(v) for v in args.room.split(","))
        fence.lo[0], fence.hi[0], fence.hi[2] = x0, x1, z1
    cfg = BrainCfg(site=site, fence=fence, guard_radius=args.guard_radius,
                   defender_speed=args.speed_cap, hold_height_y=args.hold_height_m - args.tag10_height_m,
                   formation=args.formation, hold_spread=args.hold_spread_m, tri_rise_m=args.tri_rise_m,
                   max_lead_s=args.max_lead_s, latency_s=args.latency_s)
    if args.formation == "v":
        if args.v_tag not in tag_poses:
            raise SystemExit(f"--v_tag {args.v_tag} is not in {map_path} (tags {sorted(tag_poses)})")
        t = tag_poses[args.v_tag][:3, 3]
        cfg.v_center = np.array([float(t[0]), 0.0, float(t[2]) - args.v_front_m])   # toward tag 10 = smaller z
        cfg.v_width, cfg.v_depth = args.v_width_m, args.v_depth_m
        if args.v_tag in ignore_tags:
            print(f"NOTE: v_tag {args.v_tag} is in ignore_tags (moved since mapping): the V is placed at its "
                  f"MAPPED spot, i.e. where tag {args.v_tag} was when the map was made", flush=True)
        print(f"V formation: tip {np.round(cfg.v_center[[0, 2]], 2).tolist()} (x, z), {args.v_front_m} m in front "
              f"of tag {args.v_tag}; arms {args.v_width_m} m apart, {args.v_depth_m} m back", flush=True)
    if site_auto:
        cfg.front = np.array([0.0, 0.0, -1.0])   # table behind the line: attacks come from tag 10's side
        if args.formation == "v":                # the table sits just behind the V
            cfg.site = np.array([cfg.v_center[0], 0.0, cfg.v_center[2] + args.v_depth_m + args.site_behind_m])
    brain = DefenseBrain(cfg)
    rc_cap = min(MAX_RC, int(round(args.speed_cap * RC_PER_MPS)))
    print(f"config {cfg_path if file_cfg else '(none)'}", flush=True)
    print(f"defenders {defenders} vs attackers {'auto (any non-defender in the feed)' if auto else attackers}; "
          f"map {map_path} tags {sorted(tag_poses)} (ignored: {sorted(ignore_tags) or 'none'}); "
          f"site {brain.cfg.site.round(2).tolist()}{' (auto: updated from the starting line)' if site_auto and args.formation != 'v' else ''}; fence lo {fence.lo.round(2).tolist()} hi {fence.hi.round(2).tolist()}; "
          f"rc cap {rc_cap}; mode {'SIM' if args.sim else 'FLY' if args.fly else 'DRY RUN'}", flush=True)

    feeds = []
    if args.feed in ("auto", "swarm"):
        hub = None
        if args.swarm_hub != "auto":
            h, _, p = args.swarm_hub.rpartition(":")
            hub = (h, int(p))
        feeds.append(SwarmBusFeed(None if auto else attackers, exclude=defenders, hub=hub))
    if args.feed in ("auto", "gui"):
        feeds.append(TelemetryFeed(args.feed_port, None if auto else attackers,
                                   {k: v for k, v in tag_poses.items() if k not in ignore_tags}, exclude=defenders,
                         convention=args.feed_frame,
                         height_from_h=args.feed_height == "h", tag10_height=args.tag10_height_m))
    feed = CombinedFeed(feeds)
    status_dst = None
    if args.status:
        host, _, port = args.status.rpartition(":")
        status_dst = (host, int(port))
    status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _terminate(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, _terminate)        # a stop button / kill lands like Ctrl-C
    signal.signal(signal.SIGHUP, _terminate)

    stop = threading.Event()
    units: dict[str, object] = {}
    all_units: list = []                          # every real defender created, flying or not
    if args.sim:
        for n, p in brain.hold_points(defenders).items():
            units[n] = SimDefender(n, fence.clamp(p), face)
    else:
        import test_link
        from defender_link import start_defenders
        from upright_positional import UprightPositioner as Positioner   # print-orientation proof
        units_lock = threading.Lock()
        launching = [0]                          # takeoffs in progress
        ready: dict[str, object] = {}            # connected, not yet flying (swarm mode waits for FLY)

        def make_unit(dl):
            pos = Positioner(dl.name.lower().replace("-", "_"), map_path=map_path,
                             tags_path=Path(args.tags) if args.tags else None,
                             ignore=ignore_tags, name=dl.name)
            u = RealDefender(dl, pos, face, args, stop)
            with units_lock:
                all_units.append(u)              # landed at shutdown even if caught mid-takeoff
            return u

        def launch(u) -> None:
            """takeoff (+ climb) for one defender, then hand it to the control loop."""
            with units_lock:
                launching[0] += 1
            t_launch = time.time()
            try:
                if stop.is_set():
                    return
                with u.dl.sdk_lock:              # never interleave with this drone's streamon
                    u.airborne = True            # set BEFORE takeoff: an interrupt now must still land it
                    reply = test_link.sdk(u.dl.cmd, "takeoff", 15)
                    print(f"{u.name} takeoff -> {reply}", flush=True)
                    if args.climb_cm and not stop.is_set() and not u.landed:
                        print(f"{u.name} up {args.climb_cm} -> "
                              f"{test_link.sdk(u.dl.cmd, f'up {args.climb_cm}', 10)}", flush=True)
                    if args.advance_cm and not stop.is_set() and not u.landed:
                        print(f"{u.name} forward {args.advance_cm} (toward tag 10) -> "
                              f"{test_link.sdk(u.dl.cmd, f'forward {args.advance_cm}', 10)}", flush=True)
                print(f"{u.name} airborne in {time.time() - t_launch:.1f} s -> control loop", flush=True)
                if not stop.is_set():
                    with units_lock:
                        units[u.name] = u
            finally:
                with units_lock:
                    launching[0] -= 1

        def on_ready(dl) -> None:
            """Runs in this drone's own thread the moment it answers 'command'."""
            bat = test_link.sdk(dl.cmd, "battery?", 2)
            if args.fly and (not bat.isdigit() or int(bat) < TAKEOFF_BATTERY):
                print(f"[{dl.name}] FAIL: battery {bat!r} < {TAKEOFF_BATTERY}% -> not flying it", flush=True)
                return
            u = make_unit(dl)
            if args.swarm_external:
                print(f"[{dl.name}] PASS (battery {bat}%)", flush=True)   # swarm_policies counts these
                with units_lock:
                    ready[dl.name] = u
            elif args.fly:
                launch(u)                        # take off NOW, independent of the other defenders
            else:
                with units_lock:
                    units[dl.name] = u           # dry run: localize and print rc, no takeoff

        expected = start_defenders(defenders, stop, on_ready)
        missing = sorted(set(defenders) - set(expected))
        if missing:
            print(f"WARNING: no stick plugged in for {missing} -- flying without them", flush=True)
        if not expected:
            raise SystemExit("no defender sticks plugged in (config/dongles.json maps drone -> stick)")
        if args.swarm_external:
            t0 = time.time()                     # wait for every defender to answer (or 45 s), then ask
            while len(ready) < len(expected) and time.time() - t0 < 45.0:
                time.sleep(0.1)
            late = sorted(set(expected) - set(ready))
            if late:
                print(f"WARNING: {late} not ready -- going with {sorted(ready)}", flush=True)
            if args.fly:
                print("defenders ready. Type FLY:", end="", flush=True)   # swarm_policies answers FLY
                answer = sys.stdin.readline().strip()
                print("", flush=True)
                if answer != "FLY":
                    raise SystemExit(f"no takeoff (got {answer!r})")
                threads = [threading.Thread(target=launch, args=(u,), daemon=True) for u in ready.values()]
                for th in threads:
                    th.start()
            else:
                with units_lock:
                    units.update(ready)

    out = DD_ROOT / "recordings"
    out.mkdir(exist_ok=True)
    log_path = out / f"defend_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log = open(log_path, "w", encoding="utf-8")
    print(f"log {log_path}. Ctrl-C lands all.", flush=True)

    start = prev = time.time()
    contacts: set[tuple[str, str]] = set()
    plans: dict[str, Plan] = {}
    last_plan = 0.0
    last_log = 0.0
    last_role: dict[str, tuple] = {}
    last_alive: dict[str, bool] = {}
    try:
        while not stop.is_set() and time.time() - start < args.seconds:
            now = time.time()
            dt, prev = now - prev, now
            tracks = feed.tracks()
            positions = {}
            current = list(units.items())        # defenders join as they take off
            for n, u in current:
                if isinstance(u, RealDefender):
                    u.update_fix()
                positions[n] = None if getattr(u, "landed", False) else u.position()[0]
                if positions[n] is not None and n not in brain.anchors:
                    brain.anchors[n] = fence.clamp(positions[n].copy())   # "home": first fix after takeoff
                    if args.formation == "home":
                        print(f"{n} home hold position {np.round(brain.anchors[n], 2).tolist()}", flush=True)
                    if site_auto and args.formation != "v":
                        # The table sits just behind our starting line: mean x of the starting
                        # spots, site_behind_m further from tag 10 than the rearmost one.
                        homes = np.array(list(brain.anchors.values()))
                        brain.cfg.site = np.array([float(homes[:, 0].mean()), 0.0,
                                                   float(homes[:, 2].max()) + args.site_behind_m])
                        print(f"site (table, behind our line) {np.round(brain.cfg.site[[0, 2]], 2).tolist()} "
                              f"(x, z) from {len(homes)} defender start(s)", flush=True)
            if now - last_plan >= 0.1:                       # brain at 10 Hz
                plans = brain.plan(positions, tracks)
                last_plan = now
                for n, p in plans.items():                  # log every job / target change at once
                    role = (p.mode, p.attacker)
                    if last_role.get(n) != role:
                        old_role = last_role.get(n)
                        print(f"[{now - start:6.1f}s] ROLE {n}: "
                              f"{(old_role[0] + (' ' + old_role[1] if old_role[1] else '')) if old_role else 'start'}"
                              f" -> {p.mode}{' ' + p.attacker if p.attacker else ''}"
                              f"{'' if p.target is None else '  target ' + str(np.round(p.target, 2).tolist())}",
                              flush=True)
                        last_role[n] = role
            for t in tracks:                                 # attackers appearing / going down
                if last_alive.get(t.name) != t.alive:
                    print(f"[{now - start:6.1f}s] ATTACKER {t.name} {'LIVE' if t.alive else 'DOWN'} at "
                          f"{np.round(t.pos, 2).tolist()} (report age {t.age:.2f} s)", flush=True)
                    last_alive[t.name] = t.alive
            rows = []
            for n, u in current:
                p = plans.get(n)
                pos = positions[n]
                rc = (0, 0, 0, 0)
                if isinstance(u, RealDefender):
                    rc = real_safety(u, p, pos, rc_cap, args, now)
                elif p is not None and p.target is not None and pos is not None:
                    rc = rc_from(velocity_cmd(p, pos, args.speed_cap), u.R, face, rc_cap)
                if isinstance(u, SimDefender):
                    u.command(rc, dt)
                for t in tracks:
                    if t.alive and pos is not None and np.linalg.norm(t.pos - pos) < CAPTURE_M \
                            and (n, t.name) not in contacts:
                        contacts.add((n, t.name))
                        print(f"CONTACT {n} -> {t.name} at {np.round(t.pos, 2).tolist()}", flush=True)
                        log.write(json.dumps({"t": round(now - start, 2), "event": "contact",
                                              "defender": n, "attacker": t.name}) + "\n")
                rows.append({"name": n, "pos": None if pos is None else np.round(pos, 3).tolist(),
                             "mode": p.mode if p else "idle", "attacker": p.attacker if p else None,
                             "target": None if p is None or p.target is None else np.round(p.target, 3).tolist(),
                             "rc": list(u.rc), "note": getattr(u, "note", "")})
            att = [{"name": t.name, "pos": np.round(t.pos, 3).tolist(), "vel": np.round(t.vel, 2).tolist(),
                    "age": round(t.age, 2), "alive": t.alive} for t in tracks]
            msg = {"t": round(now - start, 2), "site": np.round(brain.cfg.site, 3).tolist(),
                   "defenders": rows, "attackers": att,
                   "feed": {"datagrams": feed.datagrams, "age": round(now - feed.last_rx, 2) if feed.last_rx else None}}
            log.write(json.dumps(msg) + "\n")
            if args.log_every_s and now - last_log >= args.log_every_s:
                last_log = now
                age = f"{now - feed.last_rx:.2f} s ago" if feed.last_rx else "nothing yet"
                print(f"---- t={now - start:6.1f}s | feed {feed.datagrams} msgs, last {age} | "
                      f"table {np.round(brain.cfg.site[[0, 2]], 2).tolist()} | flying {len(current)}/{len(defenders)}",
                      flush=True)
                fmt = lambda v: "--" if v is None else "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"  # noqa: E731
                for r in rows:
                    u = units.get(r["name"])
                    h = bat = fix = None
                    if isinstance(u, RealDefender):
                        h, bat = u.dl.heights.get(u.name), u.dl.batteries.get(u.name)
                        fix = None if u.loc is None else now - u.loc_t
                    print(f"  {r['name']:<8} {r['mode']:<5} {('-> ' + r['attacker']) if r['attacker'] else '':<12} "
                          f"pos {fmt(r['pos'])} tgt {fmt(r['target'])} rc {tuple(r['rc'])} "
                          f"fix {'--' if fix is None else f'{fix:.2f}s'} h {'--' if h is None else f'{h:.2f}m'} "
                          f"bat {'--' if bat is None else f'{bat}%'} | {r['note']}", flush=True)
                for a in att:
                    print(f"  ATT {a['name']:<8} {'live' if a['alive'] else 'DOWN'}  pos {fmt(a['pos'])} "
                          f"vel {fmt(a['vel'])} age {a['age']:.2f}s", flush=True)
            if status_dst:
                try:
                    status_sock.sendto(json.dumps(msg).encode(), status_dst)
                except OSError:
                    pass
            if (not args.sim and units and all(getattr(u, "landed", False) for u in list(units.values()))
                    and not launching[0] and len(all_units) >= len(expected)
                    and all(u.landed or not u.airborne for u in all_units)):
                print("all defenders landed", flush=True)
                break
            stop.wait(0.05)
    except KeyboardInterrupt:
        print("Ctrl-C -> landing all", flush=True)
    finally:
        stop.set()
        for u in list(all_units):
            u.land("end of run")
        log.close()
        feed.close()
        print(f"contacts: {sorted(contacts)}", flush=True)
    return 0


def real_safety(u: RealDefender, p: Plan | None, pos, rc_cap: int, args, now: float):
    """rc for one real defender, with link / battery / lost-tag handling."""
    if u.landed:
        return (0, 0, 0, 0)
    if args.fly:
        if not u.dl.connected(now):
            if u.airborne and not u.abort:
                u.abort = True
                print(f"{u.name} lost link in flight; will land on reconnect", flush=True)
            u.send((0, 0, 0, 0))
            return u.rc
        if u.abort:
            u.land("link came back after a drop")
            return (0, 0, 0, 0)
        bat = u.dl.batteries.get(u.name)
        if bat is not None and bat < LAND_BATTERY:
            u.land(f"battery {bat}%")
            return (0, 0, 0, 0)
    if pos is None:                                   # no fresh fix: hover, then search, then land
        u.lost_since = u.lost_since or now
        lost = now - u.lost_since
        u.note = f"no tag {lost:.1f}s"
        limit = args.lost_timeout_s if u.loc is not None else args.first_fix_s   # never seen a tag yet
        if lost > limit:
            u.land(f"no tag for {lost:.0f} s" + ("" if u.loc is not None else " since takeoff (video?)"))
            return (0, 0, 0, 0)
        # Hold tag 10's height on the Tello's own height sensor (most tags are around there).
        h = u.dl.heights.get(u.name)
        up = 0
        if h is not None and abs(args.tag10_height_m - h) > 0.08:
            up = int(max(-20, min(20, 60 * (args.tag10_height_m - h))))
        yaw = 0
        if lost > LOST_HOVER_S:
            # Widening sweep: legs of 2, 4, 6, ... s, alternating direction, so each swing
            # turns further than the last and the camera eventually passes every wall.
            t, k = lost - LOST_HOVER_S, 0
            while t > SEARCH_LEG_S * (k + 1):
                t -= SEARCH_LEG_S * (k + 1)
                k += 1
            yaw = -SEARCH_YAW if k % 2 == 0 else SEARCH_YAW
        u.note += f" (searching, {limit - lost:.0f} s left)"
        u.send((0, 0, up, yaw))
        return u.rc
    u.lost_since = None
    tags = u.loc["tags"] if u.loc else []
    if u.upside_down:                    # a bad tag fix: never steer on it, just hold
        u.note = f"tag fix looks mirrored (tags {tags}), holding"
        u.send((0, 0, 0, 0))
        return u.rc
    if p is None or p.target is None:
        u.note = f"tags {tags} idle"
        u.send((0, 0, 0, 0))
        return u.rc
    u.note = f"{p.mode}{' ' + p.attacker if p.attacker else ''} tags {tags} pos {np.round(pos, 2).tolist()}"
    u.send(rc_from(velocity_cmd(p, pos, args.speed_cap), u.R, u.face, rc_cap))
    return u.rc


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fly scripted swarm policies: the umbrella runs one or more at once, --run says which
policy flies which drones. The policies are one file each in policies/ (add one there).

A policy decides each airborne drone's rc command every tick (13.5 Hz, the engine's rc
rate). No camera, no tags: height comes from the Tello status stream (tof, else h).

Policies (--run <policy>:<drones>):
  spin    defensive dummy: take off, hold the takeoff height, spin in place (yaw)
  levels  hover at two heights: the lower-numbered half of its drones at 3 ft (91 cm),
          the rest at 6 ft (183 cm); odd counts put the extra drone low
  hover   take off and hold the takeoff height, or --height <cm> (e.g. 200 = 2 m)
  formation  swarm_formation.py on those drones (its own process, video + tags); with it
          the run lasts until the formation ends unless --duration is given

Umbrella: give --run once per group, e.g. drones 1,2 spin while 3,4,5 split 3 ft / 6 ft:
  python swarm_policies.py --run spin:1,2 --run levels:3,4,5            # dry-run
  python swarm_policies.py --run spin:1,2 --run levels:3,4,5 --fly --duration 30
  python swarm_policies.py --run spin:6,7,9 --run formation:4,5 --fly    # defend + attack
One --run flies just that policy. A drone may be in only one group.

Telemetry: every policy can publish live data and read what the others publish
(policies/telemetry.py): self.bus.publish(topic, data) / self.bus.latest(topic). External
policies (formation) get the same calls over localhost UDP via TelemetryClient.from_env().
The umbrella itself publishes "status/drone<N>" (height, battery, rc, policy) every tick
for the drones it flies; formation publishes "formation/drone<N>" (tag-frame pose).

DEFAULT IS DRY-RUN (connect + status only, no takeoff). --fly to fly.
Safety: all-or-nothing preflight; a failed takeoff, Ctrl+C or the end of --duration lands
ALL; a drone that loses its status stream or drops below LAND_BATTERY lands on its own.
Reads drone_deployment_infra/config/dongles.json via core/paths.py. Libs: stdlib only.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import signal
import termios
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from paths import DONGLES_PATH  # noqa: E402  (config/ lives in core)
from policies import (POLICIES, FormationPolicy, HoverPolicy, SpinPolicy, TelemetryBus,  # noqa: E402
                      TelemetryHub, Umbrella)
from policies.base import HIGH_CM  # noqa: E402
from policies.formation import PROMPT as FORMATION_PROMPT  # noqa: E402
from policies.spin import SPIN_YAW_RC  # noqa: E402

# ======================= SETTINGS =======================
PERIOD_S = 0.074             # one rc packet per tick (13.5 Hz)
TAKEOFF_STAGGER_S = 0.5
STATUS_STALE_S = 3.0         # no status packets this long while airborne -> land that drone
MIN_BATTERY_FLY = 40
MIN_HOVER_CM, MAX_HOVER_CM = 40, 250   # --height limits (tof reads up to ~4 m)
EXTERNAL_READY_S = 90        # a formation group must pass its preflight (video) by then
LAND_BATTERY = 15

DRONE_ADDR = ("192.168.10.1", 8889)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
NOFLY = {"command", "battery?"}
RC_RE = re.compile(r"^rc (-?\d+) (-?\d+) (-?\d+) (-?\d+)$")


# ======================= DRONE LINK =======================
class Drone:
    def __init__(self, num, iface, ssid, fly):
        self.num, self.iface, self.ssid, self.fly = num, iface, ssid, fly
        self.name = f"drone{num}"
        self.sock = self._sock(9000 + num)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.st, self.st_t = {}, 0.0
        self.airborne = self.landed = False
        self.takeoff_t = None
        self.land_thread = None
        self.last_rc = (0, 0, 0, 0)

    def _sock(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Force traffic through THIS adapter (all drones share 192.168.10.1).
        s.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, self.iface.encode())
        s.bind(("", port))
        return s

    def log(self, msg):
        print(f"[{self.name}] {msg}", flush=True)

    def _check(self, cmd):
        if cmd in NOFLY:
            return
        if self.fly:
            if cmd in ("takeoff", "land", "emergency"):
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

    def rc(self, a, b, c, d):
        self.last_rc = (a, b, c, d)
        if self.fly:
            self.send_noreply(f"rc {a} {b} {c} {d}")

    # ---- status stream ----
    def state_loop(self):
        ss = self._sock(8890)
        ss.settimeout(1.0)
        while not self.stop.is_set():
            try:
                data = ss.recv(2048).decode(errors="ignore")
            except socket.timeout:
                continue
            self.st = dict(kv.split(":", 1) for kv in data.strip().split(";") if ":" in kv)
            self.st_t = time.time()
        ss.close()

    def _stv(self, key):
        if time.time() - self.st_t > 1.0:
            return None
        try:
            return float(self.st[key])
        except (KeyError, ValueError):
            return None

    def status_age(self):
        return time.time() - self.st_t

    def height_cm(self):
        tof = self._stv("tof")
        if tof is not None and 10 <= tof <= 400:
            return tof
        h = self._stv("h")
        return h if h is not None and 0 < h <= 400 else None

    def battery(self):
        b = self._stv("bat")
        return None if b is None else int(b)

    def on_ground(self):
        h = self._stv("h")
        return None if h is None else h <= 5

    def takeoff(self):
        self.airborne = True
        for _ in range(2):
            if self.send("takeoff", 20) == "ok":
                self.takeoff_t = time.time()
                return True
            if not self.on_ground():           # reply lost but it's up
                self.takeoff_t = time.time()
                return True
        return False

    def land(self):
        if not self.airborne or self.landed:
            return True
        self.send_noreply("rc 0 0 0 0")
        for _ in range(3):
            self.send_noreply("land")
            t0 = time.time()
            while time.time() - t0 < 8:
                if self.on_ground():
                    self.landed = True
                    self.log("landed")
                    return True
                time.sleep(0.2)
        self.log("WARNING: land NOT confirmed - check/catch this drone!")
        return False

    def land_async(self, why):
        if self.land_thread is not None or self.landed:
            return
        self.log(f"{why} -> LANDING")
        self.land_thread = threading.Thread(target=self.land, daemon=True)
        self.land_thread.start()

    def close(self):
        self.stop.set()
        self.sock.close()


# ======================= HELPERS =======================
def iface_ip(iface):
    out = subprocess.run(["ip", "-j", "-4", "addr", "show", iface],
                         capture_output=True, text=True).stdout or "[]"
    for dev in json.loads(out):
        for a in dev.get("addr_info", []):
            if a.get("local", "").startswith("192.168.10."):
                return a["local"]
    return None


def active_ssid(iface):
    out = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi", "list",
                          "ifname", iface, "--rescan", "no"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("yes:"):
            return line[4:].replace("\\:", ":")
    return None


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


def load_drones(cfg, nums, fly):
    with open(cfg) as f:
        data = json.load(f)
    data = data.get("dongles", []) if isinstance(data, dict) else data
    entries = [e for e in data if e.get("ssid") and int(e["number"]) in nums]
    missing = nums - {int(e["number"]) for e in entries}
    if missing:
        sys.exit(f"No config/ssid for drone(s) {sorted(missing)} in {cfg}")
    ifaces = [e["iface"] for e in entries]
    if len(set(ifaces)) != len(ifaces):
        sys.exit("Two drones share one adapter in the config. Fix it first.")
    drones, bad = [], []
    for e in entries:
        try:
            drones.append(Drone(int(e["number"]), e["iface"], e["ssid"], fly))
        except OSError as ex:
            bad.append(f"drone{e['number']}: {e['iface']}: {ex.strerror} (USB stick plugged in?)")
    if bad:
        for d in drones:
            d.close()
        sys.exit("Cannot run:\n  " + "\n  ".join(bad))
    return drones


class External:
    """A policy that flies its drones from its own process (e.g. policies/formation.py)."""

    def __init__(self, policy, nums, fly, config, env=None):
        self.name, self.nums, self.started = policy.name, nums, False
        self.proc = subprocess.Popen(policy.command(nums, fly, config), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)   # our Ctrl+C reaches it once, via stop()
        self.prompted = threading.Event()
        self.passed = 0                                    # its drones' preflight PASS lines
        threading.Thread(target=self._relay, daemon=True).start()

    def _relay(self):
        buf, fd = b"", self.proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode(errors="ignore")
                self.passed += "] PASS" in text
                print(f"[{self.name}] {text}", flush=True)
            if FORMATION_PROMPT.encode() in buf:        # its FLY prompt has no newline
                buf = b""
                self.prompted.set()
        if buf:
            print(f"[{self.name}] {buf.decode(errors='ignore')}", flush=True)

    def alive(self):
        return self.proc.poll() is None

    def wait_ready(self, timeout, fly):
        """True once all its drones passed preflight (and, flying, it asks for FLY)."""
        t0 = time.time()
        while time.time() - t0 < timeout and self.alive():
            if self.prompted.is_set() if fly else self.passed >= len(self.nums):
                return True
            time.sleep(0.2)
        return False

    def answer(self, text):
        try:
            self.proc.stdin.write(text.encode() + b"\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        self.started = text == "FLY"

    def stop(self, timeout=60):
        """One SIGINT = its land all; waits for its landing."""
        if not self.alive():
            return
        if self.prompted.is_set() and not self.started:
            self.answer("no")                              # still at its prompt: no takeoff
        else:
            self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            print(f"[{self.name}] WARNING: still running after {timeout} s - check its drones!", flush=True)


def parse_runs(runs, yaw_rc, formation_args, hover_cm=None):
    groups, externals = {}, []
    for r in runs:
        name, _, nums = r.partition(":")
        if name not in POLICIES or not nums:
            sys.exit(f"bad --run {r!r}: use <policy>:<drones>, policy one of {sorted(POLICIES)}")
        nums = [int(x) for x in nums.split(",") if x.strip()]
        if name == "formation":
            externals.append((FormationPolicy(formation_args), nums))
            continue
        if name == "spin":
            p = SpinPolicy(yaw_rc)
        elif name == "hover":
            p = HoverPolicy(hover_cm)
        else:
            p = POLICIES[name]()
        groups[p] = nums
    try:
        return Umbrella(groups | dict(externals)), externals   # Umbrella checks no drone is in two groups
    except ValueError as ex:
        sys.exit(str(ex))


def preflight(d):
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
    if d.fly and (not b.isdigit() or int(b) < MIN_BATTERY_FLY):
        d.log(f"FAIL: battery {b}% < {MIN_BATTERY_FLY}%")
        return False
    t0 = time.time()
    while not d.st and time.time() - t0 < 3:
        time.sleep(0.1)
    if not d.st:
        d.log("FAIL: no status packets -> no height sensor")
        return False
    d.log(f"PASS (battery {b}%, height {d.height_cm()} cm)")
    return True


# ======================= MAIN =======================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DONGLES_PATH), help="dongles JSON (default config/dongles.json)")
    ap.add_argument("--run", action="append", required=True,
                    help="<policy>:<drones>, repeatable, e.g. --run spin:1,2 --run levels:3,4,5")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds after takeoff, then land all (default 20; with formation: until it ends)")
    ap.add_argument("--height", type=int, default=None,
                    help=f"hover height above the floor, cm ({MIN_HOVER_CM}-{MAX_HOVER_CM}; default: takeoff height)")
    ap.add_argument("--yaw", type=int, default=SPIN_YAW_RC, help="spin rc yaw, -100..100")
    ap.add_argument("--formation-args", default="", help='extra swarm_formation.py args, e.g. "--hold 10 --fast"')
    ap.add_argument("--fly", action="store_true", help="actually fly (default dry-run)")
    args = ap.parse_args()

    if args.height is not None and not MIN_HOVER_CM <= args.height <= MAX_HOVER_CM:
        sys.exit(f"--height {args.height} cm: use {MIN_HOVER_CM}-{MAX_HOVER_CM}")
    umbrella, ext_specs = parse_runs(args.run, args.yaw, args.formation_args, args.height)
    ext_nums = {n for _, nums in ext_specs for n in nums}
    groups = {p: nums for p, nums in umbrella.groups.items() if not getattr(p, "external", False)}
    bus = TelemetryBus()
    hub = TelemetryHub(bus)
    policy = Umbrella(groups, bus)
    drones = load_drones(args.config, set(policy.owner), args.fly)
    policy.setup(drones)
    print("Plan: " + ", ".join([f"{d.name}={policy.describe(d)}" for d in drones]
                               + [f"drone{n}={p.name}" for p, nums in ext_specs for n in nums])
          + ("" if args.fly else "  (DRY-RUN)"), flush=True)
    if args.duration is None and not ext_specs:
        args.duration = 20.0

    externals = [External(p, nums, args.fly, args.config, hub.env()) for p, nums in ext_specs]
    res = parallel(drones, preflight)
    failed = [d.name for d in drones if not res.get(d.num)]
    for e in externals:
        if not e.wait_ready(EXTERNAL_READY_S, args.fly):
            failed.append(f"{e.name} {e.nums}")
    if failed:
        for e in externals:
            e.stop()
        for d in drones:
            d.close()
        sys.exit(f"Preflight FAILED for {failed}. Nobody takes off.")

    try:
        if args.fly:
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)  # drop keys typed during preflight
            answer = input(f"All {len(drones) + len(ext_nums)} PASS. Space clear, drones 1 m+ apart, "
                           f"{max(HIGH_CM, args.height or 0) + 10} cm headroom"
                           + (", other groups clear of the formation slots" if externals else "")
                           + "? Type FLY: ").strip()
            if answer != "FLY":
                sys.exit(f"Aborted, no takeoff (got {answer!r}).")
            for e in externals:
                e.answer("FLY")

            def to(d):
                time.sleep(TAKEOFF_STAGGER_S * drones.index(d))
                return d.takeoff()
            res = parallel(drones, to)
            if not all(res.get(d.num) for d in drones):
                print("A takeoff failed -> landing all", flush=True)
                return
        else:
            now = time.time()
            for d in drones:
                d.airborne, d.takeoff_t = True, now          # run the loop, send nothing
            args.duration = min(args.duration or 15.0, 15.0)
        print(">>> RUNNING " + (f"{args.duration:.0f} s" if args.duration else "until formation ends")
              + " (Ctrl+C = land all)", flush=True)

        end, last_print = time.time() + (args.duration or float("inf")), 0.0
        while time.time() < end:
            t_loop = time.time()
            active = [d for d in drones if d.land_thread is None and not d.landed]
            if externals and not any(e.alive() for e in externals):
                print("Formation ended -> landing all", flush=True)
                return
            if not active and not externals:
                print("All drones landed.", flush=True)
                return
            for d in active:
                bat = d.battery()
                if d.status_age() > STATUS_STALE_S:
                    d.land_async(f"no status for {d.status_age():.0f} s")
                    continue
                if args.fly and bat is not None and bat < LAND_BATTERY:
                    d.land_async(f"battery {bat}%")
                    continue
                rc = policy.step(d, t_loop - d.takeoff_t)
                if rc is None:
                    d.land_async(f"{policy.describe(d)} done")
                    continue
                d.rc(*rc)
                bus.publish(f"status/{d.name}", {"policy": policy.owner[d.num].name, "height_cm": d.height_cm(),
                                                 "battery": bat, "rc": list(d.last_rc)}, t_loop)
            if t_loop - last_print >= 1.0:
                last_print = t_loop
                for d in active:
                    h = d.height_cm()
                    a, b, c, y = d.last_rc
                    print(f"[{d.name}] {policy.describe(d):<12} height "
                          + (f"{h:.0f}cm" if h is not None else "?")
                          + f"  bat {d.battery()}%  rc {a:+d} {b:+d} {c:+d} {y:+d}"
                          + ("" if args.fly else " (DRY)"), flush=True)
            time.sleep(max(0.0, PERIOD_S - (time.time() - t_loop)))
        print("Duration over -> landing all", flush=True)
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing all", flush=True)
    finally:
        for e in externals:
            e.stop()
        if args.fly:
            for d in drones:
                if d.land_thread is not None:
                    d.land_thread.join(30)
            parallel([d for d in drones if d.airborne and not d.landed], lambda d: d.land())
        for d in drones:
            d.close()
        hub.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()

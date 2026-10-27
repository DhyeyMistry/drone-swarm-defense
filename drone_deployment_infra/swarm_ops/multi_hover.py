#!/usr/bin/env python3
"""Multi-Tello hover: N drones (one USB Wi-Fi adapter each) take off together,
hover in place, land together.

Add drones: fill "ssid" for the adapter entry in the JSON config (your dongles file).
Entries with empty ssid are skipped. Select a subset with --only 1,3,4,5.

Safety:
- All-or-nothing preflight: if ANY drone fails a check, NONE take off.
- Any takeoff failure, lost link, low battery or Ctrl+C -> land ALL.

Usage:
  python multi_hover.py --only 1,3,4,5 --hover 10      # dongles from config/dongles.json
"""
import argparse
import json
import socket
import subprocess
import sys
import threading
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from paths import DONGLES_PATH  # noqa: E402  (config/ lives in core)

DRONE_ADDR = ("192.168.10.1", 8889)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
MIN_BATTERY = 30
LAND_BATTERY = 15


class Drone:
    def __init__(self, number, iface, ssid):
        self.number, self.iface, self.ssid = number, iface, ssid
        self.name = f"drone{number}"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Force traffic through THIS adapter (all drones share 192.168.10.1).
        self.sock.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, iface.encode())
        self.sock.bind(("", 9000 + number))  # unique local port per drone
        self.lock = threading.Lock()
        self.takeoff_sent = False
        self.misses = 0

    def log(self, msg):
        print(f"[{self.name}] {msg}", flush=True)

    def _drain(self):
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recv(1024)
        except (BlockingIOError, OSError):
            pass
        self.sock.setblocking(True)

    def send(self, cmd, timeout=10, quiet=False):
        with self.lock:
            self._drain()
            self.sock.settimeout(timeout)
            self.sock.sendto(cmd.encode(), DRONE_ADDR)
            try:
                reply = self.sock.recv(1024).decode(errors="ignore").strip().lower()
            except socket.timeout:
                reply = "timeout"
        if not quiet:
            self.log(f"{cmd} -> {reply}")
        return reply

    def force_land(self):
        """Land even if another command is still waiting for its reply."""
        if self.lock.acquire(timeout=0.5):
            self.lock.release()
            for _ in range(3):
                if self.send("land", 15) == "ok":
                    return True
            return False
        self.sock.sendto(b"land", DRONE_ADDR)  # lock busy: fire without waiting
        self.log("land sent (no reply awaited)")
        time.sleep(1)
        return self.send("land", 15) == "ok"


# ---------- helpers ----------
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


def run_all(drones, fn):
    """Run fn(drone) on all drones in parallel; Ctrl+C stays responsive."""
    results = {}

    def wrap(d):
        try:
            results[d.number] = fn(d)
        except Exception as e:
            d.log(f"error: {e}")
            results[d.number] = False

    threads = [threading.Thread(target=wrap, args=(d,), daemon=True) for d in drones]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(0.2)
    return results


def load_drones(path, only):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("dongles", list(data.values()))
    entries = [e for e in data if e.get("ssid")]
    if only:
        entries = [e for e in entries if int(e["number"]) in only]
        missing = only - {int(e["number"]) for e in entries}
        if missing:
            sys.exit(f"No config (or empty ssid) for drone(s): {sorted(missing)}")
    ifaces = [e["iface"] for e in entries]
    if len(set(ifaces)) != len(ifaces):
        sys.exit("Two drones share one adapter in the config. Fix it first.")
    if not entries:
        sys.exit("No drones selected.")
    return [Drone(int(e["number"]), e["iface"], e["ssid"]) for e in entries]


# ---------- phases ----------
def preflight(d):
    ip = iface_ip(d.iface)
    if not ip:
        d.log(f"FAIL: {d.iface} has no 192.168.10.x address (not connected?)")
        return False
    ssid = active_ssid(d.iface)
    if ssid != d.ssid:
        d.log(f"FAIL: {d.iface} is on {ssid!r}, expected {d.ssid!r}")
        return False
    for _ in range(5):
        if d.send("command", 3) == "ok":
            break
        time.sleep(1)
    else:
        d.log("FAIL: no reply to 'command'")
        return False
    bat = d.send("battery?", 5)
    if not bat.isdigit() or int(bat) < MIN_BATTERY:
        d.log(f"FAIL: battery {bat}% < {MIN_BATTERY}%")
        return False
    d.log(f"PASS ({d.iface}, {ip}, {ssid}, battery {bat}%)")
    return True


def takeoff(d):
    d.takeoff_sent = True
    return d.send("takeoff", 20) == "ok"


def keepalive(d):
    """Keeps drone from auto-landing; returns False on lost link or low battery."""
    bat = d.send("battery?", 3, quiet=True)
    if bat == "timeout":
        d.misses += 1
        d.log(f"no reply ({d.misses}/2)")
        return d.misses < 2
    d.misses = 0
    if bat.isdigit() and int(bat) < LAND_BATTERY:
        d.log(f"battery {bat}% < {LAND_BATTERY}% -> landing all")
        return False
    return True


def land_all(drones):
    airborne = [d for d in drones if d.takeoff_sent]
    if not airborne:
        return
    print(">>> LANDING ALL", flush=True)
    res = run_all(airborne, lambda d: d.force_land())
    for d in airborne:
        if not res.get(d.number):
            d.log("WARNING: land NOT confirmed - check/catch this drone!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DONGLES_PATH), help="dongles JSON (default config/dongles.json)")
    ap.add_argument("--only", default="", help="e.g. 1,3,4,5 (default: all with ssid)")
    ap.add_argument("--hover", type=float, default=10, help="hover seconds")
    args = ap.parse_args()
    only = {int(x) for x in args.only.split(",") if x.strip()}

    drones = load_drones(args.config, only)
    print(f"Drones: {', '.join(f'{d.name}({d.ssid}@{d.iface})' for d in drones)}")

    res = run_all(drones, preflight)
    failed = [d.name for d in drones if not res.get(d.number)]
    if failed:
        sys.exit(f"Preflight FAILED for {failed}. Nobody takes off.")

    if input(f"All {len(drones)} drones PASS. Space clear, drones >= 1.5 m apart? "
             f"Type FLY: ").strip() != "FLY":
        sys.exit("Aborted, no takeoff.")

    try:
        res = run_all(drones, takeoff)
        if not all(res.get(d.number) for d in drones):
            print("A takeoff failed -> landing all", flush=True)
            return
        print(f">>> ALL AIRBORNE, hovering {args.hover:.0f} s (Ctrl+C = land all)")
        end = time.time() + args.hover
        while time.time() < end:
            res = run_all(drones, keepalive)
            if not all(res.get(d.number) for d in drones):
                return
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing all", flush=True)
    finally:
        land_all(drones)
        for d in drones:
            d.sock.close()
        print("Done.")


if __name__ == "__main__":
    main()

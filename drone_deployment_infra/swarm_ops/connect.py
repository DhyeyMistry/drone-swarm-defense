#!/usr/bin/env python3
"""Verify (and if needed connect) Tello drones - fast path first.

  python connect.py 3 4 6 7            # verify; connect only those not already up
  python connect.py --all              # every drone with an ssid in dongles.json
  python connect.py 3 4 --down         # disconnect
  python connect.py --create --all     # ONE TIME per laptop: build all TELLO-<n> profiles from
                                       # dongles.json (needs "bssid"); drones can be OFF
  python connect.py --setup --all      # (older) only add autoconnect + static IP to existing profiles
  python connect.py 3 4 6 7 --wait 60  # run, THEN power the drones on: each is connected
                                       # as soon as it appears (only missing ones are retried)

With --setup done, NetworkManager joins each drone by itself shortly after it powers on
(profiles are locked to one adapter + one drone radio, so this is safe), and the static
IP (192.168.10.10<n>) skips the Tello's slow DHCP. Then this script is mostly a check:
already-connected drones are confirmed instantly; stragglers are connected on the spot.

Config: ../config/dongles.json (repo layout) if it exists, else ./dongles.json.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_CFG = os.path.join(HERE, "..", "config", "dongles.json")
DEFAULT_CFG = REPO_CFG if os.path.exists(REPO_CFG) else os.path.join(HERE, "dongles.json")


def run(cmd, timeout=40):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "timeout"


def adapter_present(iface):
    return os.path.exists(f"/sys/class/net/{iface}")


def iface_ip(iface):
    _, out = run(["ip", "-j", "-4", "addr", "show", iface], 5)
    try:
        for dev in json.loads(out or "[]"):
            for a in dev.get("addr_info", []):
                if a.get("local", "").startswith("192.168.10."):
                    return a["local"]
    except json.JSONDecodeError:
        pass
    return None


def active_ssid(iface):
    _, out = run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi", "list", "ifname", iface,
                  "--rescan", "no"], 10)
    for line in out.splitlines():
        if line.startswith("yes:"):
            return line[4:].replace("\\:", ":")
    return None


def check(entry):
    """(ok, info) without touching the connection."""
    iface, ssid = entry["iface"], entry["ssid"]
    if not adapter_present(iface):
        return False, f"adapter {iface} NOT DETECTED (hub unplugged / no hub power / replug it)"
    ip, cur = iface_ip(iface), active_ssid(iface)
    if ip and cur == ssid:
        return True, f"{iface}  {cur}  {ip}"
    if ip:
        return False, f"{iface} connected to {cur!r}, expected {ssid!r}"
    return False, "not connected"


def connect(n, entry, result, tries=3):
    iface, prof = entry["iface"], f"TELLO-{n}"
    msg = ""
    for _ in range(tries):
        code, msg = run(["nmcli", "con", "up", prof])
        if code == 0:
            break
        if "unknown connection" in msg:
            result[n] = (False, f"no profile '{prof}' (set it up first)")
            return
        run(["nmcli", "dev", "wifi", "rescan", "ifname", iface], 10)
        time.sleep(2)
    ok, info = check(entry)
    if not ok and "could not be found" in msg:
        info = "drone not broadcasting (off / asleep / flat battery)"
    result[n] = (ok, info)


def parallel(fn, items):
    ths = [threading.Thread(target=fn, args=a) for a in items]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


def setup(nums):
    """One time: autoconnect (infinite retries), static IP 192.168.10.10<n>, 2.4 GHz only
    (Tellos are 2.4 GHz; skipping the slow 5 GHz scan cuts connect time ~14 s -> ~5 s)."""
    for n in nums:
        prof = f"TELLO-{n}"
        code, msg = run(["nmcli", "con", "modify", prof,
                         "connection.autoconnect", "yes",
                         "connection.autoconnect-retries", "0",
                         "ipv4.method", "manual",
                         "ipv4.addresses", f"192.168.10.{100 + n}/24",
                         "ipv4.never-default", "yes", "ipv6.method", "ignore",
                         "802-11-wireless.band", "bg"])          # 2.4 GHz only: fast scans
        print(f"  {'OK  ' if code == 0 else 'FAIL'}  {prof}: "
              + (f"autoconnect + static 192.168.10.{100 + n} + 2.4 GHz only" if code == 0 else msg.splitlines()[-1]))
    print("Reconnect each drone once (python connect.py <n>) so the static IP takes effect.")


def profile_exists(name):
    _, out = run(["nmcli", "-t", "-f", "NAME", "con", "show"], 10)
    return name in out.splitlines()


def create(table, nums):
    """Build/refresh TELLO-<n> profiles straight from dongles.json (no drone needed):
    locked to the adapter + the drone's radio (BSSID), autoconnect, static IP, 2.4 GHz."""
    bad = [n for n in nums if not table[n].get("bssid")]
    if bad:
        sys.exit(f"dongles.json has no \"bssid\" for drone(s) {bad}. Add it (scan:  nmcli -f SSID,BSSID "
                 f"dev wifi list | grep TELLO) and run again.")
    for n in nums:
        e, prof = table[n], f"TELLO-{n}"
        settings = ["connection.interface-name", e["iface"],
                    "802-11-wireless.ssid", e["ssid"],
                    "802-11-wireless.bssid", e["bssid"].upper(),
                    "802-11-wireless.band", "bg",
                    "connection.autoconnect", "yes", "connection.autoconnect-retries", "0",
                    "ipv4.method", "manual", "ipv4.addresses", f"192.168.10.{100 + n}/24",
                    "ipv4.never-default", "yes", "ipv6.method", "ignore"]
        if profile_exists(prof):
            code, msg = run(["nmcli", "con", "modify", prof] + settings)
            how = "updated"
        else:
            code, msg = run(["nmcli", "con", "add", "type", "wifi", "con-name", prof,
                             "ifname", e["iface"], "ssid", e["ssid"]] + settings)
            how = "created"
        print(f"  {'OK  ' if code == 0 else 'FAIL'}  {prof}: " +
              (f"{how} ({e['ssid']} @ {e['iface']}, {e['bssid'].upper()}, 192.168.10.{100 + n})"
               if code == 0 else msg.splitlines()[-1]))
    print("Done. Power the drones on; they join by themselves. Check with:  python connect.py --all")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drones", nargs="*", type=int, help="drone numbers, e.g. 3 4 6 7")
    ap.add_argument("--all", action="store_true", help="every drone with an ssid in dongles.json")
    ap.add_argument("--down", action="store_true", help="disconnect instead")
    ap.add_argument("--setup", action="store_true", help="add autoconnect + static IP to existing profiles")
    ap.add_argument("--create", action="store_true",
                    help="one time per laptop: build all profiles from dongles.json (needs bssid)")
    ap.add_argument("--wait", type=float, default=0,
                    help="keep retrying the missing drones for up to this many seconds")
    ap.add_argument("--config", default=DEFAULT_CFG)
    args = ap.parse_args()

    with open(args.config) as f:
        data = json.load(f)
    data = data.get("dongles", []) if isinstance(data, dict) else data
    table = {int(e["number"]): e for e in data if e.get("ssid")}
    nums = sorted(table) if args.all else sorted(set(args.drones))
    if not nums:
        sys.exit("give drone numbers (e.g. python connect.py 3 4) or --all")
    missing = [n for n in nums if n not in table]
    if missing:
        sys.exit(f"no ssid for drone(s) {missing} in {args.config}")

    if args.create:
        create(table, nums)
        return 0
    if args.setup:
        setup(nums)
        return 0
    if args.down:
        parallel(lambda n: run(["nmcli", "con", "down", f"TELLO-{n}"]), [(n,) for n in nums])
        print(f"disconnected: {nums}")
        return 0

    t0 = time.time()
    result = {}
    parallel(lambda n: result.__setitem__(n, check(table[n])), [(n,) for n in nums])   # fast path
    todo = [n for n in nums if not result[n][0] and "NOT DETECTED" not in result[n][1]]
    if todo and args.wait > 0:
        print(f"already up: {[n for n in nums if result[n][0]]}  -> waiting for {todo} "
              f"(up to {args.wait:.0f}s; power them on now)")
        while todo and time.time() - t0 < args.wait:
            tries = {}
            parallel(connect, [(n, table[n], tries, 1) for n in todo])       # one try each
            for n in todo:
                result[n] = tries[n]
                if tries[n][0]:
                    print(f"  + drone {n} connected after {time.time() - t0:.1f}s", flush=True)
            todo = [n for n in todo if not result[n][0]]
            if todo:
                time.sleep(3)
    elif todo:
        print(f"already up: {[n for n in nums if result[n][0]]}  -> connecting {todo} ...")
        parallel(connect, [(n, table[n], result) for n in todo])
    for n in nums:
        good, info = result[n]
        print(f"  {'OK  ' if good else 'FAIL'}  drone {n}: {info}")
    ok = [n for n in nums if result[n][0]]
    if any("NOT DETECTED" in result[n][1] for n in nums):
        print("  hint: check adapters with  nmcli dev status | grep wlx   (powered hub plugged in?)")
    print(f"{len(ok)}/{len(nums)} ready in {time.time() - t0:.1f}s" + ("" if len(ok) == len(nums) else
          f"  -> retry:  python connect.py {' '.join(str(n) for n in nums if n not in ok)}"))
    return 0 if len(ok) == len(nums) else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Number the USB WiFi dongles and record which Tello each one flies.

Plug in one dongle at a time, join its drone's WiFi, then let this script
capture it. Each dongle is identified by its USB interface name (``wlx<mac>``,
which is just the MAC with the colons stripped), so the mapping survives
reboots and re-plugs. The result is written to ``dongles.json`` next to this
script and can be committed so everyone shares the same dongle -> drone map.

Usage:
  python3 tello_dongle_setup.py          # interactive capture, then offer to commit
  python3 tello_dongle_setup.py --list   # print the current registry and exit

The built-in laptop radio (wlp*/wlo*) is ignored: only USB dongles are offered.
This script is pure stdlib + nmcli, so it runs without the Isaac venv.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

# drone_deployment_infra/ root (core/ is one level under it); config is centralized there.
DD_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = DD_ROOT / "config" / "dongles.json"
# Each dongle's UDP video port is this base plus its number (17003 for #3).
PORT_BASE = 17000
TELLO_PREFIX = "TELLO"


def _nmcli(args, timeout=15):
    return subprocess.run(
        ["nmcli", *args], text=True, capture_output=True, timeout=timeout, check=False
    )


def mac_from_iface(iface):
    """58:d8:12:5e:da:77 from a wlx58d8125eda77 interface name, else ''."""
    if iface.startswith("wlx") and len(iface) == 15:
        hex_mac = iface[3:]
        return ":".join(hex_mac[i : i + 2] for i in range(0, 12, 2))
    return ""


def mac_of(iface):
    """MAC straight from sysfs, falling back to the one encoded in the name."""
    try:
        return Path(f"/sys/class/net/{iface}/address").read_text().strip()
    except OSError:
        return mac_from_iface(iface)


def is_wifi(iface):
    return Path(f"/sys/class/net/{iface}/wireless").is_dir()


def is_usb(iface):
    """True when the interface is a USB device (so, a dongle, not the built-in radio)."""
    try:
        target = os.path.realpath(f"/sys/class/net/{iface}/device")
    except OSError:
        return False
    return "usb" in target


def is_dongle(iface):
    # Predictable naming gives USB WiFi sticks a wlx<mac> name; trust that too
    # in case sysfs does not expose the usb path on some adapters.
    return is_wifi(iface) and (is_usb(iface) or iface.startswith("wlx"))


def device_info(iface):
    """Return (ssid, ip) the dongle is currently associated with, '' when none."""
    ssid = ""
    result = _nmcli(["-t", "-f", "ACTIVE,SSID", "device", "wifi", "list", "ifname", iface])
    for line in result.stdout.splitlines():
        active, _, name = line.partition(":")
        if active == "yes":
            ssid = name
            break
    result = _nmcli(["-t", "-f", "GENERAL.CONNECTION,IP4.ADDRESS", "device", "show", iface])
    conn = ""
    ip = ""
    for line in result.stdout.splitlines():
        key, _, value = line.partition(":")
        if key == "GENERAL.CONNECTION" and value not in ("", "--"):
            conn = value
        elif key.startswith("IP4.ADDRESS"):
            ip = value.split("/", 1)[0]
    # The joined SSID is the ground truth; the NM profile name is the backup.
    return ssid or conn, ip


def detect_dongles():
    """List plugged-in USB WiFi dongles as dicts, drone-connected ones first."""
    found = []
    try:
        ifaces = sorted(os.listdir("/sys/class/net"))
    except OSError:
        ifaces = []
    for iface in ifaces:
        if not is_dongle(iface):
            continue
        ssid, ip = device_info(iface)
        found.append(
            {
                "iface": iface,
                "mac": mac_of(iface),
                "ssid": ssid,
                "ip": ip,
                "on_tello": ssid.upper().startswith(TELLO_PREFIX),
            }
        )
    found.sort(key=lambda d: (not d["on_tello"], d["iface"]))
    return found


def load_registry(path):
    if not path.is_file():
        return {"description": "USB WiFi dongle -> Tello map. See tello_dongle_setup.py.", "dongles": []}
    return json.loads(path.read_text())


def load_fleet(names, path=None):
    """Return the registry entries flying the given drone names, in that order.

    Flight scripts call this instead of hardcoding interface names, so the
    dongle -> drone map lives only in dongles.json. Raises SystemExit with a
    clear message when a name is unmapped or its dongle has no interface.
    """
    registry = load_registry(Path(path) if path else REGISTRY_PATH)
    by_drone = {d.get("drone"): d for d in registry["dongles"] if d.get("drone")}
    fleet = []
    for name in names:
        entry = by_drone.get(name)
        if entry is None:
            raise SystemExit(
                f"{name} has no dongle in {path or REGISTRY_PATH}. "
                f"Run tello_dongle_setup.py and set a dongle's drone to {name}."
            )
        if not entry.get("iface"):
            raise SystemExit(f"{name} dongle has no interface recorded in {path or REGISTRY_PATH}")
        fleet.append(entry)
    return fleet


def present_ifaces():
    """USB WiFi dongle interfaces plugged in right now."""
    try:
        return {name for name in os.listdir("/sys/class/net") if is_dongle(name)}
    except OSError:
        return set()


def present_fleet(path=None):
    """Registry entries whose stick is plugged in now and maps to a drone.

    This is the fleet to fly: whichever numbered dongles are physically
    present, in dongle-number order. Any combination works -- just #1, or
    #1 and #2, or all four -- because the fleet is read from the hardware,
    not hardcoded. Empty when nothing mapped is plugged in.
    """
    here = present_ifaces()
    registry = load_registry(Path(path) if path else REGISTRY_PATH)
    fleet = [d for d in registry["dongles"] if d.get("drone") and d["iface"] in here]
    fleet.sort(key=lambda d: (d.get("number") is None, d.get("number")))
    return fleet


def save_registry(registry, path):
    registry["dongles"].sort(key=lambda d: (d.get("number") is None, d.get("number"), d["iface"]))
    registry["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(registry, indent=2) + "\n")


def upsert(registry, entry):
    """Replace the entry with the same iface, else append. Warn on number clashes."""
    dongles = registry["dongles"]
    for other in dongles:
        if other["iface"] != entry["iface"] and other.get("number") == entry.get("number"):
            print(f"  warning: number {entry['number']} is also on {other['iface']} ({other.get('drone')})")
    for i, other in enumerate(dongles):
        if other["iface"] == entry["iface"]:
            dongles[i] = entry
            return "updated"
    dongles.append(entry)
    return "added"


def _ask(prompt, default=""):
    suffix = f" [{default}]" if default else ""
    try:
        reply = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return reply or default


def _trailing_number(ssid):
    digits = ""
    for char in reversed(ssid):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return digits


def _existing_drone(registry, iface):
    for d in registry["dongles"]:
        if d["iface"] == iface:
            return d.get("drone", "")
    return ""


def capture_one(registry, port_base):
    """Pick a plugged-in dongle and record it. Return True when one was saved.

    A dongle already joined to a Tello auto-fills the drone name; one that is
    not can still be numbered now, with its drone typed in (or left blank).
    """
    dongles = detect_dongles()
    if not dongles:
        print("  no USB WiFi dongle detected. Plug one in (the built-in radio is ignored).")
        return False
    # Prefer the dongle sitting on a Tello, but allow numbering an idle one too.
    if len(dongles) == 1:
        chosen = dongles[0]
    else:
        print("  more than one dongle plugged in; pick one:")
        for i, d in enumerate(dongles):
            tag = "  <- on a Tello" if d["on_tello"] else ""
            print(f"    [{i}] {d['iface']}  mac {d['mac']}  ssid {d['ssid'] or '(none)'}{tag}")
        pick = _ask("  which", "0")
        if not pick.isdigit() or int(pick) >= len(dongles):
            print("  skipped.")
            return False
        chosen = dongles[int(pick)]

    if not chosen["on_tello"]:
        print(f"  note: {chosen['iface']} is not joined to a Tello (ssid {chosen['ssid'] or 'none'}).")
        print("  Numbering it anyway; connect it to its drone and re-capture to fill the drone in.")
    elif chosen["ip"] and not chosen["ip"].startswith("192.168.10."):
        print(f"  note: {chosen['iface']} ip is {chosen['ip']}, not a 192.168.10.x Tello address.")
    print(
        f"  dongle {chosen['iface']}  mac {chosen['mac']}"
        f"  ssid {chosen['ssid'] or '(none)'}  ip {chosen['ip'] or '(none)'}"
    )
    default_num = _trailing_number(chosen["ssid"])
    number_raw = _ask("  dongle number", default_num)
    number = int(number_raw) if number_raw.isdigit() else None
    drone = _ask("  drone name", chosen["ssid"] or _existing_drone(registry, chosen["iface"]))
    note = _ask("  note (optional)", "")
    entry = {
        "number": number,
        "iface": chosen["iface"],
        "mac": chosen["mac"],
        "drone": drone,
        "ssid": chosen["ssid"],
        "video_port": port_base + number if number is not None else None,
        "note": note,
    }
    action = upsert(registry, entry)
    print(f"  {action}: #{number} {drone} on {chosen['iface']} (video port {entry['video_port']})")
    return True


def next_number(registry):
    """Smallest positive integer not already used as a dongle number."""
    used = {d["number"] for d in registry["dongles"] if d.get("number") is not None}
    n = 1
    while n in used:
        n += 1
    return n


def auto_capture(registry, port_base):
    """Record any plugged-in dongle not yet known, and fill in drones that just joined.

    Returns a list of human-readable change lines (empty when nothing changed).
    Identity is the interface name (wlx<mac>), so re-plugging a known stick is a no-op.
    """
    known = {d["iface"]: d for d in registry["dongles"]}
    changes = []
    for dongle in detect_dongles():
        iface = dongle["iface"]
        existing = known.get(iface)
        if existing is None:
            number = next_number(registry)
            entry = {
                "number": number,
                "iface": iface,
                "mac": dongle["mac"],
                "drone": dongle["ssid"],
                "ssid": dongle["ssid"],
                "video_port": port_base + number,
                "note": "",
            }
            registry["dongles"].append(entry)
            known[iface] = entry
            drone = dongle["ssid"] or "(no drone yet)"
            changes.append(f"#{number} {drone} on {iface} (mac {dongle['mac']}, port {entry['video_port']})")
        elif dongle["on_tello"] and dongle["ssid"] and not existing.get("drone"):
            # The stick was numbered idle earlier; now it has joined a drone.
            existing["drone"] = dongle["ssid"]
            existing["ssid"] = dongle["ssid"]
            changes.append(f"#{existing['number']} drone filled in: {dongle['ssid']} on {iface}")
    return changes


def watch(registry, path, port_base, period):
    """Poll for dongles forever, recording each new one with the next number."""
    print(f"watching for dongles every {period}s. Plug them in one at a time; Ctrl-C to stop.")
    print("built-in radio is ignored; each new stick gets the next free number.\n")
    print_registry(registry)
    print()
    while True:
        changes = auto_capture(registry, port_base)
        if changes:
            save_registry(registry, path)
            for line in changes:
                print(f"recorded {line}", flush=True)
            print(f"  wrote {path} (not committed; run --list or commit when done)\n", flush=True)
        time.sleep(period)


def print_registry(registry):
    dongles = registry.get("dongles", [])
    if not dongles:
        print("no dongles recorded yet.")
        return
    print(f"{'num':>3}  {'drone':<10} {'iface':<16} {'mac':<18} {'port':>5}  note")
    for d in dongles:
        num = "?" if d.get("number") is None else d["number"]
        port = d.get("video_port") if d.get("video_port") is not None else "-"
        print(
            f"{num:>3}  {d.get('drone', ''):<10} {d['iface']:<16}"
            f" {d.get('mac', ''):<18} {port:>5}  {d.get('note', '')}"
        )


def maybe_commit(path):
    reply = _ask("commit and push dongles.json to GitHub? [y/N]", "n")
    if reply.lower() not in ("y", "yes"):
        print("left uncommitted. Commit it yourself when ready.")
        return
    repo = path.parent
    rel = path.name
    add = subprocess.run(["git", "-C", str(repo), "add", rel], capture_output=True, text=True)
    if add.returncode != 0:
        print(f"git add failed: {add.stderr.strip()}")
        return
    commit = subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Update dongle -> drone registry"],
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        print(commit.stdout.strip() or commit.stderr.strip())
        return
    print(commit.stdout.strip())
    push = subprocess.run(["git", "-C", str(repo), "push"], capture_output=True, text=True)
    if push.returncode != 0:
        print(f"push failed (commit is saved locally): {push.stderr.strip()}")
        return
    print("pushed to GitHub.")


def main():
    parser = argparse.ArgumentParser(description="Number USB WiFi dongles and map each to its Tello.")
    parser.add_argument(
        "--file", default=str(REGISTRY_PATH), help="registry path (default dongles.json beside this script)"
    )
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--watch", action="store_true", help="run in the background, auto-numbering each new dongle")
    parser.add_argument("--period", type=float, default=2.0, help="watch poll interval, seconds")
    parser.add_argument("--port-base", type=int, default=PORT_BASE, help="video port = base + dongle number")
    parser.add_argument("--no-git", action="store_true", help="never offer to commit")
    args = parser.parse_args()
    path = Path(args.file)
    registry = load_registry(path)

    if args.list:
        print_registry(registry)
        return

    if args.watch:
        watch(registry, path, args.port_base, args.period)
        return

    print("Connect ONE dongle to its drone at a time, then capture it here.")
    print("Press Enter to capture the connected dongle, or type 'done' to finish.\n")
    saved = 0
    while True:
        reply = _ask("ready (Enter = capture, 'done' = finish)", "")
        if reply.lower() in ("done", "q", "quit", "exit"):
            break
        if capture_one(registry, args.port_base):
            save_registry(registry, path)
            saved += 1
            print(f"  wrote {path}\n")
    print()
    print_registry(registry)
    if saved and not args.no_git:
        print()
        maybe_commit(path)


if __name__ == "__main__":
    main()

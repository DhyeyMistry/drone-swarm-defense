#!/home/jaimeromero/action-space/.venv/bin/python
"""Autoconnect whichever numbered dongles are plugged in, then fly them.

The fleet is read from the hardware: every stick present in dongles.json that
maps to a drone gets joined on its own interface, and the flight engine is
launched for exactly those drones. Plug in just #1, or #1 and #2, or any
combination of the TP-Link sticks -- the subroutine waits for those drones to
connect and then flies them. Home WiFi on the built-in radio is never touched.

  python tello_link.py           # connect present sticks and fly
  python tello_link.py --no_fly  # connect and stream only, no takeoff
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from tello_dongle_setup import present_fleet

DD_ROOT = Path(__file__).resolve().parent.parent   # drone_deployment_infra/
REPO = DD_ROOT.parent                               # repo root holds the shared .venv
PYTHON = REPO / ".venv" / "bin" / "python"
VIDEO = DD_ROOT / "core" / "tello_dual_video.py"
PERIOD_S = 3


def fleet_specs():
    """(ssid, iface, nm_profile) for each plugged-in, mapped dongle, right now."""
    return tuple((d["drone"], d["iface"], d["drone"]) for d in present_fleet())


def _nmcli(args, timeout=25):
    try:
        return subprocess.run(
            ["nmcli", *args],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # A hung nmcli (common on a flaky AP) must not crash the watch loop.
        return subprocess.CompletedProcess(args, 1, "", "nmcli timed out")


def device_links():
    result = _nmcli(["-t", "-f", "DEVICE,STATE,CONNECTION", "device", "status"])
    links = {}
    for line in result.stdout.splitlines():
        parts = line.split(":")
        if len(parts) >= 3:
            links[parts[0]] = (parts[1], parts[2])
    return links


def ssid_visible(iface, ssid):
    _nmcli(["device", "wifi", "rescan", "ifname", iface], timeout=15)
    time.sleep(3)
    result = _nmcli(["-t", "-f", "SSID", "device", "wifi", "list", "ifname", iface])
    if any(line.split(":")[0] == ssid for line in result.stdout.splitlines()):
        return True
    # The built-in radio often hears the drone when the USB stick's scan is stale.
    # Do not join on the built-in radio. A sighting there only means the AP is up.
    cached = _nmcli(["-t", "-f", "SSID", "device", "wifi", "list"])
    return any(line.split(":")[0] == ssid for line in cached.stdout.splitlines())


def on_tello_lan(iface):
    """True when this dongle already holds a 192.168.10.x address from a Tello."""
    result = _nmcli(["-t", "-f", "IP4.ADDRESS", "device", "show", iface])
    return any("192.168.10." in line for line in result.stdout.splitlines() if line.startswith("IP4.ADDRESS"))


def connect_drone(links, spec):
    """Join one Tello on its USB dongle. True when that link is up.

    If the dongle already has the Tello LAN, leave it completely alone -- a
    rescan on a flying interface drops the link. Only a dongle that has lost
    its address gets rescanned and rejoined, which is the reconnect path.
    """
    ssid, iface, profile = spec
    if on_tello_lan(iface):
        return True
    state, connection = links.get(iface, ("missing", ""))
    if "connecting" in state:
        print(f"{ssid} still joining", flush=True)
        return False
    # Try to associate straight away -- fastest reconnect when the AP is known.
    print(f"{ssid} joining {iface}", flush=True)
    result = _nmcli(["connection", "up", profile, "ifname", iface], timeout=20)
    if result.returncode == 0:
        print(f"{ssid} connected", flush=True)
        return True
    # Fall back to a rescan in case the AP left NetworkManager's cache, then retry.
    if ssid_visible(iface, ssid):
        result = _nmcli(["connection", "up", profile, "ifname", iface], timeout=20)
        if result.returncode == 0:
            print(f"{ssid} connected", flush=True)
            return True
    err = (result.stderr or result.stdout).strip().splitlines()
    print(f"{ssid} join failed: {err[-1] if err else result.returncode}", flush=True)
    return False


def reap_orphan_engines():
    """Kill any flight engine left over from a previous run (they outlive Ctrl-C)."""
    subprocess.run(["pkill", "-f", "tello_dual_video.py"], capture_output=True)


def start_engine(names, play):
    args = [str(PYTHON), "-u", str(VIDEO)]
    if play:
        args.append("--no_fly")
    args.extend(names)
    print(f"start flight engine for {' '.join(names)}{' (play, no takeoff)' if play else ''}", flush=True)
    return subprocess.Popen(args, cwd=DD_ROOT, start_new_session=True)


def main():
    play = "--no_fly" in sys.argv[1:]
    print("link watch every", PERIOD_S, "s", flush=True)
    reap_orphan_engines()  # clear any engine orphaned by a previous run
    fleet = []  # frozen at the first cycle that sees a mapped stick
    engine = None
    cycle = 0
    try:
        while True:
            started = time.time()
            specs = fleet or fleet_specs()
            if not specs:
                print("no mapped dongle plugged in; waiting", flush=True)
                time.sleep(max(0, PERIOD_S - (time.time() - started)))
                continue
            if not fleet:
                fleet = specs
                print("fleet:", " ".join(ssid for ssid, _iface, _profile in fleet), flush=True)
            # Own exactly one engine: (re)start it if it isn't running.
            if engine is None or engine.poll() is not None:
                engine = start_engine([ssid for ssid, _iface, _profile in fleet], play)
            connected = []
            for spec in fleet:
                connected.append(connect_drone(device_links(), spec))
            cycle += 1
            if all(connected) and (cycle == 1 or cycle % 5 == 0):
                print("all drones connected:", " ".join(s[0] for s in fleet), flush=True)
            time.sleep(max(0, PERIOD_S - (time.time() - started)))
    finally:
        if engine is not None and engine.poll() is None:
            try:
                os.killpg(os.getpgid(engine.pid), signal.SIGTERM)
                print("stopped flight engine", flush=True)
            except OSError:
                engine.terminate()


if __name__ == "__main__":
    main()

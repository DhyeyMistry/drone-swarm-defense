#!/home/jaimeromero/action-space/.venv/bin/python
"""Flight link test -- steady rc, passive connection detection.

Takes off and streams rc at a fixed rate, the way the phone app streams
continuous control. It does NOT send a blocking "command" probe in the control
loop (that stalls the stream ~0.4 s each time and is what caused the ~5 s drops);
instead it detects the link passively from the drone's state broadcast on UDP
8890. With --speed it flies a forward/back pattern to test the link under motion.
Logs when the link drops and how long any reconnect takes.

  ./tello_hover_test.py                       # hover, 15 Hz, 60 s
  ./tello_hover_test.py --rate 20             # hover at 20 Hz
  ./tello_hover_test.py --rate 20 --speed 30 --leg 3   # forward/back legs

Writes outputs/hover_<stamp>.jsonl (t, connected, state_age).
"""

import fcntl
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

DD_ROOT = Path(__file__).resolve().parent.parent      # drone_deployment_infra/
sys.path.insert(0, str(DD_ROOT / "core"))             # for tello_dongle_setup

from tello_dongle_setup import load_fleet, present_fleet

TELLO = ("192.168.10.1", 8889)
SO_BINDTODEVICE = 25
OUT_DIR = DD_ROOT / "outputs"
STATE = {"t": 0.0}  # wall-clock of the last state packet from the drone


def dev_socket(iface, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, iface.encode())
    sock.bind(("0.0.0.0", port))
    return sock


def ask(sock, cmd, timeout):
    """Send an SDK command and read its reply, draining stale replies first."""
    sock.setblocking(False)
    try:
        while True:
            sock.recvfrom(256)
    except (BlockingIOError, OSError):
        pass
    sock.settimeout(timeout)
    try:
        sock.sendto(cmd.encode(), TELLO)
        data, _ = sock.recvfrom(256)
        # Some firmware replies "OK", some "ok"; fold case (callers test "ok").
        return data.decode(errors="replace").strip().lower()
    except (socket.timeout, OSError):
        return None


def state_listener(sock, stop):
    """Stamp STATE['t'] on every state packet the drone broadcasts (passive)."""
    sock.settimeout(0.5)
    while not stop.is_set():
        try:
            sock.recvfrom(2048)
            STATE["t"] = time.time()
        except socket.timeout:
            continue
        except OSError:
            return


def arg_value(flag, default, cast):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def positional_args(argv, value_flags):
    """Drone names only -- skip flags and the values that follow value-taking flags."""
    out = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in value_flags:
            skip = True
            continue
        if a.startswith("-"):
            continue
        out.append(a)
    return out


def main():
    args = positional_args(sys.argv[1:], {"--seconds", "--rate", "--speed", "--leg"})
    seconds = arg_value("--seconds", 60.0, float)
    rate = arg_value("--rate", 15.0, float)
    speed = arg_value("--speed", 0.0, float)   # >0 flies forward/back at this rc; 0 hovers
    leg = arg_value("--leg", 3.0, float)        # seconds per forward or back leg
    period = 1.0 / rate
    fleet = load_fleet(args) if args else present_fleet()
    if not fleet:
        raise SystemExit("no drone found: plug a numbered stick in or pass a drone name")
    name, iface = fleet[0]["drone"], fleet[0]["iface"]

    subprocess.run(["nmcli", "connection", "up", name, "ifname", iface], capture_output=True, text=True, timeout=30)
    cmd = dev_socket(iface, 8889)
    state = dev_socket(iface, 8890)
    stop = threading.Event()
    threading.Thread(target=state_listener, args=(state, stop), daemon=True).start()

    for _ in range(12):
        if ask(cmd, "command", 1.0) == "ok":
            break
        time.sleep(0.4)
    motion = f"forward/back at rc {speed:.0f}, {leg:.0f}s legs" if speed > 0 else "hover in place"
    print(f"link test on {name}: rc {rate:.0f} Hz, passive state, {motion}, {seconds:.0f}s.", flush=True)
    print(f"battery {ask(cmd, 'battery?', 1.0)}%. Be ready to catch it. Ctrl-C to land early.\n", flush=True)
    result = ask(cmd, "takeoff", 15)
    print(f"takeoff {result}", flush=True)
    if result != "ok":
        print("takeoff failed; aborting", flush=True)
        stop.set()
        return

    OUT_DIR.mkdir(exist_ok=True)
    log = OUT_DIR / ("hover_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    start = time.time()
    STATE["t"] = start
    connected = True
    down_since = None
    events = []
    handle = open(log, "w", encoding="utf-8")
    try:
        while time.time() - start < seconds:
            now = time.time()
            t = now - start
            # steady control stream, never blocks: hover, or forward/back legs
            forward = 0
            if speed > 0:
                forward = int(speed) if int(t // leg) % 2 == 0 else -int(speed)
            cmd.sendto(f"rc 0 {forward} 0 0".encode(), TELLO)
            age = now - STATE["t"]
            up = age < 1.0
            if up and not connected:
                dur = now - down_since
                print(f"[{t:6.1f}s] RECONNECTED after {dur:4.1f}s", flush=True)
                events.append({"t": round(t, 1), "event": "reconnect", "down_s": round(dur, 1)})
                connected = True
            elif not up and connected:
                down_since = now - age
                print(f"[{t:6.1f}s] DISCONNECTED (no state for {age:.1f}s)", flush=True)
                events.append({"t": round(t, 1), "event": "disconnect"})
                connected = False
            handle.write(json.dumps({"t": round(t, 2), "connected": up, "state_age": round(age, 2)}) + "\n")
            if int(t) != int(t - period) and connected:
                moving = "fwd " if forward > 0 else "back" if forward < 0 else "hover"
                print(f"[{t:6.1f}s] UP  {moving}", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nlanding...", flush=True)
        for _ in range(8):
            if ask(cmd, "land", 2) == "ok":
                break
            time.sleep(0.3)
        stop.set()
        handle.close()
        drops = [e for e in events if e["event"] == "disconnect"]
        recons = [e for e in events if e["event"] == "reconnect"]
        print("\n--- hover link summary ---", flush=True)
        print(f"rc {rate:.0f} Hz, flew ~{time.time() - start:.0f}s, "
              f"{len(drops)} disconnect(s), {len(recons)} reconnect(s)")
        if drops:
            print(f"first disconnect at t={drops[0]['t']}s")
        for e in recons:
            print(f"  reconnect took {e['down_s']}s")
        print(f"log: {log}")


if __name__ == "__main__":
    main()

#!/home/jaimeromero/action-space/.venv/bin/python
"""Diagnose the video path for the quick-connect drone(s). NO takeoff.

A flight test that logs `decode open failed: Immediate exit requested` got no video in
the ~8 s PyAV open window. This isolates where it breaks: it connects each quick-connect
drone (config/fleet.yaml), sends `streamon`, and counts the raw H.264 bytes the drone
sends to UDP 11111 on that drone's stick. Then it tries to decode a few frames.

Read the result like this:
  * bytes == 0            -> the drone isn't streaming to us: streamon failed, or the
                             video isn't reaching this stick (routing / driver).
  * bytes > 0, frames 0   -> video arrives but PyAV can't decode it in time: a keyframe
                             /timing problem, which the flight loop's retry handles.
  * bytes > 0, frames > 0 -> the video path is fine end to end.

  ./tello_video_check.py [--seconds 8]
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

DD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DD_ROOT / "core"))

from test_link import bring_up, insist, present_quick_connect  # noqa: E402
from tello_dual_video import TELLO, decode, dev_socket, forward, has_tello_lan, sdk  # noqa: E402


def raw_exchange(cmd: socket.socket, command: str, timeout: float = 2.0) -> None:
    """Send one SDK command and print the RAW reply bytes (hex + decoded), draining
    first. Shows exactly what the drone put on 8889 -- the clue to any desync."""
    cmd.setblocking(False)
    drained = 0
    try:
        while True:
            cmd.recvfrom(1024)
            drained += 1
    except (BlockingIOError, OSError):
        pass
    cmd.settimeout(timeout)
    cmd.sendto(command.encode(), (TELLO, 8889))
    try:
        data, addr = cmd.recvfrom(1024)
        text = data.decode("utf-8", errors="replace")
        print(f"  [raw] {command!r} <- {data.hex()}  decoded={text!r}  from {addr[0]}"
              + (f"  (drained {drained})" if drained else ""), flush=True)
    except socket.timeout:
        print(f"  [raw] {command!r} <- (no reply in {timeout:.0f}s)"
              + (f"  (drained {drained})" if drained else ""), flush=True)


def count_bytes(iface: str, seconds: float) -> tuple[int, int]:
    """Raw bytes/packets arriving on this stick's UDP 11111 over `seconds`."""
    sock = dev_socket(iface, 11111)
    sock.settimeout(0.5)
    total = packets = 0
    end = time.time() + seconds
    while time.time() < end:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            continue
        total += len(data)
        packets += 1
    sock.close()
    return total, packets


def check(name: str, iface: str, video_port: int, seconds: float) -> None:
    print(f"\n== {name} ({iface})", flush=True)
    bring_up(name, iface)
    for _ in range(20):
        if has_tello_lan(iface):
            break
        time.sleep(0.5)
    if not has_tello_lan(iface):
        print(f"  NOT associated (no 192.168.10.x on {iface}); is the drone on?", flush=True)
        return

    cmd = dev_socket(iface, 8889)
    # Raw view first: see the exact bytes on 8889 for each command.
    raw_exchange(cmd, "command")
    raw_exchange(cmd, "streamon")
    raw_exchange(cmd, "streamoff")
    # Then the real sequence the flight uses, with retry-until-ok.
    print(f"  command  -> {insist(cmd, 'command', name, attempts=4, gap=0.4, timeout=2)}", flush=True)
    print(f"  streamon -> {insist(cmd, 'streamon', name, attempts=4, gap=0.4, timeout=2)}", flush=True)

    total, packets = count_bytes(iface, seconds)
    rate = total / seconds / 1024
    print(f"  UDP 11111: {total} bytes in {packets} packets over {seconds:.0f}s "
          f"({rate:.0f} KB/s)", flush=True)
    if total == 0:
        print("  -> drone is NOT streaming to this stick. streamon may have failed, or "
              "the video isn't routing to this interface.", flush=True)
        sdk(cmd, "streamoff", 1)
        cmd.close()
        return

    # Video is arriving -> see if PyAV decodes it the same way a flight test does.
    stop = threading.Event()
    frames: dict = {}
    stats = {name: {"session": "chk", "frames": 0, "drops": 0, "overflow": 0,
                    "tag_hit": 0, "tag_miss": 0, "file": ""}}
    video = dev_socket(iface, 11111)
    threading.Thread(target=forward, args=(video, video_port, stop), daemon=True).start()
    threading.Thread(target=decode, args=(name, video_port, frames, stats, stop), daemon=True).start()
    time.sleep(seconds + 2)
    f = frames.get(name)
    print(f"  PyAV decoded frames: {stats[name]['frames']}"
          + (f", last {f.shape[1]}x{f.shape[0]}" if f is not None else " (none)"), flush=True)
    stop.set()
    time.sleep(0.5)
    sdk(cmd, "streamoff", 1)
    cmd.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0, help="how long to sample video")
    args = parser.parse_args()

    present = present_quick_connect()
    if not present:
        raise SystemExit("no quick_connect drones present: set config/fleet.yaml and plug sticks in")
    for d in present:
        check(d["drone"], d["iface"], d["video_port"], args.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Bring up each defender on its own USB stick, independently -- no drone waits for another.

core/test_link.connect_all() waits for EVERY drone and aborts the whole run if one never
answers; for a defence that must launch at once, one slow or broken drone must not ground
the others. Here every defender gets its own thread: join its Wi-Fi, wait for 'command' ok,
then call on_ready(link) at once (defend.py takes it off right there, or reports PASS to
swarm_policies.py). Its video is brought up AFTER that and retried in the background, so a
failed video stream never delays or blocks a takeoff; a defender that never sees a tag
lands after defend.py's --lost_timeout_s.

Same plumbing as the repo (stick-pinned sockets, video forward, state listener) via
core/test_link.py and core/tello_dual_video.py; nothing in the repo is changed.
"""

from __future__ import annotations

import threading
import time

import test_link
from tello_dual_video import decode, dev_socket, forward, has_tello_lan, listen_state, next_session, sdk

VIDEO_RETRY_S = 2.0          # between streamon attempts while a defender has no video

ANSWER_TIMEOUT_S = 60.0      # give up on a drone that never answers 'command' by then
RENUDGE_S = 6.0              # re-run nmcli for a stick that hasn't joined its drone yet


def start_defenders(names: list[str], stop: threading.Event, on_ready) -> list[str]:
    """Start one connector thread per defender whose stick is plugged in; returns their names.

    on_ready(link) runs in that drone's thread as soon as it answers 'command'.
    """
    test_link.quick_connect_drones = lambda: list(names)        # OUR list, not fleet.yaml
    present = test_link.present_quick_connect()                 # prints "skip X: stick not plugged in"
    session = next_session()
    frames, heights, batteries, motion, stats = {}, {}, {}, {}, {}
    started = []
    for d in present:
        name, iface, port = d["drone"], d["iface"], d["video_port"]
        cmd = dev_socket(iface, 8889)
        video = dev_socket(iface, 11111)
        state = dev_socket(iface, 8890)
        link = {"t": 0.0}
        stats[name] = {"session": session, "frames": 0, "drops": 0, "overflow": 0,
                       "tag_hit": 0, "tag_miss": 0, "file": ""}
        threading.Thread(target=forward, args=(video, port, stop), daemon=True).start()
        threading.Thread(target=listen_state, args=(state, name, heights, batteries, motion, link, stop),
                         daemon=True).start()
        dl = test_link.DroneLink(name, iface, port, cmd, frames, heights, batteries, motion, link, stats, session)
        # One thread at a time on this drone's command socket: a streamon in flight must not
        # swallow the reply to a takeoff (that is what made takeoff/streamon "timeout").
        dl.sdk_lock = threading.Lock()
        threading.Thread(target=_connect, args=(dl, stop, on_ready), daemon=True, name=f"conn-{name}").start()
        started.append(name)
    return started


def _connect(dl, stop: threading.Event, on_ready) -> None:
    t0, last_up = time.time(), 0.0
    while not stop.is_set():
        now = time.time()
        if now - t0 > ANSWER_TIMEOUT_S:
            print(f"[{dl.name}] FAIL: no answer after {ANSWER_TIMEOUT_S:.0f} s (powered on? stick?) -> skipped",
                  flush=True)
            return
        if not has_tello_lan(dl.iface):
            if now - last_up > RENUDGE_S:
                test_link.bring_up(dl.name, dl.iface)
                last_up = time.time()
            stop.wait(0.3)
            continue
        if sdk(dl.cmd, "command", 1.0) == "ok":
            print(f"[{dl.name}] connected in {time.time() - t0:.1f} s", flush=True)
            on_ready(dl)
            break
        stop.wait(0.3)
    # Video, retried here so a failing stream never blocks takeoff or the control loop.
    while not stop.is_set():
        if dl.decoder is None or not dl.decoder.is_alive():
            with dl.sdk_lock:                    # one short attempt; takeoff waits at most ~2 s
                reply = "" if stop.is_set() else sdk(dl.cmd, "streamon", 2.0)
            if reply == "ok":
                dl.decoder = threading.Thread(target=_safe_decode, args=(dl, stop), daemon=True,
                                              name=f"video-{dl.name}")
                dl.decoder.start()
            elif reply:
                print(f"[{dl.name}] streamon -> {reply!r}, retrying", flush=True)
        stop.wait(VIDEO_RETRY_S)


def _safe_decode(dl, stop: threading.Event) -> None:
    """The repo's decode(), but a corrupt packet (InvalidDataError at stream start) ends just
    this attempt; the loop above starts a fresh one instead of the thread dying with a trace."""
    try:
        decode(dl.name, dl.video_port, dl.frames, dl.stats, stop)
    except Exception as exc:  # noqa: BLE001  (av errors)
        print(f"[{dl.name}] video decoder stopped ({type(exc).__name__}); restarting", flush=True)

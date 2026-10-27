"""Connect to a configured drone, verify its identity, and start/stop video.

Shared by the scripts so every one of them goes through the same identity gate.
Typical use::

    drone, local_ip = resolve_drone("tello_4")
    with verified_link(drone, local_ip) as link, video_stream(link, local_ip) as video:
        ...
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from tello_calibration.comms.identity import check_serial, check_wifi
from tello_calibration.comms.tello_link import CMD_PORT, TelloLink, TelloTimeoutError
from tello_calibration.utils.config import DroneConfig, DroneConfigError, load_drone_config
from tello_calibration.utils.net import TELLO_IP, TelloNetworkError, active_wifi, resolve_local_ip
from tello_calibration.video.stream import VideoReceiver

log = logging.getLogger("tello_calibration.session")

LOW_BATTERY_PCT = 20


class SessionError(RuntimeError):
    """Setup failed (config, network, no reply, no video). Scripts exit 1."""


class IdentityError(SessionError):
    """The drone is not verified as the configured one. Scripts exit 2."""


def report(step: str, ok: bool, detail: str = "") -> None:
    log.log(logging.INFO if ok else logging.ERROR, "[%s] %s %s", "PASS" if ok else "FAIL",
            step, detail)


def resolve_drone(name: str) -> tuple[DroneConfig, str]:
    """Load the drone's config entry and find our IP on its interface."""
    try:
        drone = load_drone_config(name)
    except DroneConfigError as exc:
        report("load drone config", False, str(exc))
        raise SessionError(str(exc)) from exc
    log.info("Drone %s: iface=%s ssid=%s bssid=%s serial=%s", drone.name, drone.iface,
             drone.ssid, drone.bssid, drone.serial)
    try:
        net = resolve_local_ip(iface=drone.iface, local_ip=drone.local_ip)
    except TelloNetworkError as exc:
        report("local IP on Tello subnet", False, str(exc))
        raise SessionError(str(exc)) from exc
    report("local IP on Tello subnet", True, f"{net.name} -> {net.ip}")
    return drone, net.ip


def query(link: TelloLink, cmd: str) -> str:
    """Run a read-only query; firmware that lacks it (e.g. sdk? on SDK 1.3) says so."""
    try:
        reply = link.send(cmd, timeout=2.0, retries=1)
    except TelloTimeoutError:
        return "<no reply>"
    if reply.startswith("unknown command") or reply == "error":
        return f"<not supported: {reply}>"
    return reply


def wait_for_state(link: TelloLink, timeout: float = 3.0) -> bool:
    assert link.state is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = link.state.latest()
        if snap.count >= 3 and "bat" in snap.values:
            report("state packets", True,
                   f"{snap.count} pkts, {len(snap.values)} fields, bat={snap.values['bat']:.0f}%")
            return True
        time.sleep(0.1)
    report("state packets", False, f"{link.state.latest().count} pkts within {timeout:.0f} s")
    return False


def _enter_sdk_and_check_serial(link: TelloLink, drone: DroneConfig) -> bool:
    try:
        reply = link.send("command", timeout=2.0, retries=3)
    except TelloTimeoutError as exc:
        report("enter SDK mode ('command')", False, str(exc))
        raise SessionError(str(exc)) from exc
    report("enter SDK mode ('command')", reply == "ok", f"reply={reply!r}")
    if reply != "ok":
        raise SessionError(f"'command' answered {reply!r}")

    info = {cmd: query(link, cmd) for cmd in ("battery?", "sdk?", "wifi?")}
    try:
        sn_reply: str | None = link.send("sn?", timeout=2.0, retries=1)
    except TelloTimeoutError:
        sn_reply = None
    log.info("Connection report:")
    for cmd, value in info.items():
        log.info("  %-9s %s", cmd, value)
    log.info("  %-9s %s", "sn?", sn_reply if sn_reply is not None else "<no reply>")
    if info["battery?"].isdigit() and int(info["battery?"]) < LOW_BATTERY_PCT:
        log.warning("Battery low: %s%% (< %d%%)", info["battery?"], LOW_BATTERY_PCT)

    ok, detail = check_serial(drone, sn_reply)
    report("identity: serial", ok, detail)
    return ok


@contextmanager
def verified_link(
    drone: DroneConfig,
    local_ip: str,
    drone_ip: str = TELLO_IP,
    drone_cmd_port: int = CMD_PORT,
    cmd_port: int = CMD_PORT,
) -> Iterator[TelloLink]:
    """Yield a TelloLink only once Wi-Fi SSID/BSSID and sn? match the config.

    Starts the state listener and keepalive; on exit stops them and closes sockets.
    """
    ok, detail = check_wifi(drone, active_wifi(drone.iface))
    report("identity: Wi-Fi SSID/BSSID", ok, detail)
    if not ok:  # nothing has been sent to the drone yet
        log.error("Refusing to continue: %s is not verified as %s", drone_ip, drone.name)
        raise IdentityError(detail)

    with TelloLink(local_ip, drone_ip=drone_ip, cmd_port=cmd_port,
                   drone_cmd_port=drone_cmd_port) as link:
        if not _enter_sdk_and_check_serial(link, drone):
            log.error("Refusing to continue: %s is not verified as %s", drone_ip, drone.name)
            raise IdentityError(f"serial check failed for {drone.name}")
        link.start_state_listener()
        wait_for_state(link)
        link.start_keepalive(interval=5.0)
        yield link


@contextmanager
def video_stream(
    link: TelloLink, local_ip: str, first_frame_timeout: float = 5.0
) -> Iterator[VideoReceiver]:
    """streamon, wait for the first decoded frame, yield the receiver; always streamoff."""
    video = VideoReceiver(local_ip)
    video.start()
    first_id, t_start = 0, time.monotonic()
    try:
        reply = link.send("streamon", timeout=2.0, retries=2)
        report("streamon", reply == "ok", f"reply={reply!r}")
        deadline = time.monotonic() + first_frame_timeout
        while video.get_latest()[0] is None and time.monotonic() < deadline:
            time.sleep(0.05)
        frame, first_id, _ = video.get_latest()
        if frame is None:
            detail = f"{video.bytes_received} bytes received, none decoded in " \
                     f"{first_frame_timeout:.0f} s"
            report("first video frame", False, detail)
            raise SessionError(detail)
        report("first video frame", True, f"{frame.shape[1]}x{frame.shape[0]}")
        t_start = time.monotonic()
        yield video
    finally:
        log.info("Stopping video...")
        n = video.get_latest()[1] - first_id
        elapsed = time.monotonic() - t_start
        if n > 0:
            log.info("Decoded %d frames in %.1f s (%.1f fps)", n, elapsed, n / elapsed)
        try:
            link.send("streamoff", timeout=1.0, retries=1)
        except TelloTimeoutError:
            log.warning("No reply to streamoff")
        video.stop()
        log.info("Video stopped (%d decode errors tolerated)", video.decode_errors)

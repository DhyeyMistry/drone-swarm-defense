"""UDP command client for the Tello (SDK 1.3) with a safety allowlist, plus state listener.

SAFETY: only the commands in ALLOWED_COMMANDS can ever be sent. Everything else
(takeoff, land, rc, motoron, emergency, ...) raises CommandNotAllowedError before any
socket call is made.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field

from tello_calibration.utils.net import TELLO_IP

log = logging.getLogger(__name__)

CMD_PORT = 8889
STATE_PORT = 8890

# Exact strings only: no prefixes, no case folding, no whitespace tolerance.
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {"command", "streamon", "streamoff", "battery?", "sdk?", "sn?", "wifi?", "time?"}
)


class CommandNotAllowedError(PermissionError):
    """Raised for any command outside the allowlist. Nothing was sent."""


class TelloTimeoutError(TimeoutError):
    """The drone did not reply in time."""


def check_allowed(cmd: str) -> None:
    if cmd not in ALLOWED_COMMANDS:
        raise CommandNotAllowedError(
            f"Command {cmd!r} is not on the allowlist {sorted(ALLOWED_COMMANDS)}; not sent."
        )


@dataclass
class StateSnapshot:
    values: dict[str, float | str] = field(default_factory=dict)
    t_recv: float | None = None  # time.monotonic() of the last packet
    count: int = 0


def parse_state(text: str) -> dict[str, float | str]:
    """Parse 'pitch:0;roll:0;...;\\r\\n' into a dict, numbers as float."""
    values: dict[str, float | str] = {}
    for item in text.strip().split(";"):
        key, sep, raw = item.partition(":")
        if not sep:
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw
    return values


class StateListener:
    """Thread that receives state packets on <local_ip>:8890 and keeps the latest."""

    def __init__(self, local_ip: str, port: int = STATE_PORT) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((local_ip, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._snapshot = StateSnapshot()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tello-state", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> StateSnapshot:
        with self._lock:
            return StateSnapshot(dict(self._snapshot.values), self._snapshot.t_recv,
                                 self._snapshot.count)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed
            values = parse_state(data.decode("ascii", errors="replace"))
            with self._lock:
                self._snapshot.values = values
                self._snapshot.t_recv = time.monotonic()
                self._snapshot.count += 1


class TelloLink:
    """Command/response client bound to our IP on the drone's network."""

    def __init__(
        self,
        local_ip: str,
        drone_ip: str = TELLO_IP,
        cmd_port: int = CMD_PORT,
        drone_cmd_port: int = CMD_PORT,
    ) -> None:
        self.local_ip = local_ip
        self._drone_addr = (drone_ip, drone_cmd_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((local_ip, cmd_port))
        self._send_lock = threading.Lock()  # one request/reply in flight at a time
        self._stop = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self.state: StateListener | None = None

    def __enter__(self) -> TelloLink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def send(self, cmd: str, timeout: float = 2.0, retries: int = 2) -> str:
        """Send an allowlisted command and return the drone's reply text."""
        check_allowed(cmd)  # must stay the first line: nothing is sent if this raises
        with self._send_lock:
            for attempt in range(1 + retries):
                self._drain()
                log.debug("-> %s (attempt %d)", cmd, attempt + 1)
                self._sock.sendto(cmd.encode("ascii"), self._drone_addr)
                reply = self._wait_reply(timeout)
                if reply is not None:
                    log.debug("<- %s: %s", cmd, reply)
                    return reply
                log.debug("timeout waiting for reply to %r", cmd)
        raise TelloTimeoutError(f"No reply to {cmd!r} after {1 + retries} attempt(s)")

    def takeoff(self, timeout: float = 7.0) -> str:
        """Take off and return the drone's reply."""
        return self._send_flight_command("takeoff", timeout=timeout)

    def land(self, timeout: float = 7.0) -> str:
        """Land and return the drone's reply."""
        return self._send_flight_command("land", timeout=timeout)

    def send_rc(
        self,
        left_right: int,
        forward_back: int,
        up_down: int,
        yaw: int,
    ) -> None:
        """Send a bounded RC velocity command.

        Values are deliberately limited to +/-30 for this project even though
        the Tello SDK permits a wider range.
        """

        values = {
            "left_right": left_right,
            "forward_back": forward_back,
            "up_down": up_down,
            "yaw": yaw,
        }

        for name, value in values.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")

            if not -30 <= value <= 30:
                raise ValueError(
                    f"{name}={value} outside project safety limit [-30, 30]"
                )

        cmd = f"rc {left_right} {forward_back} {up_down} {yaw}"

        with self._send_lock:
            self._drain()
            log.debug("-> %s", cmd)
            self._sock.sendto(cmd.encode("ascii"), self._drone_addr)

    def hover(self) -> None:
        """Command zero velocity."""
        self.send_rc(0, 0, 0, 0)

    def _send_flight_command(
        self,
        cmd: str,
        timeout: float = 7.0,
        retries: int = 0,
    ) -> str:
        """Send one explicitly supported flight command."""

        if cmd not in {"takeoff", "land"}:
            raise CommandNotAllowedError(
                f"Flight command {cmd!r} is not explicitly permitted."
            )

        with self._send_lock:
            for attempt in range(1 + retries):
                self._drain()
                log.info("-> %s", cmd)
                self._sock.sendto(cmd.encode("ascii"), self._drone_addr)

                reply = self._wait_reply(timeout)

                if reply is not None:
                    log.info("<- %s: %s", cmd, reply)
                    return reply

        raise TelloTimeoutError(
            f"No reply to {cmd!r} after {1 + retries} attempt(s)"
        )
    
    def start_state_listener(self) -> StateListener:
        self.state = StateListener(self.local_ip)
        self.state.start()
        return self.state

    def start_keepalive(self, interval: float = 5.0) -> None:
        """Send 'battery?' periodically so the drone keeps the SDK session alive."""
        self._keepalive_thread = threading.Thread(
            target=self._keepalive, args=(interval,), name="tello-keepalive", daemon=True
        )
        self._keepalive_thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=5.0)
        if self.state is not None:
            self.state.stop()
        self._sock.close()

    def _keepalive(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.send("battery?", timeout=1.0, retries=0)
            except TelloTimeoutError:
                log.warning("Keepalive: no reply to battery?")
            except OSError:
                break  # socket closed during shutdown

    def _wait_reply(self, timeout: float) -> str | None:
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            self._sock.settimeout(remaining)
            try:
                data, addr = self._sock.recvfrom(1024)
            except socket.timeout:
                return None
            if addr[0] == self._drone_addr[0]:
                # Fold case: some firmware replies "OK", some "ok".
                return data.decode("utf-8", errors="replace").strip().lower()
        return None

    def _drain(self) -> None:
        """Discard late replies to earlier commands so they are not mistaken for ours."""
        self._sock.setblocking(False)
        try:
            while True:
                data, _ = self._sock.recvfrom(1024)
                log.debug("discarded stale reply: %r", data)
        except BlockingIOError:
            pass
        finally:
            self._sock.setblocking(True)

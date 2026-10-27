"""Drive a real Tello EDU from a policy trained in simulation.

The sim action vector is [vx, vy, vz, yaw_rate] in [-1, 1], which maps 1:1 onto the SDK
``rc a b c d`` channels. This module keeps the mapping, rate and limits identical to
``action_space_kit.control.tello_command`` so behaviour carries over from sim to hardware.

Safety, in order of priority:
  * geofence -- commands are zeroed and the drone told to hover outside the allowed box
  * speed cap -- channels clamped below the configured fraction of full scale
  * dead-man timeout -- no fresh action within ``command_timeout`` triggers a land
  * any exception -- land, then disconnect

Use ``--dry-run`` to print the exact commands without connecting to a drone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BridgeCfg:
    """Settings for driving a real drone."""

    command_hz: float = 10.0
    """Rate at which rc commands are sent. The SDK tolerates up to ~20 Hz."""

    speed_cap: float = 0.5
    """Fraction of full scale allowed on each channel (0.5 = +/-50 on rc)."""

    geofence_xyz: tuple[float, float, float] = (2.0, 2.0, 2.0)
    """[m] half-extent of the allowed box, measured from takeoff point."""

    command_timeout: float = 1.0
    """[s] without a new action before the drone is landed."""

    takeoff: bool = True
    """Send ``takeoff`` on start and ``land`` on stop."""

    drone_ips: list[str] = field(default_factory=list)
    """Tello EDU station-mode IPs for swarm flights. Empty = single drone on its own AP."""


class TelloBridge:
    """Sends policy actions to one Tello as ``rc`` commands.

    Args:
        cfg: bridge settings.
        dry_run: if True, print commands instead of connecting to hardware.
        ip: address of the drone; only used for station-mode swarms.
    """

    def __init__(self, cfg: BridgeCfg, dry_run: bool = False, ip: str | None = None):
        self.cfg = cfg
        self.dry_run = dry_run
        self.ip = ip
        self._drone = None
        self._last_command_time = 0.0
        self._position = [0.0, 0.0, 0.0]  # dead-reckoned from commands, for the geofence

    # -- connection -------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the drone (no-op when dry running)."""
        if self.dry_run:
            print("[dry-run] connect%s" % (f" to {self.ip}" if self.ip else ""))
            return
        from djitellopy import Tello  # imported lazily so dry runs need no hardware deps

        self._drone = Tello(host=self.ip) if self.ip else Tello()
        self._drone.connect()
        print(f"[bridge] connected, battery {self._drone.get_battery()}%")
        if self.cfg.takeoff:
            self._drone.takeoff()

    def close(self) -> None:
        """Land and disconnect. Safe to call more than once."""
        if self.dry_run:
            print("[dry-run] land + disconnect")
            return
        if self._drone is not None:
            try:
                if self.cfg.takeoff:
                    self._drone.land()
            finally:
                self._drone.end()
                self._drone = None

    # -- control ----------------------------------------------------------------

    def to_rc(self, action) -> list[int]:
        """Map a 4-vector action in [-1, 1] to ``rc a b c d`` integers in [-100, 100]."""
        vx, vy, vz, yaw = (float(max(-1.0, min(1.0, a))) for a in action)
        cap = 100.0 * self.cfg.speed_cap
        # a = left/right (vy), b = forward/back (vx), c = up/down (vz), d = yaw
        return [int(round(max(-cap, min(cap, v * 100.0)))) for v in (vy, vx, vz, yaw)]

    def _geofence_ok(self, dt: float, rc: list[int]) -> bool:
        """Dead-reckon the position from commands and check it against the box."""
        # rc channels are roughly +/-1 m/s at full scale
        self._position[0] += rc[1] / 100.0 * dt
        self._position[1] += rc[0] / 100.0 * dt
        self._position[2] += rc[2] / 100.0 * dt
        return all(abs(p) <= limit for p, limit in zip(self._position, self.cfg.geofence_xyz))

    def send(self, action) -> list[int]:
        """Send one action. Returns the rc channels actually sent."""
        now = time.time()
        dt = now - self._last_command_time if self._last_command_time else 1.0 / self.cfg.command_hz
        rc = self.to_rc(action)
        if not self._geofence_ok(dt, rc):
            print("[bridge] geofence reached -- holding position")
            rc = [0, 0, 0, 0]
        self._last_command_time = now
        if self.dry_run:
            print("[dry-run] rc %d %d %d %d" % tuple(rc))
        else:
            self._drone.send_rc_control(*rc)
        return rc

    def run_policy(self, policy, get_observation, duration: float) -> None:
        """Run a policy for ``duration`` seconds at the configured rate.

        Args:
            policy: callable mapping an observation to a 4-vector action.
            get_observation: callable returning the current observation.
            duration: seconds to fly.
        """
        period = 1.0 / self.cfg.command_hz
        end = time.time() + duration
        try:
            self.connect()
            while time.time() < end:
                start = time.time()
                self.send(policy(get_observation()))
                slack = period - (time.time() - start)
                if slack > 0:
                    time.sleep(slack)
                elif -slack > self.cfg.command_timeout:
                    print("[bridge] control loop too slow -- landing")
                    break
        except Exception as exc:  # land on any failure, then re-raise
            print(f"[bridge] error: {exc!r} -- landing")
            raise
        finally:
            self.close()


def _demo_policy(_obs):
    """Tiny built-in policy for dry runs: gentle forward flight."""
    return [0.3, 0.0, 0.0, 0.0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drive a Tello EDU from a policy.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without connecting.")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds to fly.")
    parser.add_argument("--ip", type=str, default=None, help="Drone IP (station mode).")
    parser.add_argument("--speed-cap", type=float, default=0.5, help="Fraction of full scale allowed.")
    args = parser.parse_args()

    bridge = TelloBridge(BridgeCfg(speed_cap=args.speed_cap), dry_run=args.dry_run, ip=args.ip)
    bridge.run_policy(_demo_policy, lambda: None, args.duration)

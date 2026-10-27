"""Policy interface and the shared height hold.

A policy gets its drones once (setup) and then, every tick, returns each airborne drone's
rc command. A drone object offers height_cm(), battery(), num and name (see Drone in
swarm_policies.py). self.bus is the shared telemetry (telemetry.py): publish what other
policies may use, read what they publish.
"""
from .telemetry import TelemetryBus

LOW_CM = 91                  # 3 ft
HIGH_CM = 183                # 6 ft
KP_HEIGHT = 0.6              # rc per cm of height error
MAX_RC_Z = 30
DEADBAND_CM = 4


def clip(v, lim):
    return int(max(-lim, min(lim, round(v))))


def hold_height(d, target_cm):
    """rc throttle toward target_cm; 0 when the height is unknown or within the deadband."""
    h = d.height_cm()
    if h is None or target_cm is None or abs(target_cm - h) <= DEADBAND_CM:
        return 0
    return clip(KP_HEIGHT * (target_cm - h), MAX_RC_Z)


class Policy:
    """step(d, t) -> (a, b, c, d) rc for drone d, t = s since its takeoff; None = land it."""
    name = "policy"
    bus = None                   # set by the umbrella before setup; else a private bus

    def setup(self, drones):
        self.drones = list(drones)
        if self.bus is None:
            self.bus = TelemetryBus()

    def step(self, d, t):
        raise NotImplementedError

    def describe(self, d):
        return self.name

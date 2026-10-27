"""The real flight engine's action space: integer ``rc right forward up yaw`` sticks.

``drone_deployment_infra/core/tello_dual_video.py`` streams ``rc a b c d`` at ~20 Hz with
a = right, b = forward, c = up, d = yaw (clockwise positive), each an integer in
[-100, 100], and its policy saturates them at ``rc_limit`` (tello_policy.py: 25 / 25 / 22 /
35). A sim policy here emits the same four channels, in the same order, as a in [-1, 1]:

    rc = round(a * hi) if a > 0 else round(-a * lo) -> exactly what the real engine sends
        [lo, hi] per channel is the command envelope: the rc range actually sent to (and
        flown by) a real Tello in the logs (fit file ``command_envelope``), so the policy can
        never ask for a stick value no drone has received. Without it: [-rc_limit, rc_limit].
    speed = gain * rc   if |rc| >= deadband else 0  -> the firmware's set-point
    -> TelloCascadeController command [vx_fwd, vy_left, vz_up, yaw_ccw] / max speeds

``gain``, ``deadband``, the response time and the command delay per channel come from a
motion-fit file (``tello_motion_fit.yaml``), written by ``scripts/fit_tello_motion.py``
from the real flights' training logs. Until real logs are fitted it holds the priors the
flight engine itself uses (0.8 m/s, 0.6 m/s, 60 deg/s at rc 100).

Only torch and yaml are imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

CHANNELS = ("right", "forward", "up", "yaw")
DEFAULT_FIT = Path(__file__).resolve().parent / "tello_motion_fit.yaml"


@dataclass
class RcActionCfg:
    rc_limit: tuple[int, int, int, int] = (25, 25, 22, 35)
    """|rc| at |action| = 1, per channel (right, forward, up, yaw): tello_policy.py's
    MAX_RIGHT, MAX_FORWARD, CLIMB and MAX_YAW, so sim and real sticks stay in one range."""
    use_envelope: bool = True
    """Limit each channel to the fit file's ``command_envelope`` (rc values real drones were sent)."""
    fit_path: str = str(DEFAULT_FIT)
    """Motion fit: per-channel gain, deadband, response time and delay."""
    gain_range: tuple[float, float] = (0.85, 1.15)
    """Per-drone, per-channel multiplier on the fitted gain, sampled on reset (Tellos differ)."""


def load_motion_fit(path: str | Path = DEFAULT_FIT) -> dict:
    """{channel: {gain, deadband, tau, delay}} plus metadata. Gains are SI per rc unit
    (m/s, or rad/s for yaw); the file stores yaw in deg/s per rc unit for readability."""
    data = yaml.safe_load(Path(path).read_text())
    out = {"source": data.get("source", "unknown"), "envelope": data.get("command_envelope")}
    for ch in CHANNELS:
        c = data["channels"][ch]
        gain = float(c["gain"]) * (math.pi / 180.0 if ch == "yaw" else 1.0)
        out[ch] = {"gain": gain, "deadband": float(c["deadband"]), "tau": float(c["tau"]), "delay": float(c["delay"])}
    return out


def apply_motion_fit(dynamics_cfg, fit: dict, policy_dt: float) -> None:
    """Point the firmware model at the fitted response.

    The velocity loop is a PI on velocity error; with fast inner loops its closed-loop time
    constant is ~1 / kp, so kp_vel = 1 / tau. The command delay becomes the latency choices
    (fitted delay in policy steps, +-1 for randomization). Max speeds are what rc 100 gives.
    """
    d = dynamics_cfg
    d.max_speed_xy = 100.0 * 0.5 * (fit["right"]["gain"] + fit["forward"]["gain"])
    d.max_speed_z = 100.0 * fit["up"]["gain"]
    d.max_yaw_rate = 100.0 * fit["yaw"]["gain"]
    tau_xy = 0.5 * (fit["right"]["tau"] + fit["forward"]["tau"])
    d.kp_vel_xy = 1.0 / max(tau_xy, 0.05)
    d.ki_vel_xy = 0.2 * d.kp_vel_xy
    d.kp_vel_z = 1.0 / max(fit["up"]["tau"], 0.05)
    d.ki_vel_z = 0.5 * d.kp_vel_z
    delay = max(fit[ch]["delay"] for ch in CHANNELS)
    steps = int(round(delay / policy_dt))
    d.latency_steps = sorted({max(steps - 1, 0), steps, steps + 1})


class RcAction:
    """Batched action -> rc -> firmware command, for N drones."""

    def __init__(self, cfg: RcActionCfg, num_drones: int, max_speed_xy: float, max_speed_z: float,
                 max_yaw_rate: float, device: str):
        self.cfg = cfg
        self.device = device
        fit = load_motion_fit(cfg.fit_path)
        self.fit = fit
        sym = torch.tensor(cfg.rc_limit, dtype=torch.float32, device=device)
        if cfg.use_envelope and fit.get("envelope"):
            env = fit["envelope"]
            self.lo = torch.tensor([env[ch][0] for ch in CHANNELS], dtype=torch.float32, device=device)
            self.hi = torch.tensor([env[ch][1] for ch in CHANNELS], dtype=torch.float32, device=device)
        else:
            self.lo, self.hi = -sym, sym
        self.limit = torch.maximum(self.hi, -self.lo)  # largest |rc| per channel
        self.gain = torch.tensor([fit[ch]["gain"] for ch in CHANNELS], device=device)
        self.deadband = torch.tensor([fit[ch]["deadband"] for ch in CHANNELS], device=device)
        self.full_scale = torch.tensor([max_speed_xy, max_speed_xy, max_speed_z, max_yaw_rate], device=device)
        self.gain_scale = torch.ones(num_drones, 4, device=device)

    def randomize(self, ids: torch.Tensor, enabled: bool = True) -> None:
        if not enabled:
            self.gain_scale[ids] = 1.0
            return
        lo, hi = self.cfg.gain_range
        self.gain_scale[ids] = lo + (hi - lo) * torch.rand(len(ids), 4, device=self.device)

    def to_rc(self, actions: torch.Tensor) -> torch.Tensor:
        """(N, 4) actions in [-1, 1] -> integer-valued rc (right, forward, up, yaw_cw), inside
        [lo, hi]: 0 stays 0, +1 is the largest value sent, -1 the most negative."""
        a = actions.clamp(-1.0, 1.0)
        return torch.round(torch.where(a >= 0, a * self.hi, -a * self.lo))

    def from_rc(self, rc: torch.Tensor) -> torch.Tensor:
        """Inverse of to_rc (for tests): the action that sends ``rc`` (clipped to the envelope)."""
        rc = torch.maximum(torch.minimum(rc, self.hi), self.lo)
        pos = torch.where(self.hi > 0, rc / self.hi.clamp(min=1e-6), torch.zeros_like(rc))
        neg = torch.where(self.lo < 0, rc / (-self.lo).clamp(min=1e-6), torch.zeros_like(rc))
        return torch.where(rc >= 0, pos, neg)

    def rc_to_speed(self, rc: torch.Tensor) -> torch.Tensor:
        """rc -> firmware set-points (right m/s, forward m/s, up m/s, yaw_cw rad/s)."""
        speed = rc * self.gain * self.gain_scale
        return torch.where(rc.abs() >= self.deadband, speed, torch.zeros_like(speed))

    def command(self, rc: torch.Tensor) -> torch.Tensor:
        """rc -> TelloCascadeController command [vx_fwd, vy_left, vz_up, yaw_ccw] in [-1, 1]."""
        s = self.rc_to_speed(rc)
        cmd = torch.stack([s[:, 1], -s[:, 0], s[:, 2], -s[:, 3]], dim=-1)
        return (cmd / self.full_scale).clamp(-1.0, 1.0)

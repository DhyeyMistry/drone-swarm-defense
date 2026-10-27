"""Velocity-command control for the Tello EDU, matching the real SDK.

Why this exists: a real Tello accepts ``rc a b c d`` -- four velocity channels in
[-100, 100] -- and nothing lower level. Isaac Lab's built-in quadcopter task commands
body thrust and torque, so a policy trained on it can never fly a Tello. Every task in
this kit therefore acts through this controller, whose action vector maps 1:1 onto the
four ``rc`` channels:

    action = [vx, vy, vz, yaw_rate] in [-1, 1]
      vx -> rc b (forward+),  vy -> rc a (left+),
      vz -> rc c (up+),       yaw_rate -> rc d (clockwise+)

The onboard firmware closes the attitude loop on the real drone, so the sim does the
same here: the policy sets velocities, and this controller produces the wrench that
tracks them, with first-order lag and acceleration limits to imitate Tello response.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class TelloCommandCfg:
    """Limits and response of the Tello velocity interface."""

    max_lin_speed: float = 1.0
    """[m/s] per axis at |action| = 1. SDK ``rc`` is roughly +/-1 m/s in normal mode."""

    max_yaw_rate: float = 1.745
    """[rad/s] at |action| = 1 (100 deg/s)."""

    response_time: float = 0.15
    """[s] first-order lag between commanded and achieved velocity."""

    max_lin_accel: float = 4.0
    """[m/s^2] acceleration clamp."""

    max_yaw_accel: float = 10.0
    """[rad/s^2] yaw acceleration clamp."""

    attitude_kp: float = 12.0
    """Proportional gain of the levelling loop (firmware analogue)."""

    attitude_kd: float = 2.0
    """Derivative gain of the levelling loop."""

    command_latency_steps: int = 0
    """Delay applied to commands, in control steps. Set >0 to model SDK latency."""


class TelloVelocityController:
    """Turns [vx, vy, vz, yaw_rate] actions into body wrenches for a rigid body.

    Works on batched environments. Call :meth:`compute` each control step and pass the
    returned force/torque to ``RigidObject.set_external_force_and_torque``.
    """

    def __init__(self, cfg: TelloCommandCfg, num_envs: int, mass: float, inertia_zz: float, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.mass = mass
        self.inertia_zz = inertia_zz
        self.device = device
        # filtered (achieved) command state
        self._lin_vel_cmd = torch.zeros(num_envs, 3, device=device)
        self._yaw_rate_cmd = torch.zeros(num_envs, device=device)
        self._delay_buffer: list[torch.Tensor] = []

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear command state for the given environments (all, if None)."""
        if env_ids is None:
            self._lin_vel_cmd.zero_()
            self._yaw_rate_cmd.zero_()
            self._delay_buffer.clear()
        else:
            self._lin_vel_cmd[env_ids] = 0.0
            self._yaw_rate_cmd[env_ids] = 0.0

    def scale_actions(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map actions in [-1, 1] to SDK velocity targets: (lin_vel [m/s], yaw_rate [rad/s])."""
        actions = actions.clamp(-1.0, 1.0)
        return actions[:, :3] * self.cfg.max_lin_speed, actions[:, 3] * self.cfg.max_yaw_rate

    def to_rc(self, actions: torch.Tensor) -> torch.Tensor:
        """Map actions to the four integer ``rc a b c d`` channels in [-100, 100].

        Used by the real-drone bridge, and available in sim so logged commands are
        directly comparable with what the hardware receives.
        """
        a = actions.clamp(-1.0, 1.0)
        rc = torch.stack([a[:, 1], a[:, 0], a[:, 2], a[:, 3]], dim=-1)  # a=left, b=fwd, c=up, d=yaw
        return (rc * 100.0).round().clamp(-100.0, 100.0)

    def compute(
        self,
        actions: torch.Tensor,
        lin_vel_w: torch.Tensor,
        ang_vel_b: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (force_w, torque_b), each shaped (num_envs, 1, 3).

        Args:
            actions: policy output in [-1, 1], shape (num_envs, 4).
            lin_vel_w: current linear velocity in world frame, (num_envs, 3).
            ang_vel_b: current angular velocity in body frame, (num_envs, 3).
            projected_gravity_b: gravity direction in body frame, (num_envs, 3).
            dt: control step in seconds.
        """
        if self.cfg.command_latency_steps > 0:
            self._delay_buffer.append(actions.clone())
            if len(self._delay_buffer) > self.cfg.command_latency_steps:
                actions = self._delay_buffer.pop(0)
            else:
                actions = torch.zeros_like(actions)

        lin_target, yaw_target = self.scale_actions(actions)

        # first-order lag towards the target, with acceleration clamps
        alpha = min(dt / max(self.cfg.response_time, 1e-6), 1.0)
        lin_step = (lin_target - self._lin_vel_cmd) * alpha
        lin_step = lin_step.clamp(-self.cfg.max_lin_accel * dt, self.cfg.max_lin_accel * dt)
        self._lin_vel_cmd = self._lin_vel_cmd + lin_step
        yaw_step = (yaw_target - self._yaw_rate_cmd) * alpha
        yaw_step = yaw_step.clamp(-self.cfg.max_yaw_accel * dt, self.cfg.max_yaw_accel * dt)
        self._yaw_rate_cmd = self._yaw_rate_cmd + yaw_step

        # force that tracks the velocity command, plus gravity compensation
        vel_error = self._lin_vel_cmd - lin_vel_w
        accel = vel_error / max(self.cfg.response_time, 1e-6)
        accel = accel.clamp(-self.cfg.max_lin_accel, self.cfg.max_lin_accel)
        force_w = self.mass * accel
        force_w[:, 2] += self.mass * 9.81

        # levelling loop: hold roll/pitch at zero, track the yaw rate command
        torque_b = torch.zeros_like(force_w)
        torque_b[:, 0] = -self.cfg.attitude_kp * projected_gravity_b[:, 1] - self.cfg.attitude_kd * ang_vel_b[:, 0]
        torque_b[:, 1] = self.cfg.attitude_kp * projected_gravity_b[:, 0] - self.cfg.attitude_kd * ang_vel_b[:, 1]
        torque_b[:, 2] = self.inertia_zz * (self._yaw_rate_cmd - ang_vel_b[:, 2]) / max(self.cfg.response_time, 1e-6)
        torque_b[:, :2] *= self.inertia_zz

        return force_w.unsqueeze(1), torque_b.unsqueeze(1)

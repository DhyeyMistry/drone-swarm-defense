"""Tello waypoint task: follow a moving goal using SDK-compatible velocity commands."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.utils.math import sample_uniform

from action_space_kit.tasks.direct.tello_hover.tello_hover_env import TelloHoverEnv

from .tello_waypoint_env_cfg import TelloWaypointEnvCfg


class TelloWaypointEnv(TelloHoverEnv):
    """Hover task with a goal that drifts, so the policy learns to chase rather than hold."""

    cfg: TelloWaypointEnvCfg

    def __init__(self, cfg: TelloWaypointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._goal_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_resample_steps = max(int(self.cfg.goal_resample_time_s / self.step_dt), 1)

    def _resample_goal_velocity(self, env_ids: torch.Tensor) -> None:
        """Give the listed environments a new goal heading and speed."""
        num = len(env_ids)
        direction = torch.randn(num, 3, device=self.device)
        direction[:, 2] *= 0.3  # mostly horizontal motion
        direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1e-6)
        speed = sample_uniform(self.cfg.goal_speed[0], self.cfg.goal_speed[1], (num, 1), self.device)
        self._goal_vel_w[env_ids] = direction * speed

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        # periodically change direction
        due = (self.episode_length_buf % self._goal_resample_steps == 0).nonzero(as_tuple=False).flatten()
        if len(due) > 0:
            self._resample_goal_velocity(due)
        # move the goal and keep it inside the allowed volume
        self._goal_pos_w += self._goal_vel_w * self.step_dt
        local = self._goal_pos_w - self.scene.env_origins
        for axis in (0, 1):
            over = local[:, axis].abs() > self.cfg.goal_bounds_xy
            local[over, axis] = torch.sign(local[over, axis]) * self.cfg.goal_bounds_xy
            self._goal_vel_w[over, axis] *= -1.0
        low = local[:, 2] < self.cfg.goal_bounds_z[0]
        high = local[:, 2] > self.cfg.goal_bounds_z[1]
        local[low, 2] = self.cfg.goal_bounds_z[0]
        local[high, 2] = self.cfg.goal_bounds_z[1]
        self._goal_vel_w[low | high, 2] *= -1.0
        self._goal_pos_w = local + self.scene.env_origins

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._resample_goal_velocity(env_ids)

"""Tello hover task: hold a sampled position using SDK-compatible velocity commands."""

from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from action_space_kit.assets import TELLO_MASS
from action_space_kit.control import TelloVelocityController

from .tello_hover_env_cfg import TelloHoverEnvCfg


class TelloHoverEnv(DirectRLEnv):
    cfg: TelloHoverEnvCfg

    def __init__(self, cfg: TelloHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        inertia_zz = float(self._robot.root_physx_view.get_inertias()[0].reshape(3, 3)[2, 2])
        self._controller = TelloVelocityController(
            cfg=self.cfg.command,
            num_envs=self.num_envs,
            mass=TELLO_MASS,
            inertia_zz=inertia_zz,
            device=self.device,
        )
        # action_space may be an int or a gym space; flatdim handles both
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_scene(self):
        self._robot = RigidObject(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self._robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # goal marker
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/goal"
        self._goal_markers = VisualizationMarkers(marker_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        force_w, torque_b = self._controller.compute(
            actions=self._actions,
            lin_vel_w=self._robot.data.root_lin_vel_w,
            ang_vel_b=self._robot.data.root_ang_vel_b,
            projected_gravity_b=self._robot.data.projected_gravity_b,
            dt=self.physics_dt,
        )
        self._robot.set_external_force_and_torque(forces=force_w, torques=torque_b, body_ids=[0])

    def _get_observations(self) -> dict:
        goal_error_w = self._goal_pos_w - self._robot.data.root_pos_w
        obs = torch.cat(
            [
                goal_error_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        distance = torch.linalg.norm(self._goal_pos_w - self._robot.data.root_pos_w, dim=-1)
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_w), dim=-1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=-1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=-1)
        upright = -self._robot.data.projected_gravity_b[:, 2]  # 1.0 when level
        rewards = (
            self.cfg.rew_scale_distance * torch.exp(-distance)
            + self.cfg.rew_scale_lin_vel * lin_vel
            + self.cfg.rew_scale_ang_vel * ang_vel
            + self.cfg.rew_scale_action_rate * action_rate
            + self.cfg.rew_scale_upright * upright
        )
        return rewards * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        distance = torch.linalg.norm(self._goal_pos_w - self._robot.data.root_pos_w, dim=-1)
        height = self._robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        died = (distance > self.cfg.max_distance) | (height < self.cfg.min_height)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        num_reset = len(env_ids)

        # sample goals around the env origin
        goal = torch.zeros(num_reset, 3, device=self.device)
        goal[:, :2] = sample_uniform(-self.cfg.goal_range_xy, self.cfg.goal_range_xy, (num_reset, 2), self.device)
        goal[:, 2] = sample_uniform(self.cfg.goal_range_z[0], self.cfg.goal_range_z[1], (num_reset,), self.device)
        self._goal_pos_w[env_ids] = goal + self.scene.env_origins[env_ids]

        # start near, but not at, the goal
        state = self._robot.data.default_root_state[env_ids].clone()
        state[:, :3] = self._goal_pos_w[env_ids] + sample_uniform(-0.3, 0.3, (num_reset, 3), self.device)
        state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(state[:, 7:], env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._controller.reset(env_ids)

    def _debug_vis_callback(self, event):
        self._goal_markers.visualize(self._goal_pos_w)

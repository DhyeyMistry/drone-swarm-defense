"""Defend task driven by onboard sensing instead of perfect state.

Observations are built from gated, noisy detections smoothed by a constant-velocity
tracker, so a policy trained here degrades gracefully when a target leaves the field of
view -- the behaviour that carries over to real drones with real cameras.

Camera images are rendered and available as ``env.scene.sensors["camera"].data`` for
teams who want to train a detector end to end; the default observation stays low
dimensional so PPO trains in hackathon time.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import quat_apply

from action_space_kit.perception import ConstantVelocityTracker, gate_detections
from action_space_kit.tasks.direct.tello_defend.tello_defend_env import TelloDefendEnv

from .tello_defend_camera_env_cfg import TelloDefendCameraEnvCfg


class TelloDefendCameraEnv(TelloDefendEnv):
    cfg: TelloDefendCameraEnvCfg

    def __init__(self, cfg: TelloDefendCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._trackers = [
            ConstantVelocityTracker(self.cfg.tracker, self.num_envs, self.cfg.num_attackers, self.device)
            for _ in range(self.cfg.num_defenders)
        ]
        self._forward_b = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

    def _setup_extra_sensors(self) -> None:
        # Runs before the scene is cloned, which is when per-env camera prims appear.
        self._camera = TiledCamera(self.cfg.camera)
        self.scene.sensors["camera"] = self._camera

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self.cfg.use_ground_truth:
            return super()._get_observations()

        attacker_pos, _ = self._attacker_states()
        site = self.scene.env_origins.clone()
        site[:, 2] += self.cfg.site_height * 0.5

        obs = {}
        for i, drone in enumerate(self._defenders):
            forward_w = quat_apply(drone.data.root_quat_w, self._forward_b)
            detections, visible = gate_detections(
                attacker_pos,
                drone.data.root_pos_w,
                forward_w,
                fov_deg=self.cfg.detection_fov_deg,
                max_range=self.cfg.detection_range,
                position_noise=self.cfg.detection_noise,
                dropout=self.cfg.detection_dropout,
            )
            # a downed attacker cannot be detected
            visible = visible & self._attacker_alive
            tracker = self._trackers[i]
            tracker.update(detections, visible, self.step_dt)
            track_pos, track_vel, active = tracker.state()

            rel_pos = track_pos - drone.data.root_pos_w.unsqueeze(1)
            rel_vel = track_vel - drone.data.root_lin_vel_w.unsqueeze(1)
            # Same encoding as the state-based task: zero out untracked slots and flag
            # them, so the policy can transfer between the two observation modes.
            mask = active.unsqueeze(-1).float()
            rel_pos = rel_pos.clamp(-self.cfg.obs_clip_pos, self.cfg.obs_clip_pos) * mask
            rel_vel = rel_vel.clamp(-self.cfg.obs_clip_vel, self.cfg.obs_clip_vel) * mask

            obs[f"defender_{i}"] = torch.cat(
                [
                    drone.data.root_pos_w - site,
                    drone.data.root_lin_vel_w,
                    drone.data.projected_gravity_b,
                    rel_pos.reshape(self.num_envs, -1),
                    rel_vel.reshape(self.num_envs, -1),
                    mask.squeeze(-1),
                ],
                dim=-1,
            )
        return obs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        ids = None if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        for tracker in getattr(self, "_trackers", []):
            tracker.reset(ids)

    def _get_terminal_observations(self):
        # This task updates noisy trackers while observing. A terminal snapshot must not
        # advance the surviving environments' trackers a second time or consume their RNG.
        import copy
        trackers = self._trackers
        self._trackers = copy.deepcopy(trackers)
        device = torch.device(self.device)
        try:
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                return self._get_observations()
        finally:
            self._trackers = trackers

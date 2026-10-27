"""Approach task: four Tellos enter a base's airspace, each through its own gate.

Every drone is an agent with the four ``rc`` channels as its action, flown through the
randomized Tello firmware model in ``action_space_kit.control.tello_dynamics``. One policy
is shared by all four (see ``action_space_kit.rl.sb3_shared``), so each real Tello can run
it on its own ego-centric observation.

Drone state is kept as (num_envs, num_drones, ...) tensors; the controller works on the
flattened (num_envs * num_drones, ...) view, row = env * num_drones + drone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz

from action_space_kit.assets import TELLO_CFG
from action_space_kit.control import TelloCascadeController
from action_space_kit.rl.terminal_observation import TerminalObservationMixin

from .tello_approach_env_cfg import TelloApproachEnvCfg


class TelloApproachEnv(TerminalObservationMixin, DirectMARLEnv):
    cfg: TelloApproachEnvCfg

    def __init__(self, cfg: TelloApproachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        E, D = self.num_envs, self.cfg.num_drones
        self._agents = self.cfg.possible_agents

        view = self._drones[0].root_physx_view
        inertia = view.get_inertias()[0].reshape(3, 3).to(self.device)
        self._nominal_mass = float(view.get_masses()[0].sum())
        self._default_masses = [d.root_physx_view.get_masses().clone() for d in self._drones]
        self.controller = TelloCascadeController(
            cfg=self.cfg.dynamics,
            num_drones=E * D,
            nominal_mass=self._nominal_mass,
            inertia=inertia,
            physics_dt=self.physics_dt,
            device=self.device,
        )

        self._actions = torch.zeros(E, D, 4, device=self.device)
        self._prev_actions = torch.zeros(E, D, 4, device=self.device)
        self._waypoints = torch.zeros(E, D, 3, device=self.device)  # env-local
        self._landing_spots = torch.zeros(E, D, 3, device=self.device)  # env-local, on the ground
        self._spawn_azimuth = torch.zeros(E, device=self.device)
        self._gate_passed = torch.zeros(E, D, dtype=torch.bool, device=self.device)
        self._arrived = torch.zeros(E, D, dtype=torch.bool, device=self.device)  # over its spot, landing
        self._landed = torch.zeros(E, D, dtype=torch.bool, device=self.device)  # touched down, motors off
        self._near_base = torch.zeros(E, D, dtype=torch.bool, device=self.device)  # came within near_base_radius
        self._prev_dist = torch.zeros(E, D, device=self.device)
        # per-step events, computed in _get_dones and consumed in _get_rewards
        self._gate_before = torch.zeros_like(self._gate_passed)
        self._arrived_before = torch.zeros_like(self._arrived)
        self._gate_event = torch.zeros_like(self._gate_passed)
        self._goal_event = torch.zeros_like(self._arrived)
        self._violation = torch.zeros_like(self._arrived)
        # episode outcome, latched before Isaac Lab resets the env (read by evaluation and logging)
        self._ep_illegal_entry = torch.zeros(E, dtype=torch.bool, device=self.device)
        self._ep_collision = torch.zeros(E, dtype=torch.bool, device=self.device)
        self._ep_crash = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.outcome_valid = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.outcome_success = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.outcome_arrived = torch.zeros(E, device=self.device)
        self.outcome_landed = torch.zeros(E, device=self.device)
        self.outcome_near_base = torch.zeros(E, device=self.device)
        self.outcome_gates = torch.zeros(E, device=self.device)
        self.outcome_illegal_entry = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.outcome_collision = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.outcome_crash = torch.zeros(E, dtype=torch.bool, device=self.device)
        self.outcome_time = torch.zeros(E, device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    # -- scene ------------------------------------------------------------------

    def _setup_scene(self):
        self._drones = []
        for i in range(self.cfg.num_drones):
            obj = RigidObject(TELLO_CFG.replace(prim_path=f"/World/envs/env_.*/Drone_{i}"))
            self._drones.append(obj)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, obj in enumerate(self._drones):
            self.scene.rigid_objects[f"drone_{i}"] = obj
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # One color per drone; its beacon, gate ring and landing pad share it, so the video shows
    # which gate and spot each drone was assigned.
    DRONE_COLORS = [(1.0, 0.5, 0.0), (0.0, 0.8, 1.0), (1.0, 0.2, 0.8), (0.55, 1.0, 0.2)]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_vis_markers"):
                self._create_markers()
            for m in self._vis_markers:
                m.set_visibility(True)
        elif hasattr(self, "_vis_markers"):
            for m in self._vis_markers:
                m.set_visibility(False)

    def _create_markers(self) -> None:
        """Markers that stay readable on camera.

        The RTX viewport ignores material opacity here, so a solid keep-out sphere would
        hide everything inside it. It is drawn as a shell of dots instead, and each gate
        as a ring of dots around its waypoint.
        """
        cfg = self.cfg
        D = cfg.num_drones

        def mat(color):
            return sim_utils.PreviewSurfaceCfg(diffuse_color=color)

        def per_drone(path, make):
            return VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=path, markers={f"d{i}": make(self.DRONE_COLORS[i]) for i in range(D)}
                )
            )

        self._shell_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/keepout",
                markers={"dot": sim_utils.SphereCfg(radius=0.03, visual_material=mat((0.85, 0.15, 0.15)))},
            )
        )
        self._zone_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/touchdown_zone",
                markers={
                    "zone": sim_utils.CylinderCfg(radius=cfg.goal_radius, height=0.004, visual_material=mat((0.15, 0.55, 0.2)))
                },
            )
        )
        self._pad_markers = per_drone(
            "/Visuals/landing_pads", lambda c: sim_utils.CylinderCfg(radius=0.13, height=0.01, visual_material=mat(c))
        )
        self._ring_markers = per_drone(
            "/Visuals/gate_rings", lambda c: sim_utils.SphereCfg(radius=0.035, visual_material=mat(c))
        )
        self._beacon_markers = per_drone(
            "/Visuals/drone_beacons", lambda c: sim_utils.SphereCfg(radius=0.08, visual_material=mat(c))
        )
        self._vis_markers = [
            self._shell_markers,
            self._zone_markers,
            self._pad_markers,
            self._ring_markers,
            self._beacon_markers,
        ]
        # upper half of a Fibonacci sphere, for the keep-out shell
        n = 500
        k = torch.arange(n, device=self.device, dtype=torch.float32) + 0.5
        z = 1.0 - 2.0 * k / n
        phi = k * math.pi * (3.0 - math.sqrt(5.0))
        rho = torch.sqrt(1.0 - z * z)
        pts = torch.stack([rho * torch.cos(phi), rho * torch.sin(phi), z], dim=-1)
        self._shell_points = cfg.keepout_radius * pts[pts[:, 2] > -0.02]
        self._ring_angles = torch.linspace(0.0, 2.0 * math.pi, 25, device=self.device)[:-1]

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_waypoints"):
            return
        cfg = self.cfg
        E, D = self.num_envs, cfg.num_drones
        origins = self.scene.env_origins
        drone_ids = torch.arange(D, device=self.device).repeat(E)

        self._shell_markers.visualize((origins.unsqueeze(1) + self._shell_points).reshape(-1, 3))
        self._zone_markers.visualize(origins + torch.tensor([0.0, 0.0, 0.002], device=self.device))
        pads = self._landing_spots + origins.unsqueeze(1) + torch.tensor([0.0, 0.0, 0.008], device=self.device)
        self._pad_markers.visualize(pads.reshape(-1, 3), marker_indices=drone_ids)
        self._beacon_markers.visualize(self._stack("root_pos_w").reshape(-1, 3), marker_indices=drone_ids)

        # gate rings: radius gate_radius, in the plane facing the approach vector
        n = self._waypoints / torch.linalg.norm(self._waypoints, dim=-1, keepdim=True).clamp(min=1e-6)
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(n)
        e1 = torch.linalg.cross(n, up)
        e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True).clamp(min=1e-6)
        e2 = torch.linalg.cross(n, e1)
        c, s = torch.cos(self._ring_angles), torch.sin(self._ring_angles)
        ring = cfg.gate_radius * (c.view(1, 1, -1, 1) * e1.unsqueeze(2) + s.view(1, 1, -1, 1) * e2.unsqueeze(2))
        ring = ring + self._waypoints.unsqueeze(2) + origins.view(E, 1, 1, 3)
        ring_ids = drone_ids.view(E, D, 1).expand(-1, -1, ring.shape[2])
        self._ring_markers.visualize(ring.reshape(-1, 3), marker_indices=ring_ids.reshape(-1))

    # -- state helpers ------------------------------------------------------------

    def _stack(self, name: str) -> torch.Tensor:
        return torch.stack([getattr(d.data, name) for d in self._drones], dim=1)

    def _landing_targets(self) -> torch.Tensor:
        """The point above each drone's spot where it is handed to ``land``, (num_envs, D, 3)."""
        return self._landing_spots + torch.tensor([0.0, 0.0, self.cfg.landing_hover_height], device=self.device)

    def _positions(self) -> torch.Tensor:
        """Drone positions relative to the base (env origin), (num_envs, num_drones, 3)."""
        return self._stack("root_pos_w") - self.scene.env_origins.unsqueeze(1)

    # -- stepping ---------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = torch.stack([actions[a] for a in self._agents], dim=1).clamp(-1.0, 1.0)
        # Over its landing spot the drone is sent the SDK ``land`` command: the firmware
        # descends vertically and ignores the policy. Once down, it takes no command at all.
        land = torch.tensor([0.0, 0.0, -self.cfg.land_speed / self.cfg.dynamics.max_speed_z, 0.0], device=self.device)
        command = torch.where(self._arrived.unsqueeze(-1), land, self._actions)
        command[self._landed] = 0.0
        self.controller.set_command(command.reshape(-1, 4))

    def _apply_action(self) -> None:
        E, D = self.num_envs, self.cfg.num_drones
        force_b, torque_b = self.controller.compute(
            lin_vel_w=self._stack("root_lin_vel_w").reshape(-1, 3),
            quat_w=self._stack("root_quat_w").reshape(-1, 4),
            ang_vel_b=self._stack("root_ang_vel_b").reshape(-1, 3),
            lin_vel_b=self._stack("root_lin_vel_b").reshape(-1, 3),
        )
        force_b = force_b.view(E, D, 3)
        torque_b = torque_b.view(E, D, 3)
        # a drone that has touched down stops its motors and rests on the ground
        down = self._landed.unsqueeze(-1)
        force_b = torch.where(down, torch.zeros_like(force_b), force_b)
        torque_b = torch.where(down, torch.zeros_like(torque_b), torque_b)
        for i, drone in enumerate(self._drones):
            drone.set_external_force_and_torque(
                forces=force_b[:, i].unsqueeze(1), torques=torque_b[:, i].unsqueeze(1), body_ids=[0]
            )

    # -- MDP --------------------------------------------------------------------

    def _pairwise_distance(self, pos: torch.Tensor) -> torch.Tensor:
        """(num_envs, D, D) distances with +inf on the diagonal."""
        dist = torch.cdist(pos, pos)
        eye = torch.eye(self.cfg.num_drones, dtype=torch.bool, device=self.device)
        return dist.masked_fill(eye, float("inf"))

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        cfg = self.cfg
        pos = self._positions()
        dist_base = torch.linalg.norm(pos, dim=-1)
        dist_wp = torch.linalg.norm(pos - self._waypoints, dim=-1)

        self._gate_before = self._gate_passed.clone()
        self._arrived_before = self._arrived.clone()
        self._gate_event = ~self._gate_passed & (dist_wp < cfg.gate_radius)
        self._gate_passed |= self._gate_event
        dist_spot = torch.linalg.norm(pos - self._landing_targets(), dim=-1)
        self._goal_event = self._gate_passed & ~self._arrived & (dist_spot < cfg.landing_capture_radius)
        self._arrived |= self._goal_event
        self._landed |= self._arrived & (pos[..., 2] < cfg.touchdown_height)
        self._near_base |= dist_base < cfg.near_base_radius

        illegal = (dist_base < cfg.keepout_radius) & ~self._gate_passed
        tilt_cos = -self._stack("projected_gravity_b")[..., 2]
        crash = (
            ((pos[..., 2] < cfg.min_height) & ~self._arrived)  # landing is the one way down
            | (pos[..., 2] > cfg.max_height)
            | (torch.linalg.norm(pos[..., :2], dim=-1) > cfg.max_xy)
            | (tilt_cos < math.cos(cfg.max_tilt))
            | ~torch.isfinite(pos).all(dim=-1)
        )
        # contact counts unless both drones are already landing
        both_landing = self._arrived.unsqueeze(-1) & self._arrived.unsqueeze(-2)
        collision = ((self._pairwise_distance(pos) < cfg.collision_distance) & ~both_landing).any(dim=-1)
        self._violation = illegal | crash | collision
        self._ep_illegal_entry |= illegal.any(dim=-1)
        self._ep_collision |= collision.any(dim=-1)
        self._ep_crash |= crash.any(dim=-1)

        success = self._landed.all(dim=-1)
        done = self._violation.any(dim=-1) | success
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        finished = done | time_out
        self.outcome_valid = finished
        latch = lambda new, old: torch.where(finished, new, old)  # noqa: E731
        self.outcome_success = latch(success, self.outcome_success)
        self.outcome_arrived = latch(self._arrived.float().sum(-1), self.outcome_arrived)
        self.outcome_landed = latch(self._landed.float().sum(-1), self.outcome_landed)
        self.outcome_near_base = latch(self._near_base.float().mean(-1), self.outcome_near_base)
        self.outcome_gates = latch(self._gate_passed.float().sum(-1), self.outcome_gates)
        self.outcome_illegal_entry = latch(self._ep_illegal_entry, self.outcome_illegal_entry)
        self.outcome_collision = latch(self._ep_collision, self.outcome_collision)
        self.outcome_crash = latch(self._ep_crash, self.outcome_crash)
        self.outcome_time = latch(self.episode_length_buf.float() * self.step_dt, self.outcome_time)

        terminated = {agent: done for agent in self._agents}
        time_outs = {agent: time_out for agent in self._agents}
        return terminated, time_outs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        pos = self._positions()
        dist_base = torch.linalg.norm(pos, dim=-1)
        dist_wp = torch.linalg.norm(pos - self._waypoints, dim=-1)

        # +1 for ending the step closer to the target than the last one, -0.5 otherwise.
        # The target is the drone's own waypoint, then (once the gate is passed) the point
        # above its own landing spot.
        # Distance is measured against the target the drone had at the start of the step,
        # then re-based when the gate opens, so switching targets is neither paid nor punished.
        dist_spot = torch.linalg.norm(pos - self._landing_targets(), dim=-1)
        dist_target = torch.where(self._gate_before, dist_spot, dist_wp)
        progress = torch.where(dist_target < self._prev_dist, cfg.rew_closer, cfg.rew_not_closer)
        self._prev_dist = torch.where(self._gate_passed, dist_spot, dist_wp)

        # soft wall around the keep-out sphere, fading out near one's own gate
        clearance = dist_base - cfg.keepout_radius
        near_wall = ((cfg.barrier_margin - clearance) / cfg.barrier_margin).clamp(0.0, 1.0)
        away_from_gate = ((dist_wp - cfg.gate_radius) / 0.6).clamp(0.0, 1.0)
        barrier = near_wall * away_from_gate * (~self._gate_passed).float()

        pair = self._pairwise_distance(pos)
        separation = ((cfg.separation_distance - pair) / cfg.separation_distance).clamp(min=0.0).sum(dim=-1)

        effort = torch.sum(self._actions**2, dim=-1)
        action_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)

        rates = (
            cfg.rew_scale_barrier * barrier
            + cfg.rew_scale_separation * separation
            + cfg.rew_scale_effort * effort
            + cfg.rew_scale_action_rate * action_rate
        ) * self.step_dt
        shaping = progress + rates + cfg.rew_time
        # speed into the sphere as the gate is reached: the radial component towards the base
        inward_speed = (-(self._stack("root_lin_vel_w") * pos).sum(-1) / dist_base.clamp(min=1e-6)).clamp(min=0.0)
        gate_reward = cfg.rew_gate + cfg.rew_scale_entry_speed * inward_speed
        events = gate_reward * self._gate_event.float() + cfg.rew_goal * self._goal_event.float()
        # a landing or landed drone earns nothing, except on the step it arrived
        active = (~self._arrived_before).float()
        reward = active * (shaping + events) + cfg.rew_violation * self._violation.float()
        reward = torch.nan_to_num(reward, nan=cfg.rew_violation)
        return {agent: reward[:, i] for i, agent in enumerate(self._agents)}

    def _get_observations(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        D = cfg.num_drones
        pos = self._positions().clamp(-cfg.obs_clip_pos, cfg.obs_clip_pos)
        quat = self._stack("root_quat_w")
        quat = torch.where(quat[..., :1] < 0.0, -quat, quat)  # q and -q are the same attitude
        vel = self._stack("root_lin_vel_w")
        base = torch.zeros(self.num_envs, 3, device=self.device)  # base sits at the env origin
        obs = {}
        for i, agent in enumerate(self._agents):
            others = [(i + k) % D for k in range(1, D)]
            obs[agent] = torch.cat(
                [
                    pos[:, i],
                    quat[:, i],
                    pos[:, others].reshape(self.num_envs, -1),
                    quat[:, others].reshape(self.num_envs, -1),
                    base,
                    vel[:, i],
                    self._waypoints[:, i],
                    self._gate_passed[:, i : i + 1].float(),
                    self._landing_spots[:, i],
                ],
                dim=-1,
            )
            obs[agent] = torch.nan_to_num(obs[agent], nan=0.0, posinf=cfg.obs_clip_pos, neginf=-cfg.obs_clip_pos)
        return obs

    # -- reset ------------------------------------------------------------------

    def _sample_approach_vectors(self, azimuth: torch.Tensor) -> torch.Tensor:
        """Unit approach vectors, (n, D, 3), sorted by azimuth (left to right seen from the base).

        Each is elevated within ``approach_elevation`` and within ``approach_azimuth_spread``
        of the spawn direction; every pair is at least ``approach_min_separation`` apart.
        Uses vectorized rejection sampling over K candidate sets, with an evenly spaced
        fallback that always satisfies the constraints.
        """
        cfg = self.cfg
        n, D, K = azimuth.shape[0], cfg.num_drones, 32
        dev = self.device
        az = (torch.rand(n, K, D, device=dev) * 2.0 - 1.0) * cfg.approach_azimuth_spread
        el_lo, el_hi = cfg.approach_elevation
        el = el_lo + (el_hi - el_lo) * torch.rand(n, K, D, device=dev)

        # fallback: evenly spread azimuths, mid elevation
        fb_az = torch.linspace(-1.0, 1.0, D, device=dev) * min(cfg.approach_azimuth_spread, math.radians(30.0))
        fb_el = torch.full((D,), 0.5 * (el_lo + el_hi), device=dev)

        def to_dir(a, e):
            a = a + azimuth.view(-1, *([1] * (a.dim() - 1)))
            return torch.stack([torch.cos(e) * torch.cos(a), torch.cos(e) * torch.sin(a), torch.sin(e)], dim=-1)

        dirs = to_dir(az, el)  # (n, K, D, 3)
        cos = torch.einsum("nkid,nkjd->nkij", dirs, dirs)
        cos = cos - 2.0 * torch.eye(D, device=dev)  # ignore the diagonal
        valid = (cos.amax(dim=(-1, -2)) < math.cos(cfg.approach_min_separation))  # (n, K)
        pick = torch.argmax(valid.float(), dim=1)
        has = valid.any(dim=1)
        rows = torch.arange(n, device=dev)
        az_sel = torch.where(has.unsqueeze(-1), az[rows, pick], fb_az.expand(n, D))
        el_sel = torch.where(has.unsqueeze(-1), el[rows, pick], fb_el.expand(n, D))
        order = torch.argsort(az_sel, dim=-1)
        az_sel = torch.gather(az_sel, 1, order)
        el_sel = torch.gather(el_sel, 1, order)
        return to_dir(az_sel, el_sel)

    def _sample_spawn_line(self, azimuth: torch.Tensor) -> torch.Tensor:
        """Env-local spawn positions, (n, D, 3), ordered left to right seen from the base.

        The drones stand in a line inside a ``spawn_square`` square centred
        ``spawn_distance`` from the base, the line roughly perpendicular to the base
        direction.
        """
        cfg = self.cfg
        n, D, dev = azimuth.shape[0], cfg.num_drones, self.device
        half = 0.5 * cfg.spawn_square
        phi = (torch.rand(n, device=dev) * 2.0 - 1.0) * cfg.spawn_line_angle
        spacing = cfg.spawn_spacing[0] + (cfg.spawn_spacing[1] - cfg.spawn_spacing[0]) * torch.rand(n, device=dev)
        span = 0.5 * (D - 1) * spacing
        ext_r, ext_s = span * torch.sin(phi).abs(), span * torch.cos(phi)
        c_r = cfg.spawn_distance + (torch.rand(n, device=dev) * 2.0 - 1.0) * (half - ext_r)
        c_s = (torch.rand(n, device=dev) * 2.0 - 1.0) * (half - ext_s)
        k = torch.arange(D, device=dev) - 0.5 * (D - 1)
        r = c_r.unsqueeze(-1) + k * (spacing * torch.sin(phi)).unsqueeze(-1)
        s = c_s.unsqueeze(-1) + k * (spacing * torch.cos(phi)).unsqueeze(-1)
        e = torch.stack([torch.cos(azimuth), torch.sin(azimuth)], dim=-1).unsqueeze(1)  # towards the spawn
        lat = torch.stack([-torch.sin(azimuth), torch.cos(azimuth)], dim=-1).unsqueeze(1)  # left, seen from base
        xy = r.unsqueeze(-1) * e + s.unsqueeze(-1) * lat
        z = cfg.spawn_height[0] + (cfg.spawn_height[1] - cfg.spawn_height[0]) * torch.rand(n, D, device=dev)
        return torch.cat([xy, z.unsqueeze(-1)], dim=-1)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._drones[0]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._log_outcomes(env_ids)
        super()._reset_idx(env_ids)
        cfg = self.cfg
        n, D = len(env_ids), cfg.num_drones
        if n == self.num_envs and self.num_envs > 1:
            # spread initial resets over time so episodes don't finish in lockstep (as ese651 does)
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        azimuth = (torch.rand(n, device=self.device) * 2.0 - 1.0) * cfg.spawn_azimuth_range
        self._spawn_azimuth[env_ids] = azimuth
        spawn = self._sample_spawn_line(azimuth)
        waypoints = cfg.keepout_radius * self._sample_approach_vectors(azimuth)
        spot_az = azimuth.unsqueeze(-1) + cfg.landing_spot_spacing * (torch.arange(D, device=self.device) - 0.5 * (D - 1))
        spots = torch.stack(
            [cfg.landing_spot_radius * torch.cos(spot_az), cfg.landing_spot_radius * torch.sin(spot_az), torch.zeros_like(spot_az)],
            dim=-1,
        )
        # Line slot k, gate k and landing spot k are all k-th from the left, so paths don't
        # cross. Which drone gets which slot is shuffled so no agent index learns a fixed role.
        slot = torch.argsort(torch.rand(n, D, device=self.device), dim=-1).unsqueeze(-1).expand(-1, -1, 3)
        spawn = torch.gather(spawn, 1, slot)
        self._waypoints[env_ids] = torch.gather(waypoints, 1, slot)
        self._landing_spots[env_ids] = torch.gather(spots, 1, slot)

        yaw = azimuth.unsqueeze(-1) + math.pi + (torch.rand(n, D, device=self.device) * 2.0 - 1.0) * cfg.spawn_yaw_noise
        zeros = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(zeros.flatten(), zeros.flatten(), yaw.flatten()).view(n, D, 4)

        rows = (env_ids.unsqueeze(-1) * D + torch.arange(D, device=self.device)).flatten()
        self.controller.reset(rows, randomize=True)
        mass_scale = self.controller.mass_scale.view(self.num_envs, D)
        env_ids_cpu = env_ids.cpu()
        origins = self.scene.env_origins[env_ids]
        for i, drone in enumerate(self._drones):
            state = drone.data.default_root_state[env_ids].clone()
            state[:, :3] = origins + spawn[:, i]
            state[:, 3:7] = quat[:, i]
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            # true mass differs from what the firmware model assumes
            masses = drone.root_physx_view.get_masses()
            masses[env_ids_cpu] = self._default_masses[i][env_ids_cpu] * mass_scale[env_ids, i].cpu().unsqueeze(-1)
            drone.root_physx_view.set_masses(masses, env_ids_cpu)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._gate_passed[env_ids] = False
        self._arrived[env_ids] = False
        self._landed[env_ids] = False
        self._near_base[env_ids] = False
        self._prev_dist[env_ids] = torch.linalg.norm(spawn - self._waypoints[env_ids], dim=-1)
        self._ep_illegal_entry[env_ids] = False
        self._ep_collision[env_ids] = False
        self._ep_crash[env_ids] = False

    def _log_outcomes(self, env_ids: torch.Tensor) -> None:
        """Episode statistics for the envs about to reset, as ``extras['log']``."""
        finished = env_ids[self.outcome_valid[env_ids]]
        self.extras["episode_records"] = []
        if len(finished) == 0:
            self.extras["log"] = {}
            return
        fields = {"success_rate": "success", "drones_arrived": "arrived", "drones_landed": "landed",
                  "near_base_frac": "near_base", "gates_passed": "gates", "illegal_entry_rate": "illegal_entry",
                  "collision_rate": "collision", "crash_rate": "crash", "duration_s": "time"}
        values = {key: getattr(self, "outcome_" + name)[finished].detach().cpu().tolist()
                  for key, name in fields.items()}
        self.extras["episode_records"] = [
            {"env_id": env_id, "metrics": {"Episode/" + key: float(v[j]) for key, v in values.items()}}
            for j, env_id in enumerate(finished.cpu().tolist())]
        self.extras["log"] = {
            "Episode/success_rate": self.outcome_success[finished].float().mean().item(),
            "Episode/drones_arrived": self.outcome_arrived[finished].mean().item(),
            "Episode/drones_landed": self.outcome_landed[finished].mean().item(),
            "Episode/near_base_frac": self.outcome_near_base[finished].mean().item(),
            "Episode/gates_passed": self.outcome_gates[finished].mean().item(),
            "Episode/illegal_entry_rate": self.outcome_illegal_entry[finished].float().mean().item(),
            "Episode/collision_rate": self.outcome_collision[finished].float().mean().item(),
            "Episode/crash_rate": self.outcome_crash[finished].float().mean().item(),
            "Episode/duration_s": self.outcome_time[finished].mean().item(),
            "Episode/count": float(len(finished)),
        }

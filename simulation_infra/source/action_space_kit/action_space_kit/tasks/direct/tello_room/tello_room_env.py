"""Room task: four Tellos form up 1 m in front of tag 10, localized only by AprilTags.

Per policy step (``command_rate_hz`` = 13.5 Hz, the measured in-flight rc rate; one step =
one rc packet):

  action    [right, forward, up, yaw_cw, land] in [-1, 1]: the sticks become integer rc inside
            the logged command envelope (``RcAction``: only values real drones were sent);
            land > land_threshold for land_confirm_steps sends the SDK ``land``: the firmware
            descends and cuts the motors at touchdown, after the estimated-state dwell gate
            approves it (optional ``auto_land`` uses the same gate without a policy request)
            -> lost on the WiFi link with the episode's loss rate (the drone holds the last rc)
            -> lands 0-16 ms late in the step -> the motion-fitted Tello firmware model -> PhysX
  sensing   the frame captured now is checked against every mapped tag (``tags.sensing``);
            its fix reaches the controller ``video_delay_steps`` later. That fix is all the
            positional data there is (core/positional.py): no odometry. Each drone publishes
            its fix to the others.
  lost      no fix for ``lost_after`` -> the drone is lost: its position stays frozen and
            flagged until the policy itself turns back to a tag (nothing takes over)
  noise     camera-model noise + per-fix noise + a per-drone bias on every fix; every drone
            has its own PID / flight-dynamics gains, rc->speed gains, mass and latency
  waypoint  at tag-center height above each landing spot, radius 0.3 m: until its published fix is
            inside it, the drone is rewarded for closing on the waypoint; from then on only
            for landing near its spot, plus the penalties
  success   all four landed, each truly within ``success_radius`` of its spot

The policy is trained end to end: its sticks always fly the drone. Observations are the
positional feed only, never the true pose. One central controller sees all four drones'
blocks and sends all four drones' commands (rl.sb3_shared.Sb3CentralVecEnv). Drone state is
(num_envs, num_drones, ...); the controller works on the flattened view,
row = env * num_drones + drone.
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
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

from action_space_kit.assets import TELLO_CFG
from action_space_kit.control import RcAction, TelloCascadeController, apply_motion_fit, load_motion_fit
from action_space_kit.control.landing import update_landing_gate
from action_space_kit.control.team_features import peer_features, predicted_separation_cost
from action_space_kit.rl.rewards import potential_delta, precision_potential
from action_space_kit.rl.start_curriculum import mix_starts, validate_start_mixture
from action_space_kit.rl.terminal_observation import TerminalObservationMixin
from action_space_kit.tags.kalman import KalmanCV
from action_space_kit.tags.room_map import load_room_tags
from action_space_kit.tags.sensing import camera_axes, sighting
from action_space_kit.tags.spawn import Wall, spawn_room

from .tello_room_env_cfg import TelloRoomEnvCfg


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class TelloRoomEnv(TerminalObservationMixin, DirectMARLEnv):
    cfg: TelloRoomEnvCfg

    def __init__(self, cfg: TelloRoomEnvCfg, render_mode: str | None = None, **kwargs):
        steps = 1.0 / (cfg.sim.dt * cfg.command_rate_hz)
        if abs(steps - round(steps)) > 1e-3:
            raise ValueError(f"command_rate_hz {cfg.command_rate_hz} is not a whole number of {1 / cfg.sim.dt:.0f} Hz physics steps")
        cfg.decimation = int(round(steps))
        cfg.sim.render_interval = cfg.decimation
        if cfg.use_motion_fit:
            apply_motion_fit(cfg.dynamics, load_motion_fit(cfg.rc.fit_path), cfg.sim.dt * cfg.decimation)
        self.room_tags = load_room_tags(cfg.tag10_height, cfg.room_map_path or None, cfg.room_tags_path or None)
        if cfg.land_gate_enabled and not cfg.kalman.enabled:
            raise ValueError("The landing gate requires Kalman velocity/covariance (env.kalman.enabled=True).")
        validate_start_mixture(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        E, D = self.num_envs, self.cfg.num_drones
        dev = self.device
        self._agents = self.cfg.possible_agents

        view = self._drones[0].root_physx_view
        inertia = view.get_inertias()[0].reshape(3, 3).to(dev)
        self._nominal_mass = float(view.get_masses()[0].sum())
        self._default_masses = [d.root_physx_view.get_masses().clone() for d in self._drones]
        dyn = self.cfg.dynamics
        self.controller = TelloCascadeController(
            cfg=dyn, num_drones=E * D, nominal_mass=self._nominal_mass, inertia=inertia,
            physics_dt=self.physics_dt, device=dev,
        )
        self.rc = RcAction(self.cfg.rc, E * D, dyn.max_speed_xy, dyn.max_speed_z, dyn.max_yaw_rate, dev)

        tags = list(self.room_tags.values())
        self._tag_ids = [t.tag_id for t in tags]
        self._tag_pos = torch.tensor([t.pos.tolist() for t in tags], dtype=torch.float32, device=dev)
        self._tag_normal = torch.tensor([t.normal.tolist() for t in tags], dtype=torch.float32, device=dev)
        self._tag_size = torch.tensor([t.size_m for t in tags], dtype=torch.float32, device=dev)
        self._cam_offset = torch.tensor(self.cfg.tag_sensor.camera_offset, device=dev)
        self._tag10 = torch.tensor([0.0, 0.0, self.cfg.tag10_height], device=dev)

        # spots: k-th from the left (smallest Y) is k; drones get them per episode (_reset_idx)
        S = self.cfg.spot_count
        if S < D:
            raise ValueError(f"spot_count {S} < num_drones {D}: every drone needs its own spot")
        k = torch.arange(S, device=dev, dtype=torch.float32) - 0.5 * (S - 1)
        self._spot_table = torch.stack(  # landing spots, on the floor
            [torch.full_like(k, self.cfg.spot_distance), k * self.cfg.spot_spacing, torch.zeros_like(k)], -1
        )
        self._spots = self._spot_table[:D].expand(E, D, 3).clone()
        self._wp_offset = torch.tensor([0.0, 0.0, self.cfg.waypoint_height], device=dev)
        self._start_kind = torch.zeros(E, dtype=torch.long, device=dev)
        self._start_positions = torch.zeros(E, D, 3, device=dev)

        self._actions = torch.zeros(E, D, 5, device=dev)
        self._prev_actions = torch.zeros(E, D, 5, device=dev)
        self.last_rc = torch.zeros(E, D, 4, device=dev)  # the integer sticks, as the real engine would send

        # positional feed: what core/positional.py publishes for each drone
        H = self.cfg.tag_sensor.video_delay_steps + 1
        self._fix_buf = torch.zeros(E, D, H, 3, device=dev)
        self._fix_ok_buf = torch.zeros(E, D, H, dtype=torch.bool, device=dev)
        self._yaw_buf = torch.zeros(E, D, H, device=dev)
        self.pub_pos = torch.zeros(E, D, 3, device=dev)  # last published position (Kalman estimate)
        self.raw_pos = torch.zeros(E, D, 3, device=dev)  # last raw fix, before the filter
        self.kf = KalmanCV(self.cfg.kalman, (E, D), xp=torch, device=dev)
        self.pub_yaw = torch.zeros(E, D, device=dev)  # last published heading (from the tag pose)
        self.fresh = torch.zeros(E, D, dtype=torch.bool, device=dev)  # a fix arrived this tick
        self.acquired = torch.zeros(E, D, dtype=torch.bool, device=dev)  # ever had a fix
        self.fix_age = torch.zeros(E, D, device=dev)
        self.lost = torch.zeros(E, D, dtype=torch.bool, device=dev)  # no position being published
        self._lost_event = torch.zeros(E, D, dtype=torch.bool, device=dev)
        self.height_meas = torch.zeros(E, D, device=dev)
        self.tags_seen = torch.zeros(E, D, len(tags), dtype=torch.bool, device=dev)
        self._link_loss = torch.zeros(E, D, device=dev)
        self._fix_bias = torch.zeros(E, D, 3, device=dev)
        self._delivered = torch.zeros(E, D, 4, device=dev)  # the firmware command the drone is flying
        self.rc_lost = torch.zeros(E, D, dtype=torch.bool, device=dev)

        self.at_waypoint = torch.zeros(E, D, dtype=torch.bool, device=dev)  # reached it (latched)
        self._waypoint_event = torch.zeros(E, D, dtype=torch.bool, device=dev)
        self._wp_before = torch.zeros(E, D, dtype=torch.bool, device=dev)
        self.landing = torch.zeros(E, D, dtype=torch.bool, device=dev)  # land sent, descending
        self.taking_off = torch.zeros(E, D, dtype=torch.bool, device=dev)  # takeoff sent, firmware climbing
        self._takeoff_z = torch.zeros(E, D, device=dev)
        self.landed = torch.zeros(E, D, dtype=torch.bool, device=dev)  # touched down, motors off
        self.land_ok = torch.zeros(E, D, dtype=torch.bool, device=dev)  # touched down on the spot
        self.touchdown_err = torch.zeros(E, D, device=dev)
        self._land_votes = torch.zeros(E, D, dtype=torch.long, device=dev)
        self.land_dwell = torch.zeros(E, D, device=dev)
        self.land_ready = torch.zeros(E, D, dtype=torch.bool, device=dev)
        self._prev_potential = torch.zeros(E, D, device=dev)
        self._land_event = torch.zeros(E, D, dtype=torch.bool, device=dev)  # land sent this step
        self._early_land = torch.zeros(E, D, dtype=torch.bool, device=dev)  # land asked for before the waypoint
        self._touchdown_event = torch.zeros(E, D, dtype=torch.bool, device=dev)
        self._done_now = torch.zeros(E, dtype=torch.bool, device=dev)
        self._all_event = torch.zeros(E, dtype=torch.bool, device=dev)
        self._violation = torch.zeros(E, D, dtype=torch.bool, device=dev)
        self._crash = torch.zeros(E, D, dtype=torch.bool, device=dev)  # this step: crash / wall
        self._fast_entry = torch.zeros(E, D, dtype=torch.bool, device=dev)  # this step: entered the waypoint too fast
        self._ep_fast_entry = torch.zeros(E, dtype=torch.bool, device=dev)
        self.outcome_fast_entry = torch.zeros(E, dtype=torch.bool, device=dev)
        self._best_speed = torch.full((E, D), float("inf"), device=dev)  # slowest so far in the slow zone
        self.est_speed = torch.zeros(E, D, device=dev)
        self._collision = torch.zeros(E, D, dtype=torch.bool, device=dev)  # this step: too close to another drone
        # per pair: collision_distance, or less for a pair that spawned closer (start distance - start_contact_margin)
        self._pair_limit = torch.full((E, D, D), cfg.collision_distance, device=dev)
        self._substep_collision = torch.zeros(E, D, dtype=torch.bool, device=dev)
        self._prev_dist = torch.zeros(E, D, device=dev)
        self._ep_collision = torch.zeros(E, dtype=torch.bool, device=dev)
        self._ep_crash = torch.zeros(E, dtype=torch.bool, device=dev)
        self._ep_fresh_steps = torch.zeros(E, device=dev)
        self._ep_losses = torch.zeros(E, device=dev)
        self._ep_gate_ready = torch.zeros(E, device=dev)
        self.outcome_gate_ready_frac = torch.zeros(E, device=dev)
        self._ep_min_sep = torch.full((E,), float("inf"), device=dev)
        # per-episode sums of reward terms, over the drones (logged to TensorBoard)
        self._ep_rew = {k: torch.zeros(E, device=dev) for k in self.REWARD_TERMS}
        self.outcome_rew = {k: torch.zeros(E, device=dev) for k in self._ep_rew}
        self._finished = torch.zeros(E, dtype=torch.bool, device=dev)
        self.outcome_valid = torch.zeros(E, dtype=torch.bool, device=dev)
        self.outcome_success = torch.zeros(E, dtype=torch.bool, device=dev)
        self.outcome_waypoints = torch.zeros(E, device=dev)
        self.outcome_landed = torch.zeros(E, device=dev)
        self.outcome_land_ok = torch.zeros(E, device=dev)
        self.outcome_collision = torch.zeros(E, dtype=torch.bool, device=dev)
        self.outcome_crash = torch.zeros(E, dtype=torch.bool, device=dev)
        self.outcome_timeout = torch.zeros(E, dtype=torch.bool, device=dev)
        self.outcome_time = torch.zeros(E, device=dev)
        self.outcome_fresh_frac = torch.zeros(E, device=dev)
        self.outcome_losses = torch.zeros(E, device=dev)
        self.outcome_spot_err = torch.zeros(E, device=dev)
        self.outcome_min_sep = torch.zeros(E, device=dev)

        self.set_debug_vis(self.cfg.debug_vis)

    # -- scene ------------------------------------------------------------------

    def _room_walls(self) -> list[Wall]:
        cfg = self.cfg
        depth = cfg.room_x_max - cfg.wall_x
        width = cfg.side_wall_y - cfg.room_y_min
        return [
            Wall("tag10_wall", (cfg.wall_x - 0.05, 0.5 * (cfg.side_wall_y + cfg.room_y_min), 0.5 * cfg.ceiling),
                 (0.1, width, cfg.ceiling)),
            Wall("tag5_wall", (cfg.wall_x + 0.5 * depth, cfg.side_wall_y + 0.05, 0.5 * cfg.ceiling),
                 (depth, 0.1, cfg.ceiling)),
            Wall("left_wall", (cfg.wall_x + 0.5 * depth, cfg.room_y_min - 0.05, 0.5 * cfg.ceiling),
                 (depth, 0.1, cfg.ceiling)),
        ]

    def _setup_scene(self):
        self._drones = []
        for i in range(self.cfg.num_drones):
            obj = RigidObject(TELLO_CFG.replace(prim_path=f"/World/envs/env_.*/Drone_{i}"))
            self._drones.append(obj)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # visual only; cloned with env_0
        spawn_room("/World/envs/env_0/Room", self.room_tags, self._room_walls())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, obj in enumerate(self._drones):
            self.scene.rigid_objects[f"drone_{i}"] = obj
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # every reward term, tallied per episode and logged as Rewards/avg_<term>_per_episode
    REWARD_TERMS = (
        "closer_reward",  # +10 per metre gained toward the own waypoint until reached, -5 per metre lost
        "waypoint_reward",  # +1000 once, on first getting inside the waypoint radius at <= 0.3 m/s
        "slowdown_reward",  # +200 per m/s of new low speed within 2 m of tag 10 (before the waypoint)
        "settling_loss",
        "fast_entry_loss",  # entering the waypoint too fast: -500 per 0.3 m/s over; the episode fails
        "landing_loss",  # -0.5 per step not landed on the spot (+ the rest, if ended early)
        "no_tag_loss",  # -0.5 per step a flying drone has no AprilTag in view
        "crash_loss",  # -100 per crash / wall / leaving the room
        "collision_loss",  # -1000 per drone that came within collision_distance of another
        "separation_loss",  # closer than 0.5 m to another drone
        "predictive_separation_loss",
        "action_rate_loss",  # jerky sticks
        "early_land_loss",  # -100 each time land is asked for before the waypoint (blocked)
        "land_reward",  # +100 at touchdown (only possible after the waypoint)
        "formation_reward",  # +200 when all four landed on their spots
        "total",
    )

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
        D = self.cfg.num_drones

        def per_drone(path, make):
            return VisualizationMarkers(VisualizationMarkersCfg(
                prim_path=path,
                markers={f"d{i}": make(sim_utils.PreviewSurfaceCfg(diffuse_color=self.DRONE_COLORS[i % 4])) for i in range(D)},
            ))

        self._spot_markers = per_drone("/Visuals/spots", lambda m: sim_utils.CylinderCfg(
            radius=self.cfg.success_radius, height=0.01, visual_material=m))
        self._wp_markers = per_drone("/Visuals/waypoints", lambda m: sim_utils.SphereCfg(
            radius=self.cfg.waypoint_radius, visual_material=m.replace(opacity=0.3)))  # see-through: tags behind it
        self._beacon_markers = per_drone("/Visuals/drone_beacons", lambda m: sim_utils.SphereCfg(
            radius=0.05, visual_material=m))
        self._pub_markers = per_drone("/Visuals/published_positions", lambda m: sim_utils.CuboidCfg(
            size=(0.07, 0.07, 0.07), visual_material=m))
        self._vis_markers = [self._spot_markers, self._wp_markers, self._beacon_markers, self._pub_markers]

    def _debug_vis_callback(self, event):
        if not hasattr(self, "pub_pos") or not hasattr(self, "scene"):  # before setup / after close
            return
        E, D = self.num_envs, self.cfg.num_drones
        origins = self.scene.env_origins.unsqueeze(1)
        ids = torch.arange(D, device=self.device).repeat(E)
        self._spot_markers.visualize((self._spots + origins).reshape(-1, 3), marker_indices=ids)
        self._wp_markers.visualize((self._spots + self._wp_offset + origins).reshape(-1, 3), marker_indices=ids)
        self._beacon_markers.visualize(self._stack("root_pos_w").reshape(-1, 3), marker_indices=ids)
        self._pub_markers.visualize((self.pub_pos + origins).reshape(-1, 3), marker_indices=ids)

    # -- helpers ----------------------------------------------------------------

    def _stack(self, name: str) -> torch.Tensor:
        return torch.stack([getattr(d.data, name) for d in self._drones], dim=1)

    def _positions(self) -> torch.Tensor:
        """True positions in the room frame (env-local), (num_envs, D, 3)."""
        return self._stack("root_pos_w") - self.scene.env_origins.unsqueeze(1)

    def _pairwise_distance(self, pos: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(pos, pos)
        eye = torch.eye(self.cfg.num_drones, dtype=torch.bool, device=self.device)
        return dist.masked_fill(eye, float("inf"))

    # -- stepping ---------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        E, D = self.num_envs, self.cfg.num_drones
        self._substep_collision.zero_()
        self._prev_actions = self._actions.clone()
        self._actions = torch.stack([actions[a] for a in self._agents], dim=1).clamp(-1.0, 1.0)
        rc = self.rc.to_rc(self._actions[..., :4].reshape(-1, 4)).view(E, D, 4)  # the policy always flies
        # land: asked for land_confirm_steps in a row -> the SDK land command is sent
        want = self._actions[..., 4] > self.cfg.land_threshold
        want &= ~self.taking_off  # the firmware is busy with takeoff
        self._land_votes = torch.where(want, self._land_votes + 1, torch.zeros_like(self._land_votes))
        asked = ~self.landing & (self._land_votes >= self.cfg.land_confirm_steps)
        if self.cfg.land_gate_enabled:
            self.land_dwell, self.land_ready = update_landing_gate(
                self.cfg, self.pub_pos, self.kf.v, self.kf.a, self.kf.c, self._spots,
                self.fix_age, self.acquired, self.lost, self.at_waypoint,
                self.taking_off | self.landing | self.landed, self.land_dwell, self.step_dt, torch)
            allowed = self.land_ready
        else:  # exact replay of legacy checkpoints, never the new training default
            allowed = self.at_waypoint
        # Penalize requests before the observable waypoint milestone. After reaching it,
        # the gate may still need a fresh/confident dwell; penalizing every blocked request
        # there teaches the discrete head to stop asking instead of learning touchdown.
        self._early_land = asked & ~self.at_waypoint
        self._ep_gate_ready += self.land_ready.float().mean(-1)
        self._land_votes = torch.where(asked & ~allowed, torch.zeros_like(self._land_votes), self._land_votes)
        self._land_event = asked & allowed
        if self.cfg.auto_land:  # land goes out on its own once inside the waypoint; the 5th output is unused
            self._early_land = torch.zeros_like(self._early_land)
            self._land_event = ~self.landing & allowed & ~self.taking_off
        self.landing |= self._land_event
        rc[self.landing | self.taking_off] = 0.0  # during land / takeoff the firmware ignores sticks
        self.last_rc = rc
        command = self.rc.command(rc.reshape(-1, 4)).view(E, D, 4)
        # a lost rc packet: the Tello keeps flying the last rc it received
        self.rc_lost = (torch.rand(E, D, device=self.device) < self._link_loss) & ~self.landing & ~self.taking_off
        command = torch.where(self.rc_lost.unsqueeze(-1), self._delivered, command)
        # land (an SDK command, retried until acknowledged): descend at land_speed; down: off
        descend = torch.tensor([0.0, 0.0, -self.cfg.land_speed / self.cfg.dynamics.max_speed_z, 0.0], device=self.device)
        command = torch.where(self.landing.unsqueeze(-1), descend, command)
        # takeoff (sent at the start when start_on_floor): climb at takeoff_speed to takeoff_height
        climb = torch.tensor([0.0, 0.0, self.cfg.takeoff_speed / self.cfg.dynamics.max_speed_z, 0.0], device=self.device)
        command = torch.where(self.taking_off.unsqueeze(-1), climb, command)
        command[self.landed] = 0.0
        self._delivered = command
        # the new rc reaches the firmware a random 0..command_jitter into the step; until
        # then the drone keeps flying the previous one
        held = self.controller._cmd.clone()
        self.controller.set_command(command.reshape(-1, 4))
        self._due_cmd = self.controller._cmd.clone()
        max_sub = int(self.cfg.command_jitter / self.physics_dt)
        self._apply_at = torch.randint(0, max_sub + 1, (E * D,), device=self.device)
        late = self._apply_at > 0
        self.controller._cmd[late] = held[late]
        self._substep = 0

    def _apply_action(self) -> None:
        if self.cfg.collision_substeps and self.cfg.num_drones > 1:
            pair = self._pairwise_distance(self._positions())
            both_down = self.landed.unsqueeze(-1) & self.landed.unsqueeze(-2)
            self._substep_collision |= ((pair < self._pair_limit) & ~both_down).any(-1)
            self._ep_min_sep = torch.minimum(self._ep_min_sep, pair.amin(dim=(-1, -2)))
        due = self._apply_at == self._substep
        if self._substep > 0 and due.any():
            self.controller._cmd[due] = self._due_cmd[due]
        self._substep += 1
        force_b, torque_b = self.controller.compute(
            lin_vel_w=self._stack("root_lin_vel_w").reshape(-1, 3),
            quat_w=self._stack("root_quat_w").reshape(-1, 4),
            ang_vel_b=self._stack("root_ang_vel_b").reshape(-1, 3),
            lin_vel_b=self._stack("root_lin_vel_b").reshape(-1, 3),
        )
        E, D = self.num_envs, self.cfg.num_drones
        force_b, torque_b = force_b.view(E, D, 3), torque_b.view(E, D, 3)
        down = self.landed.unsqueeze(-1)  # motors off after touchdown
        force_b = torch.where(down, torch.zeros_like(force_b), force_b)
        torque_b = torch.where(down, torch.zeros_like(torque_b), torque_b)
        for i, drone in enumerate(self._drones):
            drone.set_external_force_and_torque(
                forces=force_b[:, i].unsqueeze(1), torques=torque_b[:, i].unsqueeze(1), body_ids=[0]
            )

    # -- positional feed ----------------------------------------------------------

    def _update_feed(self) -> None:
        """One engine tick: this frame's sighting goes into the video pipeline; the one from
        video_delay_steps ago (if it survived the link) is published as the drone's position."""
        cfg = self.cfg
        E, D = self.num_envs, cfg.num_drones
        dt = self.step_dt
        pos = self._positions()
        quat = self._stack("root_quat_w")
        yaw = _yaw_of(quat)
        R = matrix_from_quat(quat.reshape(-1, 4))
        offset = R @ self._cam_offset
        cam_pos = pos.reshape(-1, 3) + offset
        cam_rot = camera_axes(R, cfg.tag_sensor.camera_pitch)
        ts = cfg.tag_sensor
        speed = torch.linalg.norm(self._stack("root_lin_vel_w"), dim=-1).reshape(-1)
        yaw_rate = self._stack("root_ang_vel_w")[..., 2].reshape(-1)
        fix, ok, mask = sighting(ts, cam_pos, cam_rot, self._tag_pos, self._tag_normal, self._tag_size,
                                 speed=speed, yaw_rate=yaw_rate)
        self.tags_seen = mask.view(E, D, -1)
        self._fix_buf = torch.roll(self._fix_buf, 1, dims=2)
        self._fix_ok_buf = torch.roll(self._fix_ok_buf, 1, dims=2)
        self._yaw_buf = torch.roll(self._yaw_buf, 1, dims=2)
        self._fix_buf[:, :, 0] = (fix - offset).view(E, D, 3)  # back to the body centre
        self._fix_ok_buf[:, :, 0] = ok.view(E, D)
        self._yaw_buf[:, :, 0] = yaw
        arrived = self._fix_ok_buf[:, :, -1] & (torch.rand(E, D, device=self.device) >= self._link_loss)

        noisy = self._fix_buf[:, :, -1] + self._fix_bias + cfg.fix_noise * torch.randn(E, D, 3, device=self.device)
        self.raw_pos = torch.where(arrived.unsqueeze(-1), noisy, self.raw_pos)
        if cfg.kalman.enabled:
            # the laptop's filter: predict every tick, fuse the fix when one arrives; lost or
            # landed drones hold their last estimate (no odometry)
            self.kf.predict(dt, active=~self.lost & ~self.landed)
            self.kf.reset(self.lost & arrived)  # back from lost: restart from this fix
            accepted = self.kf.update(noisy, mask=arrived)
            self.pub_pos = torch.where(self.kf.ready.unsqueeze(-1), self.kf.p, self.pub_pos)
        else:
            self.pub_pos = self.raw_pos.clone()
            accepted = arrived
        if cfg.land_gate_enabled:
            arrived = accepted  # rejected outliers must not refresh heading or fix age
        yaw_fix = self._yaw_buf[:, :, -1] + cfg.yaw_fix_noise * torch.randn(E, D, device=self.device)
        self.pub_yaw = torch.where(arrived, yaw_fix, self.pub_yaw)
        self.fresh = arrived
        self.acquired |= arrived
        self.fix_age = torch.where(arrived, torch.zeros_like(self.fix_age), self.fix_age + dt)
        h = pos[..., 2] + cfg.height_noise * torch.randn_like(pos[..., 2])
        self.height_meas = torch.round(h * 10.0) / 10.0  # the state packet's h is in 0.1 m steps

        # lost the tags long enough: no position until a tag is in view again
        lost = (self.fix_age > cfg.lost_after) & ~self.landed
        self._lost_event = lost & ~self.lost
        self.lost = lost
        if cfg.kalman.enabled:
            self.kf.freeze(lost | self.landed)  # stop extrapolating: velocity 0

    # -- MDP --------------------------------------------------------------------

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        cfg = self.cfg
        self._update_feed()
        pos = self._positions()

        # takeoff done: the firmware levels off at its takeoff height and hands over to rc
        self.taking_off &= pos[..., 2] < self._takeoff_z
        # entering the waypoint: decided on the published position (what the controller sees) and
        # the filter's speed estimate (the laptop has it too; true speed without the filter).
        # At or below waypoint_max_speed it is reached (latched); faster, the episode fails
        self._wp_before = self.at_waypoint.clone()
        near_wp = torch.linalg.norm(self.pub_pos - (self._spots + self._wp_offset), dim=-1) < cfg.waypoint_radius
        self.est_speed = torch.linalg.norm(self.kf.v if cfg.kalman.enabled else self._stack("root_lin_vel_w"), dim=-1)
        slow = self.est_speed <= cfg.waypoint_max_speed
        # a just-restarted filter reads velocity 0: judge the speed only once it has settled
        trusted = (torch.sqrt(self.kf.c).amax(-1) < cfg.speed_trust_std) if cfg.kalman.enabled else torch.ones_like(slow)
        entry = ~self.at_waypoint & near_wp & self.fresh & ~self.landing & trusted
        self._waypoint_event = entry & slow
        self._fast_entry = entry & ~slow
        self._ep_fast_entry |= self._fast_entry.any(dim=-1)
        self.at_waypoint |= self._waypoint_event
        # touchdown: motors off; judged on the true touchdown point
        self._touchdown_event = self.landing & ~self.landed & (pos[..., 2] < cfg.touchdown_height)
        err = torch.linalg.norm((pos - self._spots)[..., :2], dim=-1)
        self.touchdown_err = torch.where(self._touchdown_event, err, self.touchdown_err)
        self.land_ok |= self._touchdown_event & (err < cfg.success_radius)
        self.landed |= self._touchdown_event
        all_landed = self.landed.all(dim=-1)
        self._all_event = all_landed & self.land_ok.all(dim=-1)

        tilt_cos = -self._stack("projected_gravity_b")[..., 2]
        m = cfg.wall_clearance
        crash = (
            ((pos[..., 2] < cfg.min_height) & ~self.landing & ~self.taking_off)  # land is the one way down
            | (pos[..., 2] > cfg.ceiling - m)
            | (pos[..., 0] < cfg.wall_x + m)
            | (pos[..., 0] > cfg.room_x_max - m)
            | (pos[..., 1] > cfg.side_wall_y - m)
            | (pos[..., 1] < cfg.room_y_min + m)
            | (tilt_cos < math.cos(cfg.max_tilt))
            | ~torch.isfinite(pos).all(dim=-1)
        )
        pair = self._pairwise_distance(pos)
        both_down = self.landed.unsqueeze(-1) & self.landed.unsqueeze(-2)
        collision = ((pair < self._pair_limit) & ~both_down).any(dim=-1)
        if cfg.collision_substeps:
            collision |= self._substep_collision
        self._violation = crash | collision
        self._crash, self._collision = crash, collision
        self._ep_collision |= collision.any(dim=-1)
        self._ep_crash |= crash.any(dim=-1)
        self._ep_fresh_steps += self.fresh.float().mean(-1)
        self._ep_losses += self._lost_event.float().sum(-1)
        if cfg.num_drones > 1:
            self._ep_min_sep = torch.minimum(self._ep_min_sep, pair.amin(dim=(-1, -2)))

        spot_err = torch.where(self.landed, self.touchdown_err, err)
        success = self._all_event
        done = self._violation.any(dim=-1) | (self._fast_entry.any(dim=-1) & cfg.terminate_fast_entry) | all_landed
        self._done_now = done
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        finished = done | time_out
        self.outcome_valid = finished
        steps = self.episode_length_buf.float().clamp(min=1.0)
        latch = lambda new, old: torch.where(finished, new, old)  # noqa: E731
        self.outcome_success = latch(success & ~self._violation.any(dim=-1), self.outcome_success)
        self.outcome_waypoints = latch(self.at_waypoint.float().sum(-1), self.outcome_waypoints)
        self.outcome_landed = latch(self.landed.float().sum(-1), self.outcome_landed)
        self.outcome_land_ok = latch(self.land_ok.float().sum(-1), self.outcome_land_ok)
        self.outcome_collision = latch(self._ep_collision, self.outcome_collision)
        self.outcome_crash = latch(self._ep_crash, self.outcome_crash)
        self.outcome_timeout = latch(time_out & ~done, self.outcome_timeout)
        self.outcome_fast_entry = latch(self._ep_fast_entry, self.outcome_fast_entry)
        self.outcome_time = latch(self.episode_length_buf.float() * self.step_dt, self.outcome_time)
        self.outcome_fresh_frac = latch(self._ep_fresh_steps / steps, self.outcome_fresh_frac)
        self.outcome_losses = latch(self._ep_losses, self.outcome_losses)
        self.outcome_gate_ready_frac = latch(self._ep_gate_ready / steps, self.outcome_gate_ready_frac)
        self.outcome_spot_err = latch(spot_err.mean(-1), self.outcome_spot_err)
        self.outcome_min_sep = latch(self._ep_min_sep, self.outcome_min_sep)
        self._finished = finished

        terminated = {agent: done for agent in self._agents}
        time_outs = {agent: time_out for agent in self._agents}
        return terminated, time_outs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        pos = self._positions()
        dist = torch.linalg.norm(pos - (self._spots + self._wp_offset), dim=-1)  # to the waypoint, 3D
        flying = (~self.landed & ~self.taking_off).float()  # the policy is flying it
        heading_in = (~self._wp_before & ~self.landing).float()  # waypoint phase only
        gain = self._prev_dist - dist  # metres closer this step (negative: moved away)
        gain = torch.where(gain > 0, gain, cfg.rew_away_ratio * gain)
        closer = cfg.rew_closer_scale * gain * flying * heading_in
        self._prev_dist = dist
        if cfg.reward_version >= 2:
            # Bounded state potential; landing advances toward a second potential level.
            # Its discounted sum telescopes, so a closed flight loop cannot farm progress.
            potential = self._potential(pos)
            closer = potential_delta(self._prev_potential, potential, self._done_now.unsqueeze(-1),
                                      cfg.reward_gamma, torch)
            self._prev_potential = potential
        # braking near tag 10: pays for each new low of the true speed inside the slow zone
        speed = torch.linalg.norm(self._stack("root_lin_vel_w"), dim=-1)
        in_zone = (torch.linalg.norm(pos[..., :2], dim=-1) < cfg.slow_zone_radius) & (flying * heading_in).bool()
        best = torch.where(in_zone & torch.isinf(self._best_speed), speed, self._best_speed)  # entering: start here
        slowdown = cfg.rew_slowdown * torch.where(in_zone, (best - speed).clamp(min=0.0), 0.0)
        self._best_speed = torch.where(in_zone, torch.minimum(best, speed), self._best_speed)
        over = (self.est_speed - cfg.waypoint_max_speed).clamp(min=0.0)

        sep = ((cfg.separation_distance - self._pairwise_distance(pos)) / cfg.separation_distance).clamp(min=0.0).sum(-1)
        action_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)
        not_done = (~self.land_ok).float()
        landing = cfg.rew_time * not_done
        # ended early (crash, contact, or every drone down with some off their spot): the
        # drones not landed on their spot pay the landing loss for the rest of the episode,
        # so ending an episode early never beats flying it
        remaining = (self.max_episode_length - 1 - self.episode_length_buf).clamp(min=0).float()
        landing = landing + cfg.rew_time * not_done * (self._done_now.float() * remaining).unsqueeze(-1)
        settling = torch.zeros_like(closer)
        if cfg.reward_version >= 2:
            xy_error = torch.linalg.norm((pos - self._spots)[..., :2], dim=-1)
            settling = ((cfg.rew_settle_position * xy_error + cfg.rew_settle_speed * speed)
                        * self.step_dt * self.at_waypoint.float() * flying * (~self.landing).float())
        touchdown_rewarded = self._touchdown_event & (self.land_ok if cfg.reward_version >= 2 else True)
        fast_entry_weight = cfg.rew_fast_entry_per_mps if cfg.reward_version == 1 else -5.0
        predictive = torch.zeros_like(flying)
        if cfg.rew_predictive_separation:
            predictive = predicted_separation_cost(
                pos, self._stack("root_lin_vel_w"), self.landed,
                cfg.separation_lookahead_s, cfg.separation_prediction_margin, torch)
        terms = {
            "closer_reward": closer,
            "waypoint_reward": cfg.rew_waypoint * self._waypoint_event.float(),
            "slowdown_reward": slowdown,
            "settling_loss": settling,
            "fast_entry_loss": fast_entry_weight * over * self._fast_entry.float(),
            "landing_loss": landing,
            "no_tag_loss": cfg.rew_no_fix * (~self.fresh).float() * flying,
            "crash_loss": cfg.rew_violation * (self._crash & ~self._collision).float(),
            "collision_loss": cfg.rew_collision * self._collision.float(),
            "separation_loss": cfg.rew_scale_separation * sep * self.step_dt,
            "predictive_separation_loss": cfg.rew_predictive_separation * predictive * self.step_dt,
            "action_rate_loss": cfg.rew_scale_action_rate * action_rate * self.step_dt,
            "early_land_loss": cfg.rew_early_land * self._early_land.float(),
            "land_reward": cfg.rew_land * touchdown_rewarded.float(),
            "formation_reward": cfg.rew_all_landed * self._all_event.float().unsqueeze(-1).expand_as(flying),
        }
        reward = torch.nan_to_num(sum(terms.values()), nan=cfg.rew_violation)
        terms["total"] = reward
        for key, term in terms.items():
            self._ep_rew[key] += term.sum(-1)
            # rewards are computed after dones: latch this episode's totals now, the reset follows
            self.outcome_rew[key] = torch.where(self._finished, self._ep_rew[key], self.outcome_rew[key])
        return {agent: reward[:, i] for i, agent in enumerate(self._agents)}

    def _potential(self, pos):
        cfg = self.cfg
        dist = torch.linalg.norm(pos - (self._spots + self._wp_offset), dim=-1)
        approach = cfg.rew_potential_scale * torch.exp(-dist / 2.0)
        peak = cfg.rew_potential_scale
        if cfg.rew_precision_potential:
            speed = torch.linalg.norm(self._stack("root_lin_vel_w")[..., :2], dim=-1)
            approach += precision_potential(dist, speed, cfg.rew_precision_potential,
                                            cfg.precision_position_scale, cfg.precision_speed_scale, torch)
            peak += cfg.rew_precision_potential
        descent = 1.0 + (1.0 - pos[..., 2] / cfg.waypoint_height).clamp(0.0, 1.0)
        return torch.where(self.landing, peak * descent, approach)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Each drone's block of the central observation (Sb3CentralVecEnv concatenates them)."""
        cfg = self.cfg
        p = self.pub_pos
        c, s = torch.cos(self.pub_yaw), torch.sin(self.pub_yaw)

        def heading_frame(target):
            rel = target - p
            return torch.stack([c * rel[..., 0] + s * rel[..., 1], s * rel[..., 0] - c * rel[..., 1], rel[..., 2]], -1)

        to_wp = heading_frame(self._spots + self._wp_offset)
        to_spot = heading_frame(self._spots)
        to10 = self._tag10 - p
        bearing = _wrap(torch.atan2(to10[..., 1], to10[..., 0]) - self.pub_yaw)  # + = tag 10 to the left
        age = self.fix_age.clamp(max=cfg.max_fix_age)
        block = torch.cat(
            [
                p,
                torch.stack([c, s], -1),
                self.fresh.float().unsqueeze(-1),
                age.unsqueeze(-1),
                self.lost.float().unsqueeze(-1),
                to_wp,
                to_spot,
                self.at_waypoint.float().unsqueeze(-1),
                torch.stack([torch.sin(bearing), torch.cos(bearing)], -1),
                self.height_meas.unsqueeze(-1),
                self._actions,
                (self.landing | self.landed).float().unsqueeze(-1),
            ] + ([self.taking_off.float().unsqueeze(-1)] if cfg.start_on_floor else []),
            dim=-1,
        )
        if cfg.peer_observations:
            if not cfg.kalman.enabled:
                raise ValueError("Peer observations require estimated Kalman velocity and uncertainty")
            block = torch.cat((block, peer_features(p, self.kf.v, self.pub_yaw, self.kf.a, self.kf.c, torch,
                                                  sort_neighbors=cfg.peer_sort_by_distance,
                                                  order_version=cfg.peer_order_version)), -1)
        block = torch.nan_to_num(block, nan=0.0, posinf=10.0, neginf=-10.0)
        return {agent: block[:, i] for i, agent in enumerate(self._agents)}

    # -- reset ------------------------------------------------------------------

    def _sample_starts(self, n: int) -> torch.Tensor:
        """(n, D, 2) start XY: inside the box, 6-7 m from tag 10, >= spawn_min_separation apart.
        Vectorized rejection over K candidate sets with a fixed valid fallback."""
        cfg = self.cfg
        D, K, dev = cfg.num_drones, 64, self.device
        cx, cy = cfg.spawn_box_center
        half = 0.5 * cfg.spawn_box_size
        xy = torch.stack([cx + half * (2 * torch.rand(n, K, D, device=dev) - 1),
                          cy + half * (2 * torch.rand(n, K, D, device=dev) - 1)], -1)
        r = torch.linalg.norm(xy, dim=-1)
        in_range = ((r >= cfg.spawn_range[0]) & (r <= cfg.spawn_range[1])).all(-1)
        d = torch.cdist(xy.reshape(n * K, D, 2), xy.reshape(n * K, D, 2)).reshape(n, K, D, D)
        d = d + 1e3 * torch.eye(D, device=dev)
        apart = d.amin(dim=(-1, -2)) >= cfg.spawn_min_separation
        valid = in_range & apart
        pick = torch.argmax(valid.float(), dim=1)
        chosen = xy[torch.arange(n, device=dev), pick]
        k = torch.arange(D, device=dev, dtype=torch.float32) - 0.5 * (D - 1)
        fallback = torch.stack([6.3 + 0.4 * (torch.arange(D, device=dev) % 2), 0.55 * k], -1)
        return torch.where(valid.any(1).view(n, 1, 1), chosen, fallback.expand(n, D, 2))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._drones[0]._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._log_outcomes(env_ids)
        super()._reset_idx(env_ids)
        cfg = self.cfg
        n, D, dev = len(env_ids), cfg.num_drones, self.device
        if n == self.num_envs and self.num_envs > 1:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        def U(lo, hi, *shape):
            return lo + (hi - lo) * torch.rand(*shape, device=dev)

        xy = self._sample_starts(n)
        floor_start = cfg.start_on_floor and cfg.takeoff_at_reset
        z = torch.full((n, D), cfg.rest_height, device=dev) if floor_start else U(*cfg.spawn_height, n, D)
        start = torch.cat([xy, z.unsqueeze(-1)], -1)
        if cfg.random_assignment:  # any drone may get any spot (D of the spot_count)
            order = torch.argsort(torch.rand(n, cfg.spot_count, device=dev), dim=-1)[:, :D]
        else:  # drone k-th from the left (smallest Y) takes spot k: no crossing paths
            order = torch.argsort(torch.argsort(xy[..., 1], dim=-1), dim=-1)
        if cfg.fixed_spot_index >= 0:
            if D != 1 or cfg.fixed_spot_index >= cfg.spot_count:
                raise ValueError("fixed_spot_index requires one drone and an existing landing spot")
            order[:] = cfg.fixed_spot_index
        self._spots[env_ids] = self._spot_table[order]
        if cfg.spawn_near_target_radius > 0:
            if D != 1:
                raise ValueError("Near-target curriculum requires a single drone")
            xy = self._spots[env_ids, :, :2] + U(-cfg.spawn_near_target_radius, cfg.spawn_near_target_radius, n, D, 2)
            start = torch.cat([xy, z.unsqueeze(-1)], -1)
        xy, z, takeoff, kind = mix_starts(cfg, xy, z, self._spots[env_ids], floor_start)
        start = torch.cat([xy, z.unsqueeze(-1)], -1)
        self._start_kind[env_ids] = kind
        self._start_positions[env_ids] = start
        self._pair_limit[env_ids] = torch.minimum(
            torch.full_like(self._pair_limit[env_ids], cfg.collision_distance),
            self._pairwise_distance(start) - cfg.start_contact_margin)
        # facing tag 10 +- spawn_yaw_noise, or (spawn_search_prob) turned away: no tag in view
        search = torch.rand(n, D, device=dev) < cfg.spawn_search_prob
        side = torch.where(torch.rand(n, D, device=dev) < 0.5, -1.0, 1.0)
        offset = torch.where(search, side * U(*cfg.spawn_search_yaw, n, D), U(-cfg.spawn_yaw_noise, cfg.spawn_yaw_noise, n, D))
        yaw = torch.atan2(-xy[..., 1], -xy[..., 0]) + offset
        zeros = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(zeros.flatten(), zeros.flatten(), yaw.flatten()).view(n, D, 4)

        rows = (env_ids.unsqueeze(-1) * D + torch.arange(D, device=dev)).flatten()
        self.controller.reset(rows, randomize=True)
        self.rc.randomize(rows, enabled=cfg.dynamics.domain_randomization)
        mass_scale = self.controller.mass_scale.view(self.num_envs, D)
        env_ids_cpu = env_ids.cpu()
        origins = self.scene.env_origins[env_ids]
        for i, drone in enumerate(self._drones):
            state = drone.data.default_root_state[env_ids].clone()
            state[:, :3] = origins + start[:, i]
            state[:, 3:7] = quat[:, i]
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            masses = drone.root_physx_view.get_masses()
            masses[env_ids_cpu] = self._default_masses[i][env_ids_cpu] * mass_scale[env_ids, i].cpu().unsqueeze(-1)
            drone.root_physx_view.set_masses(masses, env_ids_cpu)

        # nothing published until the first sighting
        self.pub_pos[env_ids] = start
        self.raw_pos[env_ids] = start
        reset_mask = torch.zeros(self.num_envs, D, dtype=torch.bool, device=dev)
        reset_mask[env_ids] = True
        self.kf.reset(reset_mask)
        self.pub_yaw[env_ids] = yaw
        self._fix_ok_buf[env_ids] = False
        self.fresh[env_ids] = False
        self.acquired[env_ids] = False
        self.fix_age[env_ids] = 0.0
        if cfg.reward_version >= 2:
            self.height_meas[env_ids] = torch.round(z * 10.0) / 10.0
        self.lost[env_ids] = False
        self._link_loss[env_ids] = U(*cfg.link_loss_range, n, D)
        self._fix_bias[env_ids] = cfg.fix_bias * torch.randn(n, D, 3, device=dev)
        for v in self._ep_rew.values():
            v[env_ids] = 0.0
        self._delivered[env_ids] = 0.0

        self._actions[env_ids] = 0.0
        self._substep_collision[env_ids] = False
        self._prev_actions[env_ids] = 0.0
        self.last_rc[env_ids] = 0.0
        self.at_waypoint[env_ids] = False
        self.landing[env_ids] = False
        self.taking_off[env_ids] = takeoff
        self._takeoff_z[env_ids] = U(*cfg.takeoff_height, n, D)
        self.landed[env_ids] = False
        self.land_ok[env_ids] = False
        self.touchdown_err[env_ids] = 0.0
        self._land_votes[env_ids] = 0
        self.land_dwell[env_ids] = 0.0
        self.land_ready[env_ids] = False
        self._early_land[env_ids] = False
        self._prev_dist[env_ids] = torch.linalg.norm(start - (self._spots[env_ids] + self._wp_offset), dim=-1)
        self._prev_potential[env_ids] = cfg.rew_potential_scale * torch.exp(-self._prev_dist[env_ids] / 2.0)
        if cfg.rew_precision_potential:
            self._prev_potential[env_ids] += precision_potential(
                self._prev_dist[env_ids], torch.zeros_like(self._prev_dist[env_ids]),
                cfg.rew_precision_potential, cfg.precision_position_scale, cfg.precision_speed_scale, torch)
        self._ep_collision[env_ids] = False
        self._ep_crash[env_ids] = False
        self._ep_fast_entry[env_ids] = False
        self._best_speed[env_ids] = float("inf")
        self._ep_fresh_steps[env_ids] = 0.0
        self._ep_losses[env_ids] = 0.0
        self._ep_gate_ready[env_ids] = 0.0
        self._ep_min_sep[env_ids] = float("inf")

    def _log_outcomes(self, env_ids: torch.Tensor) -> None:
        finished = env_ids[self.outcome_valid[env_ids]]
        self.extras["episode_records"] = []
        if len(finished) == 0:
            self.extras["log"] = {}
            return
        # Capture individual outcomes and assignments BEFORE the automatic reset.
        fields = {"success_rate": "success", "fast_entry_rate": "fast_entry",
                  "drones_reached_waypoint": "waypoints", "drones_landed": "landed",
                  "drones_landed_on_spot": "land_ok", "collision_rate": "collision",
                  "crash_rate": "crash", "timeout_rate": "timeout", "duration_s": "time", "tag_fix_frac": "fresh_frac",
                  "tag_losses": "losses", "spot_err_m": "spot_err", "land_gate_ready_frac": "gate_ready_frac"}
        if self.cfg.num_drones > 1:
            fields["min_separation_m"] = "min_sep"
        values = {key: getattr(self, "outcome_" + name)[finished].detach().cpu().tolist()
                  for key, name in fields.items()}
        spots = self._spots[finished].detach().cpu().tolist()
        landed = self.landed[finished].detach().cpu().tolist()
        ok = self.land_ok[finished].detach().cpu().tolist()
        errors = self.touchdown_err[finished].detach().cpu().tolist()
        start_kinds = self._start_kind[finished].detach().cpu().tolist()
        starts = self._start_positions[finished].detach().cpu().tolist()
        terminal = {}
        if self.cfg.log_terminal_state:
            terminal = {key: value[finished].detach().cpu().tolist() for key, value in {
                "position": self._positions(), "velocity": self._stack("root_lin_vel_w"),
                "estimated_position": self.pub_pos, "estimated_velocity": self.kf.v,
                "fix_age": self.fix_age, "actions": self._actions,
                "taking_off": self.taking_off, "takeoff_target_height": self._takeoff_z,
                "landing": self.landing, "landed": self.landed,
                "position_variance": self.kf.a, "velocity_variance": self.kf.c,
            }.items()}
        for j, env_id in enumerate(finished.cpu().tolist()):
            metrics = {"Episode/" + key: float(value[j]) for key, value in values.items()}
            self.extras["episode_records"].append({
                "env_id": env_id, "metrics": metrics,
                "start_kind": ("configured", "short_approach", "hover_landing")[start_kinds[j]],
                "start_positions": starts[j],
                **({"terminal_state": {key: value[j] for key, value in terminal.items()}} if terminal else {}),
                "drones": [{"target_index": round(spot[1] / self.cfg.spot_spacing + (self.cfg.spot_count - 1) / 2),
                            "target": spot, "landed": landed[j][d], "land_ok": ok[j][d],
                            "touchdown_error_m": errors[j][d]} for d, spot in enumerate(spots[j])],
            })
        self.extras["log"] = {
            "Episode/success_rate": self.outcome_success[finished].float().mean().item(),
            "Episode/fast_entry_rate": self.outcome_fast_entry[finished].float().mean().item(),
            "Episode/drones_reached_waypoint": self.outcome_waypoints[finished].mean().item(),
            # share of drones that got inside their waypoint this episode, 0-100
            "Episode/waypoint_reached_pct": 100.0 * (self.outcome_waypoints[finished] / self.cfg.num_drones).mean().item(),
            "Episode/drones_landed": self.outcome_landed[finished].mean().item(),
            "Episode/drones_landed_on_spot": self.outcome_land_ok[finished].mean().item(),
            "Episode/collision_rate": self.outcome_collision[finished].float().mean().item(),
            "Episode/crash_rate": self.outcome_crash[finished].float().mean().item(),
            "Episode/timeout_rate": self.outcome_timeout[finished].float().mean().item(),
            "Episode/duration_s": self.outcome_time[finished].mean().item(),
            "Episode/tag_fix_frac": self.outcome_fresh_frac[finished].mean().item(),
            "Episode/tag_losses": self.outcome_losses[finished].mean().item(),
            "Episode/land_gate_ready_frac": self.outcome_gate_ready_frac[finished].mean().item(),
            "Episode/spot_err_m": self.outcome_spot_err[finished].mean().item(),
            **({"Episode/min_separation_m": self.outcome_min_sep[finished].mean().item()} if self.cfg.num_drones > 1 else {}),
            # each reward term: average per drone, per episode (summed over the episode's steps)
            **{f"Rewards/avg_{k}_per_episode": (self.outcome_rew[k][finished] / self.cfg.num_drones).mean().item()
               for k in self.REWARD_TERMS},
            "Episode/count": float(len(finished)),
        }

"""Configuration for the room task: four Tellos form up in front of tag 10 in the mapped room.

World frame (see ``action_space_kit.tags.room_map``): tag 10's wall is X = 0, the room is
X > 0, Z is up from the floor, Y = the tag's "right" (seen facing the tag).

Scenario: the four drones start hovering inside a 2 m x 2 m box 6-7 m from tag 10, fly to
four landing spots on the floor 1 m in front of the tag-10 wall, 0.5 m apart, and send the
SDK ``land`` command over their spot. Start
positions and the drone-to-spot assignment are random every episode, so paths can cross. Each
drone must get there fast, never touch another drone, and keep AprilTags in view: its
position comes only from tag sightings, and a drone that loses them stops publishing a
position until it turns back to a tag. The policy is trained end to end on that positional
data alone: no scripted search, no hold, no hand-written controller anywhere.
"""

from __future__ import annotations

import math

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from action_space_kit.control import RcActionCfg, TelloDynamicsCfg
from action_space_kit.tags.kalman import KalmanCfg
from action_space_kit.tags.sensing import TagSensorCfg

NUM_DRONES = 4

# One central controller (the laptop) flies all four: its observation is the four drones'
# blocks below, concatenated (4 x 24 = 96), and it outputs all four drones' 5 commands.
# Per drone, all from its positional feed (core/positional.py) as the laptop receives it:
#   last fix position (3), heading cos/sin (2), fix this tick (1), seconds since fix (1),
#   lost (1), waypoint relative in its heading frame fwd/right/up (3), landing spot relative
#   fwd/right/up (3), waypoint reached (1), bearing to tag 10 sin/cos (2), Tello height h (1),
#   previous action (5), landing or landed (1)
_OBS_DIM = 3 + 2 + 1 + 1 + 1 + 3 + 3 + 1 + 2 + 1 + 5 + 1

SIM_RATE_HZ = 500
# The flight engine sends one rc per control tick. Measured in flight (recordings/
# 027_TELLO-1.train.jsonl, 2026-10-08, one row per rc sent): 13.43 Hz, period median 70 ms
# (80 ms while cruising), p95 90 ms, range 55-110 ms: every tick runs a full-frame tag
# detection before its stop.wait(0.05). The nearest whole number of 2 ms physics steps is
# 37 -> 74 ms; command_jitter spreads the rest.
COMMAND_RATE_HZ = SIM_RATE_HZ / 37  # 13.51 Hz


@configclass
class TelloRoomEnvCfg(DirectMARLEnvCfg):
    """Four Tellos fly from a start box 6-7 m out to their spots 1 m in front of tag 10."""

    command_rate_hz = COMMAND_RATE_HZ
    """One policy step = one rc packet. Physics steps per command are derived from this (the
    env enforces it), so the sim can only act as often as the real engine can send."""
    decimation = round(SIM_RATE_HZ / COMMAND_RATE_HZ)
    command_jitter = 0.016
    """[s] each rc lands up to this late inside its step (per drone, per step, uniform), so
    the gaps between commands spread over ~58-90 ms like the real loop's."""
    episode_length_s = 75.0  # ~5.5 m at rc 25 (~0.18 m/s) is ~30 s, plus turns and searches
    debug_vis = False
    peer_observations = False  # append estimated velocities, uncertainty and relative peer state
    peer_sort_by_distance = False  # shared-policy experiments use nearest-first peer slots
    peer_order_version = 2  # deterministic millimeter-distance keys; old snapshots restore version 1
    log_terminal_state = False  # diagnostic records only; never changes observations or physics
    collision_substeps = True  # detect proximity violations at every 2 ms physics step

    # the real Tello commands: the four rc channels [right, forward, up, yaw_cw] in [-1, 1] ->
    # integer rc -100..100 (the full SDK range), plus the SDK ``land`` command (5th output >
    # land_threshold for land_confirm_steps in a row)
    possible_agents = [f"drone_{i}" for i in range(NUM_DRONES)]
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(5,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
    state_space = -1

    sim: SimulationCfg = SimulationCfg(dt=1 / SIM_RATE_HZ, render_interval=decimation)
    # every env in the same room (env_spacing 0): the GUI shows the whole batch at once. Envs
    # never touch: PhysX filters contacts between envs (GPU env IDs; filter_collisions on CPU),
    # and every reward / crash / collision uses each env's own positions. Measured at 512 envs x
    # 4 drones: no slower than the 14 m grid. env.scene.env_spacing=14.0 lays them out on a grid.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=0.0, replicate_physics=True)
    # GUI camera: env 0, behind and above the start box looking at tag 10 (record_room.py's
    # "overview" view). Isaac Lab's default looks at the world origin from 13 m away.
    viewer: ViewerCfg = ViewerCfg(eye=(9.6, -3.4, 3.0), lookat=(3.2, 0.0, 0.6), origin_type="env", env_index=0)

    # Every drone flies its own firmware: drawn per drone, per episode (TelloDynamicsCfg
    # defaults, the ese651 ranges): thrust-to-weight x0.95-1.05, rotor drag k_aero_xy/z
    # x0.5-2.0, body-rate PID kp/ki x0.85-1.15 and kd x0.7-1.3 (roll/pitch and yaw), plus
    # velocity/attitude loop gains x0.85-1.15, motor lag x0.8-1.2, mass x0.95-1.05 and
    # 0-2 steps of command latency, around the motion fit.
    dynamics: TelloDynamicsCfg = TelloDynamicsCfg(
        twr_range=(0.95, 1.05),
        k_aero_range=(0.5, 2.0),
        kp_omega_range=(0.85, 1.15),
        ki_omega_range=(0.85, 1.15),
        kd_omega_range=(0.7, 1.3),
    )
    # full SDK range on every channel; speeds beyond the logged sticks extrapolate the fitted gains
    rc: RcActionCfg = RcActionCfg(rc_limit=(100, 100, 100, 100), use_envelope=False)
    use_motion_fit = True
    """Retune ``dynamics`` (max speeds, velocity-loop gains, latency) from ``rc.fit_path``."""
    # Tello H.264 stream + decode + detect: ~0.2 s = 3 steps of 74 ms
    tag_sensor: TagSensorCfg = TagSensorCfg(video_delay_steps=3)

    # -- room (from drone_deployment_infra/config/map.yaml; only the floor is hand-measured)
    num_drones = NUM_DRONES
    room_map_path = ""
    room_tags_path = ""
    tag10_height = 1.22  # [m] centre of tag 10 above the floor: measured ~48 in (2026-10-14)
    wall_x = -0.10  # [m] tag-10 wall surface (tags 8/9 map ~8 cm behind tag 10)
    side_wall_y = 1.70  # [m] tag-5 wall surface
    room_y_min = -2.0  # [m] left wall (facing tag 10): measured 2 m left of tag 10's centre
    room_x_max = 8.5  # [m] not mapped; the start box reaches 7.5 m
    ceiling = 2.74  # [m] 9 ft
    wall_clearance = 0.2  # [m] closer than this to a wall counts as a crash

    # -- start: a 2 m x 2 m box on tag 10's normal, every drone 6-7 m from tag 10
    spawn_box_center = (6.5, 0.0)  # [m] X, Y
    spawn_box_size = 2.0  # [m]
    spawn_range = (6.0, 7.0)  # [m] horizontal distance to tag 10
    spawn_min_separation = 0.2  # [m]
    spawn_near_target_radius = 0.0  # single-drone landing curriculum only; 0 uses the start box
    # Optional training replay: remaining episodes use the configured start box unchanged.
    # Full-room evaluation must set both fractions to zero.
    spawn_short_fraction = 0.0
    spawn_short_x_range = (1.5, 5.5)
    spawn_short_y_radius = 0.4
    spawn_hover_fraction = 0.0
    spawn_hover_radius = 0.10
    spawn_height = (0.9, 1.4)  # [m] hovering after takeoff (start_on_floor = False)
    start_on_floor = False
    takeoff_at_reset = True  # False: curriculum starts hovering, preserving the observation layout
    """True: every drone starts on the floor, motors off, and the episode opens with the SDK
    ``takeoff`` the flight engine sends before any rc: the firmware climbs to takeoff_height on
    its own (sticks ignored, like land), then the policy flies. The observation gains a
    taking-off flag. False: the drones start hovering at spawn_height (takeoff already done)."""
    rest_height = 0.025  # [m] body centre of a Tello on the floor (collision box 4.1 cm tall)
    takeoff_height = (0.8, 1.0)  # [m] where the firmware's takeoff stops, per drone, per episode
    takeoff_speed = 0.5  # [m/s] firmware climb during takeoff
    spawn_yaw_noise = 15.0 * math.pi / 180.0  # [rad] around facing tag 10
    spawn_search_prob = 0.3
    """Per drone, per episode: chance it starts turned away from the tags (yaw spawn_search_yaw
    off facing tag 10, either side), so no tag is in view and it must turn to find one. Every
    mapped tag is on or near the tag-10 end of the room, within ~25 deg of tag 10 from the box."""
    spawn_search_yaw = (75.0 * math.pi / 180.0, math.pi)  # [rad] |offset| from facing tag 10

    # -- goal: four landing spots on the floor 1 m in front of the tag-10 wall, 0.5 m apart
    spot_distance = 1.0  # [m] from the tag-10 plane
    spot_spacing = 0.5  # [m] between neighbouring spots
    spot_count = NUM_DRONES
    """Landing spots in the row; each drone gets a different one (num_drones <= spot_count)."""
    random_assignment = True
    fixed_spot_index = -1  # single-drone evaluation only; -1 samples all pads
    """Which drone gets which spot: a random permutation every episode (paths may cross).
    False: the drone k-th from the left takes spot k."""
    # -- waypoint: aligned with tag 10's height above each landing spot. Until its estimate is inside
    # its radius, it is rewarded for closing on the waypoint (not the spot); from then on only
    # for landing (land is gated until then; landing on the spot: formation) on top of the penalties.
    waypoint_height = 1.22  # [m] align with measured tag-10 centre so its 571.5 mm square fits the camera
    waypoint_radius = 0.3  # [m]
    waypoint_max_speed = 0.3
    """[m/s] entering the waypoint counts only at or below this speed (the Kalman filter's speed
    estimate, which the laptop has too; the true speed with env.kalman.enabled=False): flights
    032-034 crossed the bubble at ~2 m/s and kept going. Faster: the episode fails
    (``rew_fast_entry_per_mps``)."""
    speed_trust_std = 0.35
    """[m/s] the filter's speed counts only once its velocity std (sqrt of the covariance) is
    below this: a filter restart (back from lost, outliers) sets velocity 0 with std 0.5, and
    ~8 fixes (~0.6 s) bring it to ~0.3. Until then an entry is neither passed nor failed."""
    slow_zone_radius = 2.0
    """[m] horizontal distance to tag 10 inside which slowing down pays (``rew_slowdown``)."""
    success_radius = 0.3  # [m] true touchdown point within this of the spot: landed on it
    # success: all four landed on their spots

    # -- land: the SDK command; the firmware descends and stops the motors at touchdown
    land_threshold = 0.5  # 5th action above this ...
    land_confirm_steps = 3  # ... this many steps in a row (~0.2 s) sends land
    auto_land = False
    """True: request landing automatically when the current-state dwell gate passes.
    False (default): require the policy's confirmed request AND the same gate.
    Legacy replay may disable land_gate_enabled to recover historical behavior."""
    land_speed = 0.4  # [m/s] firmware descent during land
    land_gate_enabled = True
    land_max_xy_error = 0.15
    land_max_horizontal_speed = 0.10
    land_height_tolerance = 0.20
    land_max_position_std = 0.12
    land_max_velocity_std = 0.35
    land_max_observation_age = 0.35  # includes the video pipeline's delay
    land_dwell_s = 0.5
    terminate_fast_entry = False  # allow braking/recovery; still log fast entries
    touchdown_height = 0.08  # [m] body centre below this: down, motors off

    # -- localization: positions come only from tag sightings (core/positional.py)
    lost_after = 0.4  # [s] without a fix the drone counts as lost (SEARCH_AFTER_S): no position
    yaw_fix_noise = 2.0 * math.pi / 180.0  # [rad] heading from a tag pose
    height_noise = 0.02  # [m] the state packet's h (always available)
    max_fix_age = 5.0  # [s] obs clip

    # -- positional noise on top of the camera model (core/positional.py output)
    fix_noise = 0.03  # [m] per fix, per axis
    fix_bias = 0.03  # [m] per drone, per episode, per axis (calibration / map error)
    # the laptop's Kalman filter (drone_deployment_infra/core/positional_kalman_filter.py, the
    # same file) smooths the fixes; the observation's position is its estimate. enabled=False:
    # the raw fix, as core/positional.py alone gives it.
    kalman: KalmanCfg = KalmanCfg()

    # -- WiFi link: per drone, per episode loss rate. A lost rc packet leaves the Tello flying
    # the last rc it received; a lost video frame means no tag fix that tick. The 27 flight
    # reports (26.9 min of video) lost 5.8% of frames overall, 3-19% per flight.
    link_loss_range = (0.0, 0.15)

    # -- safety
    min_height = 0.2
    max_tilt = 60.0 * math.pi / 180.0
    collision_distance = 0.25  # [m] centre to centre (props span ~0.18 m): ends the episode
    start_contact_margin = 0.03
    """[m] Drones may spawn closer than collision_distance (spawn_min_separation 0.2 m). Such a pair
    collides only below its start distance minus this margin, so a start is never a collision."""
    separation_distance = 0.5  # [m] spots are this far apart; closer is penalized

    # -- rewards (per drone, per policy step)
    reward_version = 2
    reward_gamma = 0.995  # train.py verifies this equals PPO gamma
    rew_potential_scale = 40.0
    rew_precision_potential = 0.0  # opt-in; replaces milestone-triggered settling costs in new runs
    precision_position_scale = 0.20
    precision_speed_scale = 0.15
    rew_settle_position = -2.0  # per metre per second after reaching the waypoint
    rew_settle_speed = -1.0  # per m/s per second after reaching the waypoint
    rew_closer_scale = 100.0  # legacy reward_version=1 only
    rew_away_ratio = 1.0  # legacy only; version 2 uses discounted potential shaping
    rew_waypoint = 25.0
    rew_fast_entry_per_mps = -500.0 / 0.3  # entering the waypoint faster than waypoint_max_speed: per m/s
    #                  over it (0.6 m/s: -500), and the episode ends (fails)
    rew_slowdown = 0.0  # version 2 uses continuous position/speed settling costs
    #                  (true speed; until the waypoint): pays for braking, can't be farmed by re-speeding
    rew_time = -0.5  # every step a drone is not landed on its spot ("landing loss"); a drone
    #                  that ends the episode early (crash, landed off the spot) pays the rest
    rew_land = 1000.0  # version 2: only an accurate touchdown earns this
    rew_early_land = -5.0  # confirmed requests before the waypoint; later requests wait for the gate
    rew_all_landed = 200.0  # to every drone, when all four landed on their spots (success)
    rew_no_fix = -0.5  # every step a flying drone has no AprilTag in view
    rew_violation = -100.0  # crash, wall, ceiling, flipping, leaving the room ("crash loss")
    rew_collision = -1000.0  # to each drone closer than collision_distance to another ("collision loss")
    rew_scale_separation = -20.0  # per second, scaled by how far inside separation_distance
    rew_predictive_separation = 0.0  # opt-in per-second cost for predicted close approaches
    separation_lookahead_s = 1.5
    separation_prediction_margin = 0.5
    rew_scale_action_rate = -0.05  # per second

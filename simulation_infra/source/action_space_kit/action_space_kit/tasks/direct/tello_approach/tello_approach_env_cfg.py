"""Configuration for the approach task: 4 Tellos enter a base's airspace through assigned gates."""

from __future__ import annotations

import gymnasium as gym
import math

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from action_space_kit.control import TelloDynamicsCfg

NUM_DRONES = 4
FT = 0.3048  # [m]

# obs per drone: own pos (3) + own quat (4) + others' pos (3 * 3) + others' quat (3 * 4)
#              + base pos (3) + own lin vel (3) + own waypoint (3) + gate passed (1)
#              + own landing spot (3)
_OBS_DIM = 3 + 4 + 3 * (NUM_DRONES - 1) + 4 * (NUM_DRONES - 1) + 3 + 3 + 3 + 1 + 3

SIM_RATE_HZ = 500
POLICY_RATE_HZ = 20


@configclass
class TelloApproachEnvCfg(DirectMARLEnvCfg):
    """Four Tellos fly from ~30 ft out to a base at the env origin, in a room with a 9 ft ceiling.

    Around the base is a keep-out sphere. Each drone owns one approach vector out of the
    base; where it crosses the sphere is that drone's waypoint (gate). A drone may only
    cross into the sphere after passing within ``gate_radius`` of its own waypoint. It then
    flies to its own designated landing spot inside the ``goal_radius`` touchdown zone, is
    sent ``land`` there, and descends; the episode succeeds when all four have touched down.
    Drones may touch only once both are landing; any other contact is a collision.

    All values are SI; the scenario was specified in feet, so ``FT`` converts.
    """

    # env: policy at 20 Hz (a realistic ``rc`` rate over WiFi), physics and rate PID at 500 Hz
    decimation = SIM_RATE_HZ // POLICY_RATE_HZ
    episode_length_s = 25.0
    debug_vis = False  # scene markers (keep-out shell, gates, touchdown zone, pads); the recorder and play.py turn it on

    # every drone uses the four SDK rc channels [vx, vy, vz, yaw_rate] in [-1, 1]
    possible_agents = [f"drone_{i}" for i in range(NUM_DRONES)]
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
    state_space = -1

    sim: SimulationCfg = SimulationCfg(dt=1 / SIM_RATE_HZ, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=28.0, replicate_physics=True)

    # Tello firmware + rotor model with ese651-style gain randomization
    dynamics: TelloDynamicsCfg = TelloDynamicsCfg()

    # -- scenario geometry
    num_drones = NUM_DRONES
    spawn_distance = 30.0 * FT  # [m] base to centre of the spawn square
    spawn_square = 5.0 * FT  # [m] side of the spawn square
    spawn_azimuth_range = math.pi  # [rad] spawn direction ~ U(-range, range) around the base; 0 = always +x
    spawn_line_angle = 15.0 * math.pi / 180.0  # [rad] line tilt from perpendicular-to-base
    spawn_spacing = (0.35, 0.45)  # [m] between neighbours on the line; four must fit the 5 ft square
    spawn_height = (0.8, 1.2)  # [m] Tello takeoff hover height
    spawn_yaw_noise = 15.0 * math.pi / 180.0  # [rad] around facing the base

    keepout_radius = 3.0  # [m] ~10 ft; must stay below the nearest spawn (~8.4 m)
    approach_elevation = (20.0 * math.pi / 180.0, 40.0 * math.pi / 180.0)  # [rad] above the floor: gates 1.0-1.9 m (3.4-6.3 ft)
    approach_azimuth_spread = 45.0 * math.pi / 180.0  # [rad] around the spawn direction; math.pi = any side, behind the base too
    approach_min_separation = 10.0 * math.pi / 180.0  # [rad] pairwise angle between approach vectors
    gate_radius = 0.4  # [m] passing this close to one's waypoint opens the sphere
    goal_radius = 1.0  # [m] touchdown zone around the base; every landing spot lies inside it
    near_base_radius = 3.0 * FT  # [m] reported only: share of drones that got this close to the base

    # -- designated landing spots: one per drone, on the ground inside the touchdown zone.
    # Spot k is the k-th from the left seen from the spawn side, like gate k, so paths
    # don't cross. At 0.6 m out and 40 deg apart, neighbouring spots are 0.41 m apart.
    landing_spot_radius = 0.6  # [m] from the base, horizontally
    landing_spot_spacing = 40.0 * math.pi / 180.0  # [rad] between neighbouring spots
    landing_hover_height = 0.5  # [m] the drone must reach this point above its spot ...
    landing_capture_radius = 0.25  # [m] ... to within this distance, then it is sent ``land``

    # -- safety / termination
    min_height = 0.2  # [m]
    max_height = 8.0 * FT  # [m] 1 ft under the 9 ft ceiling
    max_xy = 13.0  # [m] from the base
    max_tilt = 60.0 * math.pi / 180.0  # [rad]
    collision_distance = 0.15  # [m] drone-drone, centre to centre
    separation_distance = 0.3  # [m] below this the separation penalty kicks in (spawn spacing is >= 0.35)
    obs_clip_pos = 13.0  # [m]

    # -- landing: inside the inner sphere a drone gets the SDK ``land`` command, like a real Tello
    land_speed = 0.5  # [m/s] firmware descent rate during ``land``
    touchdown_height = 0.08  # [m] below this (body centre) the drone is down and its motors stop

    # -- rewards (per drone, per policy step)
    # Progress is a sign, not a distance: +1 for any step that ends closer to the current
    # target (own waypoint, then the base) than the last one, -0.5 for any that doesn't.
    rew_closer = 1.0
    rew_not_closer = -0.5
    rew_time = -0.05  # every policy step until the drone has arrived: finish sooner
    rew_gate = 100.0  # reaching one's own waypoint
    rew_goal = 100.0  # reaching its landing spot inside the touchdown zone (then it lands)
    rew_scale_entry_speed = 0.0  # per m/s of inward (towards the base) speed on the step the gate is reached
    # Safety terms on top of that structure; set any of them to 0 to switch it off.
    rew_violation = -30.0  # sphere entered without the gate, collision, crash, leaving the arena
    rew_scale_barrier = -4.0  # per second: hugging the keep-out sphere away from one's gate
    barrier_margin = 0.5  # [m]
    rew_scale_separation = -4.0  # per second: closer than separation_distance to another drone
    rew_scale_effort = -0.01  # per second
    rew_scale_action_rate = -0.05  # per second

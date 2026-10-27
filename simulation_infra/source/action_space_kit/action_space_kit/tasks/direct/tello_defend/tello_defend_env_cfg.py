"""Configuration for the multi-agent defend task: the hackathon scenario."""

from __future__ import annotations

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from action_space_kit.control import TelloCommandCfg

NUM_DEFENDERS = 2
NUM_ATTACKERS = 2

# obs per defender: own pos rel. site (3) + own vel (3) + gravity (3)
#                 + per attacker (rel pos 3, rel vel 3, visible flag 1)
_OBS_DIM = 9 + 7 * NUM_ATTACKERS


@configclass
class TelloDefendEnvCfg(DirectMARLEnvCfg):
    """Defenders intercept attackers before they reach the protected site.

    Attackers are scripted by default (``trainable_attackers = False``), which is the
    fastest path to a working defender policy. Set it True to train both sides.
    """

    # env
    decimation = 2
    episode_length_s = 20.0

    # multi-agent spaces -- every agent uses the four SDK rc channels
    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS)]
    # Bounded on purpose: the four SDK rc channels live in [-1, 1], and an unbounded
    # space leaves skrl nothing to clip sampled actions against.
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
    state_space = -1  # concatenate agent observations

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene: 1024 envs x 4 drones fits comfortably on a 24 GB GPU shared by several teams
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=12.0, replicate_physics=True)

    # Tello SDK velocity interface (shared by both sides)
    command: TelloCommandCfg = TelloCommandCfg()

    # - scenario geometry
    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS
    site_radius = 1.5  # [m] protected perimeter around the env origin
    site_height = 2.5  # [m] height of the protected cylinder
    spawn_radius = 6.0  # [m] where attackers enter from
    defender_radius = 2.5  # [m] where defenders start, around the site
    capture_radius = 0.35  # [m] distance counting as an interception
    obs_clip_pos = 12.0  # [m] relative positions are clipped to this before normalisation
    obs_clip_vel = 3.0  # [m/s] relative velocities are clipped to this
    trainable_attackers = False

    # - scripted attacker behaviour (curriculum stage 0 = straight dive)
    attacker_speed = [0.4, 0.9]  # [m/s]
    attacker_evasion = 0.0  # 0 = straight line, 1 = strong lateral weave

    # - reward scales (per defender)
    # Balance matters here: a breach happens in nearly every early episode, so a huge
    # penalty drowns the shaping terms and training drifts below a do-nothing baseline
    # (measured: -45.8 return for zero actions). Keep the penalty comparable to what a
    # defender can earn by doing its job.
    rew_scale_closing = 5.0  # getting nearer its target attacker
    rew_scale_proximity = 2.0  # continuous pull towards the nearest attacker
    rew_scale_capture = 30.0  # intercepting one
    rew_scale_cleared = 30.0  # bonus for downing the last attacker of a wave
    respawn_waves = True
    """Launch a fresh wave when the field is cleared, instead of ending the episode.

    Ending on success truncates the value bootstrap, which makes a short successful
    episode look worse than a long passive one -- measured as a policy that started at
    +137 return and unlearned its way to -260 while episodes grew longer.
    """
    rew_scale_breach = -20.0  # an attacker reaching the site (shared)
    rew_scale_effort = -0.01
    rew_scale_alive = 0.1

"""One Tello, from the floor: the room task cut down to a single drone, for tuning the rewards.

Same room, tags, sensing, rc action space, per-drone dynamics randomization and reward terms
as ``TelloRoomEnvCfg``; differences:
- one drone (no teammates: separation and collision terms are always 0);
- it starts on the floor, motors off, inside the same 2 m x 2 m box 6-7 m from tag 10, and
  the episode opens with the SDK ``takeoff`` (the firmware climbs to 0.8-1.0 m, then the
  policy flies; the observation has a taking-off flag);
- it gets one of the four landing spots at random each episode;
- success: it landed within success_radius of its spot (formation_reward pays then).
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.utils import configclass

from .tello_room_env_cfg import _OBS_DIM, TelloRoomEnvCfg


@configclass
class TelloRoomSingleEnvCfg(TelloRoomEnvCfg):
    num_drones = 1
    possible_agents = ["drone_0"]
    action_spaces = {"drone_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(5,))}
    observation_spaces = {"drone_0": _OBS_DIM + 1}  # + taking-off flag
    start_on_floor = True
    spot_count = 4
    random_assignment = True

"""Tello waypoint task: follow a moving goal."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Tello-Waypoint-v0",
    entry_point=f"{__name__}.tello_waypoint_env:TelloWaypointEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_waypoint_env_cfg:TelloWaypointEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

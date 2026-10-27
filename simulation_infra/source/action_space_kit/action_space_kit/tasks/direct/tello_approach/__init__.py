"""Approach task: four Tellos enter a base's airspace, each through its own gate."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Tello-Approach-v0",
    entry_point=f"{__name__}.tello_approach_env:TelloApproachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_approach_env_cfg:TelloApproachEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

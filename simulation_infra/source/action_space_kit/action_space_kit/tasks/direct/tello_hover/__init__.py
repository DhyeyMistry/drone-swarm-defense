"""Tello hover task: single drone, velocity-commanded."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Tello-Hover-v0",
    entry_point=f"{__name__}.tello_hover_env:TelloHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_hover_env_cfg:TelloHoverEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

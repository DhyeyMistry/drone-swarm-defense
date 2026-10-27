"""Defend task with onboard sensing: detect, track, defeat."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Defend-Camera-v0",
    entry_point=f"{__name__}.tello_defend_camera_env:TelloDefendCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_defend_camera_env_cfg:TelloDefendCameraEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

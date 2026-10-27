"""Multi-agent defend task: the Action Space hackathon scenario."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Defend-v0",
    entry_point=f"{__name__}.tello_defend_env:TelloDefendEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_defend_env_cfg:TelloDefendEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

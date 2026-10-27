"""Room task: Tellos fly lanes toward AprilTag 10 in the mapped room and land at the standoff."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Tello-Room-v0",
    entry_point=f"{__name__}.tello_room_env:TelloRoomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_room_env_cfg:TelloRoomEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="AS-Tello-Room-Single-v0",
    entry_point=f"{__name__}.tello_room_env:TelloRoomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_room_single_env_cfg:TelloRoomSingleEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_single_cfg.yaml",
    },
)

gym.register(
    id="AS-Tello-Room-Team-v0",
    entry_point=f"{__name__}.tello_room_env:TelloRoomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_room_team_env_cfg:TelloRoomTeamEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="AS-Tello-Room-Peer-v0",
    entry_point=f"{__name__}.tello_room_env:TelloRoomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_room_team_env_cfg:TelloRoomPeerEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

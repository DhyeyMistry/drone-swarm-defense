"""Four drones starting on the floor, with the single policy's 25-feature layout."""

from isaaclab.utils import configclass

from .tello_room_env_cfg import _OBS_DIM, TelloRoomEnvCfg


@configclass
class TelloRoomTeamEnvCfg(TelloRoomEnvCfg):
    start_on_floor = True
    observation_spaces = {f"drone_{i}": _OBS_DIM + 1 for i in range(4)}


@configclass
class TelloRoomPeerEnvCfg(TelloRoomTeamEnvCfg):
    peer_observations = True
    rew_predictive_separation = -100.0
    # Original 25, own velocity (3), uncertainty (2), three peers' relative p/v (18).
    observation_spaces = {f"drone_{i}": 48 for i in range(4)}

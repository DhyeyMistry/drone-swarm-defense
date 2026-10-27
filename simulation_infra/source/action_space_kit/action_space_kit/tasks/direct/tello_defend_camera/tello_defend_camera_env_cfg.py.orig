"""Configuration for the defend task with onboard sensing.

Same scenario as ``AS-Defend-v0``, but defenders no longer receive perfect knowledge of
the attackers. Observations come from field-of-view limited, noisy detections passed
through a tracker -- the detect-track-defeat loop the hackathon asks for.
"""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from action_space_kit.perception import TrackerCfg
from action_space_kit.tasks.direct.tello_defend.tello_defend_env_cfg import TelloDefendEnvCfg


@configclass
class TelloDefendCameraEnvCfg(TelloDefendEnvCfg):
    # rendering is the bottleneck here: keep env counts low and book a GPU slot
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=12.0, replicate_physics=True)

    # onboard camera, matched to the real Tello: 82.6 deg horizontal FOV, 960x720 at 30 Hz.
    # Downscaled here because policies rarely need full resolution and VRAM is shared.
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Defender_0/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=["rgb"],  # depth adds a second annotator; re-enable once RGB is stable
        spawn=PinholeCameraCfg(focal_length=12.0, focus_distance=4.0, horizontal_aperture=21.6, clipping_range=(0.1, 30.0)),
        width=320,
        height=240,  # under ~300 px wide, Isaac Sim routes rendering through DLSS upscaling
    )
    enable_cameras = True

    # - sensing model used to build observations
    detection_fov_deg = 82.6  # Tello camera horizontal FOV
    detection_range = 8.0  # [m]
    detection_noise = 0.05  # [m] std-dev
    detection_dropout = 0.05  # probability a visible attacker is missed per frame
    tracker: TrackerCfg = TrackerCfg()
    use_ground_truth = False  # set True to fall back to perfect state, for debugging

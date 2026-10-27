"""Configuration for the Tello waypoint-following task."""

from __future__ import annotations

from isaaclab.utils import configclass

from action_space_kit.tasks.direct.tello_hover.tello_hover_env_cfg import TelloHoverEnvCfg


@configclass
class TelloWaypointEnvCfg(TelloHoverEnvCfg):
    """Chase a goal that moves, the building block of pursuit behaviour.

    Same action space as hover -- the four SDK rc channels -- so a policy trained here
    still maps onto a real Tello.
    """

    episode_length_s = 15.0

    # - moving goal
    goal_speed = [0.2, 0.8]  # [m/s] sampled per episode
    goal_resample_time_s = 3.0  # how often the goal picks a new direction
    goal_bounds_xy = 2.0  # [m] goal stays within this half-extent of the env origin
    goal_bounds_z = [0.6, 2.0]  # [m] vertical band the goal stays inside

    # reward: chasing needs a stronger pull and tolerates more speed than hovering
    rew_scale_distance = 10.0
    rew_scale_lin_vel = -0.01
    max_distance = 4.0

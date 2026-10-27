"""Shared estimated-state landing gate; supports numpy and torch, never sends commands."""


def update_landing_gate(
    cfg,
    position,
    velocity,
    pos_variance,
    vel_variance,
    target,
    fix_age,
    acquired,
    lost,
    at_waypoint,
    busy,
    dwell,
    dt,
    xp,
):
    """Return (continuous safe dwell seconds, ready).

    Estimates describe the delayed camera frame. Project by the configured video delay;
    freshness includes that delay. A stopped/restarted filter cannot pass on zero velocity
    alone: both position and velocity covariance must also be small. Every failed check
    resets the dwell. All inputs use the simulator's world frame (Z up), metres and seconds.
    """
    delay = cfg.tag_sensor.video_delay_steps * cfg.sim.dt * cfg.decimation
    projected = position + velocity * delay
    error = ((projected[..., :2] - target[..., :2]) ** 2).sum(-1) ** 0.5
    speed = (velocity[..., :2] ** 2).sum(-1) ** 0.5
    finite = (
        xp.isfinite(position).all(-1)
        & xp.isfinite(velocity).all(-1)
        & xp.isfinite(pos_variance).all(-1)
        & xp.isfinite(vel_variance).all(-1)
    )
    trusted = (
        (pos_variance >= 0).all(-1)
        & (vel_variance >= 0).all(-1)
        & (pos_variance <= cfg.land_max_position_std**2).all(-1)
        & (vel_variance <= cfg.land_max_velocity_std**2).all(-1)
    )
    safe = (
        finite
        & trusted
        & acquired
        & ~lost
        & at_waypoint
        & ~busy
        & (error <= cfg.land_max_xy_error)
        & (speed <= cfg.land_max_horizontal_speed)
        & (abs(projected[..., 2] - cfg.waypoint_height) <= cfg.land_height_tolerance)
        & (fix_age + delay <= cfg.land_max_observation_age)
    )
    dwell = xp.where(safe, dwell + dt, 0 * dwell)
    return dwell, safe & (dwell >= cfg.land_dwell_s)

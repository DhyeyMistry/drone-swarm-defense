"""Mix shorter starts into room training without changing control or observations."""

import torch


def validate_start_mixture(cfg):
    short, hover = cfg.spawn_short_fraction, cfg.spawn_hover_fraction
    if min(short, hover) < 0 or short + hover > 1:
        raise ValueError("Start mixture fractions must be nonnegative and sum to at most one")
    if short + hover and cfg.num_drones != 1:
        raise ValueError("Start mixtures require a single drone")
    if cfg.spawn_short_x_range[0] > cfg.spawn_short_x_range[1]:
        raise ValueError("spawn_short_x_range must be ordered")


def mix_starts(cfg, xy, z, spots, floor_start):
    """Return positions, takeoff flags, and start kind (0 configured, 1 short, 2 hover).

    The zero-mixture path consumes no RNG, preserving existing seeded evaluations.
    Fractions describe episode resets, not proportions of elapsed training steps.
    """
    takeoff = torch.full_like(z, floor_start, dtype=torch.bool)
    kind = torch.zeros(xy.shape[0], dtype=torch.long, device=xy.device)
    if cfg.spawn_short_fraction + cfg.spawn_hover_fraction == 0:
        return xy, z, takeoff, kind
    draw = torch.rand(xy.shape[0], device=xy.device)
    hover = draw < cfg.spawn_hover_fraction
    short = (draw >= cfg.spawn_hover_fraction) & (draw < cfg.spawn_hover_fraction + cfg.spawn_short_fraction)
    kind[short], kind[hover] = 1, 2
    lo, hi = cfg.spawn_short_x_range
    short_xy = torch.stack(
        [
            lo + (hi - lo) * torch.rand_like(z),
            spots[..., 1] + cfg.spawn_short_y_radius * (2 * torch.rand_like(z) - 1),
        ],
        -1,
    )
    hover_xy = spots[..., :2] + cfg.spawn_hover_radius * (2 * torch.rand_like(xy) - 1)
    xy = torch.where(short[:, None, None], short_xy, xy)
    xy = torch.where(hover[:, None, None], hover_xy, xy)
    z = torch.where(hover[:, None], cfg.waypoint_height + 0.05 * (2 * torch.rand_like(z) - 1), z)
    takeoff[hover] = False
    return xy, z, takeoff, kind

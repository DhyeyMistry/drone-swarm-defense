"""Potential shaping with the same discount as the PPO return."""


def potential_delta(previous, current, terminated, gamma, xp):
    # Timeouts bootstrap; only absorbing terminations have zero next potential.
    return gamma * xp.where(terminated, 0 * current, current) - previous


def precision_potential(distance, speed, weight, position_scale, speed_scale, xp):
    """Bounded shaping for precise, slow arrival, independent of milestone flags."""
    if weight < 0 or min(position_scale, speed_scale) <= 0:
        raise ValueError("Precision weight must be nonnegative and scales must be positive")
    return weight * xp.exp(-distance / position_scale - speed / speed_scale)

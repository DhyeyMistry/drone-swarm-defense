"""Peer features from estimated telemetry; compatible with NumPy and torch.

No hardware APIs. Neighbor slots follow ascending drone index, excluding self,
unless nearest-first ordering is requested by the saved policy configuration.
"""


def peer_feature_count(num_drones):
    return 5 + 6 * (num_drones - 1)


def peer_features(position, velocity, yaw, position_variance, velocity_variance, xp, sort_neighbors=False, order_version=1):
    """Own velocity/uncertainty and peers' relative position/velocity, in heading axes."""
    is_torch = xp.__name__ == "torch"
    cat = xp.cat if is_torch else xp.concatenate
    c, s = xp.cos(yaw), xp.sin(yaw)
    if order_version not in (1, 2):
        raise ValueError("Unknown peer ordering version")
    distance_key = None
    if sort_neighbors and order_version == 2:
        # Sort in world axes, independent of heading/trigonometric roundoff.
        # A millimeter lattice and drone-index tie break give CPU/CUDA identical
        # keys for symmetric starts. The actual observation values stay unrounded.
        p64 = position.to(xp.float64) if is_torch else position.astype(xp.float64)
        relative = p64[..., None, :, :] - p64[..., :, None, :]
        millimeters = xp.round(relative * 1000.)
        millimeters = millimeters.to(xp.int64) if is_torch else millimeters.astype(xp.int64)
        count = position.shape[-2]
        ids = xp.arange(count, device=position.device) if is_torch else xp.arange(count)
        distance_key = (millimeters * millimeters).sum(-1) * count + ids

    def rotate(vector, i):
        return xp.stack((c[..., i] * vector[..., 0] + s[..., i] * vector[..., 1],
                         s[..., i] * vector[..., 0] - c[..., i] * vector[..., 1], vector[..., 2]), -1)

    def std_max(variance):
        variance = variance.clamp(min=0) if is_torch else xp.maximum(variance, 0)
        std = xp.sqrt(variance)
        return std.amax(-1, keepdim=True) if is_torch else std.max(-1, keepdims=True)

    p_std, v_std = std_max(position_variance), std_max(velocity_variance)
    rows = []
    for i in range(position.shape[-2]):
        pieces = [rotate(velocity[..., i, :], i), p_std[..., i, :], v_std[..., i, :]]
        neighbors = []
        for j in range(position.shape[-2]):
            if i != j:
                neighbors.append(cat((rotate(position[..., j, :] - position[..., i, :], i),
                                      rotate(velocity[..., j, :] - velocity[..., i, :], i)), -1))
        if neighbors:
            neighbors = xp.stack(neighbors, -2)
            if sort_neighbors:
                neighbor_ids = [j for j in range(position.shape[-2]) if j != i]
                distance2 = (distance_key[..., i, neighbor_ids] if order_version == 2 else
                             (neighbors[..., :3] ** 2).sum(-1))
                order = xp.argsort(distance2, dim=-1) if is_torch else xp.argsort(distance2, axis=-1)
                if is_torch:
                    neighbors = xp.gather(neighbors, -2, order.unsqueeze(-1).expand_as(neighbors))
                else:
                    neighbors = xp.take_along_axis(neighbors, order[..., None], axis=-2)
            pieces.append(neighbors.reshape(*neighbors.shape[:-2], -1))
        rows.append(cat(pieces, -1))
    return xp.stack(rows, -2)


def predicted_separation_cost(position, velocity, landed, horizon, margin, torch):
    """Training cost for a close approach under current velocity; no action overrides.

Uses simulator state only as a reward, never as an actor observation. A bounded
lookahead supplies a warning before contact; stationary separated pads cost zero.
"""
    if horizon <= 0 or margin <= 0:
        raise ValueError("Prediction horizon and separation margin must be positive")
    rel_p = position.unsqueeze(-3) - position.unsqueeze(-2)
    rel_v = velocity.unsqueeze(-3) - velocity.unsqueeze(-2)
    speed2 = (rel_v * rel_v).sum(-1)
    time = (-(rel_p * rel_v).sum(-1) / speed2.clamp(min=1e-6)).clamp(0., horizon)
    closest = torch.linalg.vector_norm(rel_p + time.unsqueeze(-1) * rel_v, dim=-1)
    cost = (1. - closest / margin).clamp(min=0.) * torch.exp(-time / horizon)
    count = position.shape[-2]
    valid = ~torch.eye(count, dtype=torch.bool, device=position.device)
    valid = valid & ~(landed.unsqueeze(-1) & landed.unsqueeze(-2))
    return (cost * valid).sum(-1)

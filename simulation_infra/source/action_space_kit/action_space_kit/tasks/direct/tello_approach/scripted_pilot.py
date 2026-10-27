"""A hand-written pilot for AS-Tello-Approach-v0, flying through the same rc actions as a policy.

Used as a baseline by ``scripts/check_approach_env.py`` (to prove the task is solvable and
the reward separates good from bad flying) and to record reference videos.
"""

from __future__ import annotations

import torch


def scripted_actions(u, phase: torch.Tensor) -> dict:
    """Actions for every drone: out-point, gate, then its landing spot.

    phase 0 flies to a point 0.6 m outside the gate (pushed away from the sphere if
    it gets close), phase 1 goes radially in through the gate, phase 2 flies to the point
    above its designated landing spot, where the env sends ``land``.
    """
    cfg = u.cfg
    pos = u._positions()
    wp = u._waypoints
    wp_dir = wp / torch.linalg.norm(wp, dim=-1, keepdim=True)
    outer = wp_dir * (cfg.keepout_radius + 0.6)
    phase[(phase == 0) & (torch.linalg.norm(pos - outer, dim=-1) < 0.35)] = 1
    phase[u._gate_passed] = 2
    target = torch.where((phase == 0).unsqueeze(-1), outer, wp)
    # after the gate: the point above the drone's own landing spot
    target = torch.where((phase == 2).unsqueeze(-1), u._landing_targets(), target)
    v = 1.5 * (target - pos)
    r = torch.linalg.norm(pos, dim=-1, keepdim=True)
    push = ((cfg.keepout_radius + 0.5 - r) * 3.0).clamp(min=0.0) * pos / r
    v = v + push * (phase == 0).unsqueeze(-1).float()
    # keep clear of the other drones
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)
    rep = ((0.6 - dist).clamp(min=0.0) * 3.0 * diff / dist).sum(dim=2)
    v = v + rep
    v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1.0)  # <= 1 m/s
    quat = u._stack("root_quat_w")
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    fwd = c * v[..., 0] + s * v[..., 1]
    left = -s * v[..., 0] + c * v[..., 1]
    act = torch.stack([fwd, left, v[..., 2], torch.zeros_like(fwd)], dim=-1) / cfg.dynamics.max_speed_xy
    return {a: act[:, i].clamp(-1, 1) for i, a in enumerate(cfg.possible_agents)}

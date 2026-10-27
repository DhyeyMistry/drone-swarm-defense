"""Sanity checks for AS-Tello-Approach-v0: geometry, flight model and baselines.

Run this before trusting a training curve. It verifies, inside Isaac Sim:

  geometry    spawns, approach vectors and waypoints respect the scenario spec
  controller  the randomized Tello model hovers and tracks velocity commands in PhysX
  baselines   a scripted "gate, then base" pilot succeeds and zero actions do not, so the
              task is solvable through the rc action space and the reward separates them

    python scripts/check_approach_env.py --headless --test all
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--test", choices=["geometry", "controller", "baselines", "all"], default="all")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=256, help="Finished episodes per baseline.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: E402, F401
from action_space_kit.tasks.direct.tello_approach.scripted_pilot import scripted_actions  # noqa: E402
from action_space_kit.tasks.direct.tello_approach.tello_approach_env_cfg import FT, TelloApproachEnvCfg  # noqa: E402

FAILURES: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  [ok]   " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAILURES.append(msg)


def make_env(num_envs: int, **overrides):
    cfg = TelloApproachEnvCfg()
    cfg.scene.num_envs = num_envs
    for k, v in overrides.items():
        setattr(cfg, k, v)
    env = gym.make("AS-Tello-Approach-v0", cfg=cfg)
    env.reset()
    return env, env.unwrapped


def zero_actions(u):
    return {a: torch.zeros(u.num_envs, 4, device=u.device) for a in u.cfg.possible_agents}


# -- geometry -------------------------------------------------------------------


def test_geometry(u) -> None:
    cfg = u.cfg
    print("\n== geometry ==")
    print(f"  base (0, 0, 0); spawn square {cfg.spawn_square:.2f} m ({cfg.spawn_square / FT:.0f} ft) "
          f"centred {cfg.spawn_distance:.2f} m ({cfg.spawn_distance / FT:.0f} ft) out")
    print(f"  keep-out R {cfg.keepout_radius:.2f} m ({cfg.keepout_radius / FT:.1f} ft), gate r {cfg.gate_radius} m, "
          f"goal r {cfg.goal_radius} m, speed <= {cfg.dynamics.max_speed_xy} m/s per axis")
    rounds, D = 40, cfg.num_drones
    ids = torch.arange(u.num_envs, device=u.device)
    min_sep, min_el, max_el, max_az = math.pi, math.pi, 0.0, 0.0
    wp_r_err, min_spawn_r, min_line_gap, max_line_dev, max_sq = 0.0, 1e9, 1e9, 0.0, 0.0
    pairing_ok, spot_ok, spot_r, spot_gap = True, True, 0.0, 9.0
    for _ in range(rounds):
        u._reset_idx(ids)
        wp = u._waypoints
        pos = u._positions()
        az0 = u._spawn_azimuth
        r = torch.linalg.norm(wp, dim=-1)
        wp_r_err = max(wp_r_err, (r - cfg.keepout_radius).abs().max().item())
        d = wp / r.unsqueeze(-1)
        el = torch.asin(d[..., 2])
        min_el, max_el = min(min_el, el.min().item()), max(max_el, el.max().item())
        az = torch.atan2(d[..., 1], d[..., 0]) - az0.unsqueeze(-1)
        az = torch.atan2(torch.sin(az), torch.cos(az))
        max_az = max(max_az, az.abs().max().item())
        cos = torch.einsum("eid,ejd->eij", d, d) - 2.0 * torch.eye(D, device=u.device)
        sep = torch.acos(cos.amax(dim=(-1, -2)).clamp(-1, 1))
        min_sep = min(min_sep, sep.min().item())
        # spawn: in the rotated square frame (radial, lateral)
        e = torch.stack([torch.cos(az0), torch.sin(az0)], -1).unsqueeze(1)
        lat = torch.stack([-torch.sin(az0), torch.cos(az0)], -1).unsqueeze(1)
        rad = (pos[..., :2] * e).sum(-1) - cfg.spawn_distance
        side = (pos[..., :2] * lat).sum(-1)
        max_sq = max(max_sq, rad.abs().max().item(), side.abs().max().item())
        min_spawn_r = min(min_spawn_r, torch.linalg.norm(pos, dim=-1).min().item())
        # collinearity: distance of each drone from the line through the outer two
        order = torch.argsort(side, dim=-1)
        ps = torch.gather(pos[..., :2], 1, order.unsqueeze(-1).expand(-1, -1, 2))
        a, b = ps[:, 0], ps[:, -1]
        ab = (b - a) / torch.linalg.norm(b - a, dim=-1, keepdim=True)
        rel = ps - a.unsqueeze(1)
        dev = (rel[..., 0] * ab[:, None, 1] - rel[..., 1] * ab[:, None, 0]).abs()
        max_line_dev = max(max_line_dev, dev.max().item())
        gaps = torch.linalg.norm(ps[:, 1:] - ps[:, :-1], dim=-1)
        min_line_gap = min(min_line_gap, gaps.min().item())
        # corridors must not cross: the k-th drone from the left owns the k-th gate from the left
        gate_rank = torch.argsort(torch.argsort(az, dim=-1), dim=-1)
        drone_rank = torch.argsort(torch.argsort(side, dim=-1), dim=-1)
        pairing_ok &= bool((gate_rank == drone_rank).all())
        # landing spots: on the ground, inside the touchdown zone, apart, same order as the gates
        spots = u._landing_spots
        spot_r = max(spot_r, torch.linalg.norm(u._landing_targets(), dim=-1).max().item())
        spot_gap = min(spot_gap, torch.cdist(spots, spots).masked_fill(torch.eye(D, dtype=torch.bool, device=u.device), 9.0).min().item())
        spot_az = torch.atan2(spots[..., 1], spots[..., 0]) - az0.unsqueeze(-1)
        spot_az = torch.atan2(torch.sin(spot_az), torch.cos(spot_az))
        spot_ok &= bool((torch.argsort(torch.argsort(spot_az, dim=-1), dim=-1) == gate_rank).all()) and bool((spots[..., 2] == 0).all())
    n = rounds * u.num_envs
    print(f"  sampled {n} scenarios")
    check(min_sep >= math.radians(10.0) - 1e-4, f"approach vectors pairwise >= 10 deg (min {math.degrees(min_sep):.2f})")
    el_lo, el_hi = cfg.approach_elevation
    check(min_el >= el_lo - 1e-4 and max_el <= el_hi + 1e-4, f"approach vectors {math.degrees(el_lo):.0f}-{math.degrees(el_hi):.0f} deg elevated (min {math.degrees(min_el):.2f}, max {math.degrees(max_el):.2f})")
    check(max_az <= cfg.approach_azimuth_spread + 1e-4, f"approach azimuth within +-45 deg of spawn side (max {math.degrees(max_az):.1f})")
    check(wp_r_err < 1e-4, f"waypoints lie on the keep-out sphere (max err {wp_r_err:.2e} m)")
    check(max_sq <= 0.5 * cfg.spawn_square + 1e-4, f"spawns inside the {cfg.spawn_square / FT:.0f}x{cfg.spawn_square / FT:.0f} ft square (max offset {max_sq:.3f} m <= {0.5 * cfg.spawn_square:.3f})")
    check(min_spawn_r > cfg.keepout_radius + 0.5, f"spawns well outside the keep-out sphere (min {min_spawn_r:.2f} m)")
    check(max_line_dev < 1e-3, f"drones stand in a line (max deviation {max_line_dev:.2e} m)")
    check(min_line_gap >= cfg.spawn_spacing[0] - 1e-3, f"line spacing >= {cfg.spawn_spacing[0]} m (min {min_line_gap:.3f})")
    check(pairing_ok, "corridors don't cross: k-th drone from the left owns the k-th gate from the left")
    check(spot_ok, "landing spot k is the k-th from the left, like gate k, and on the ground")
    check(spot_r < cfg.goal_radius, f"hand-over points above the spots are inside the {cfg.goal_radius} m touchdown zone (max {spot_r:.2f} m)")
    check(spot_gap > 0.35, f"landing spots are well apart (min {spot_gap:.2f} m)")


# -- controller -------------------------------------------------------------------


def fresh_start(u) -> None:
    """Reset every env with the full episode ahead (a full reset otherwise staggers timeouts)."""
    u._reset_idx(torch.arange(u.num_envs, device=u.device))
    u.episode_length_buf[:] = 0


def test_controller(env, u) -> None:
    """Fly all drones with fixed commands through PhysX and check the response."""
    print("\n== controller (PhysX, domain randomization on) ==")
    fresh_start(u)
    z0 = u._positions()[..., 2].clone()
    steps = int(3.0 / u.step_dt)
    aborted = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    for _ in range(steps):
        u.step(zero_actions(u))
        aborted |= u.reset_buf
    pos = u._positions()
    sag = (pos[..., 2] - z0).abs().max().item()
    vel = torch.linalg.norm(u._stack("root_lin_vel_w"), dim=-1).max().item()
    check(sag < 0.25, f"hover 3 s with rc 0: max altitude change {sag:.3f} m")
    check(vel < 0.1, f"hover 3 s with rc 0: max residual speed {vel:.3f} m/s")

    check(not aborted.any().item(), "no env aborted while hovering")

    # drones face the base, so fly backwards (away from the sphere) at rc b = -100 for 2 s
    fresh_start(u)
    act = {a: torch.tensor([[-1.0, 0.0, 0.0, 0.0]], device=u.device).repeat(u.num_envs, 1) for a in u.cfg.possible_agents}
    max_tilt = 0.0
    aborted = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    for _ in range(int(2.0 / u.step_dt)):
        u.step(act)
        aborted |= u.reset_buf
        g = u._stack("projected_gravity_b")
        max_tilt = max(max_tilt, torch.acos((-g[..., 2]).clamp(-1, 1)).max().item())
    v = u._stack("root_lin_vel_w")
    quat = u._stack("root_quat_w")
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    fwd = (v[..., 0] * torch.cos(yaw) + v[..., 1] * torch.sin(yaw))[~aborted]
    print(f"  {int(aborted.sum())} env(s) left the arena while backing off and were reset; excluded")
    check(fwd.mean().item() < -0.85, f"rc b=-100 -> {-fwd.mean():.2f} m/s backwards (worst {-fwd.max():.2f}) after 2 s")
    check(max_tilt < math.radians(30.0), f"tilt stays under the firmware limit (max {math.degrees(max_tilt):.1f} deg)")
    check(bool(torch.isfinite(v).all()), "no NaNs in drone state")


# -- baselines ------------------------------------------------------------------


def run_baseline(env, u, name: str, episodes: int) -> dict:
    fresh_start(u)
    phase = torch.zeros(u.num_envs, u.cfg.num_drones, dtype=torch.long, device=u.device)
    returns = torch.zeros(u.num_envs, device=u.device)
    totals = {"success": 0.0, "arrived": 0.0, "landed": 0.0, "gates": 0.0, "illegal": 0.0, "collision": 0.0, "crash": 0.0, "time": 0.0, "return": 0.0}
    done_count = 0
    while done_count < episodes:
        act = scripted_actions(u, phase) if name == "scripted" else zero_actions(u)
        _, rew, term, trunc, _ = env.step(act)
        returns += sum(rew.values())
        fin = u.outcome_valid
        if fin.any():
            idx = fin.nonzero().flatten()
            done_count += len(idx)
            totals["success"] += u.outcome_success[idx].float().sum().item()
            totals["arrived"] += u.outcome_arrived[idx].sum().item()
            totals["landed"] += u.outcome_landed[idx].sum().item()
            totals["gates"] += u.outcome_gates[idx].sum().item()
            totals["illegal"] += u.outcome_illegal_entry[idx].float().sum().item()
            totals["collision"] += u.outcome_collision[idx].float().sum().item()
            totals["crash"] += u.outcome_crash[idx].float().sum().item()
            totals["time"] += u.outcome_time[idx].sum().item()
            totals["return"] += returns[idx].sum().item()
            returns[idx] = 0.0
            phase[idx] = 0
    stats = {k: v / done_count for k, v in totals.items()}
    print(f"  {name:9s}: success {stats['success']:.0%}, arrived {stats['arrived']:.2f}/4, landed {stats['landed']:.2f}/4, gates {stats['gates']:.2f}/4, "
          f"illegal entry {stats['illegal']:.0%}, collision {stats['collision']:.0%}, crash {stats['crash']:.0%}, "
          f"time {stats['time']:.1f} s, team return {stats['return']:.1f}  ({done_count} episodes)")
    return stats


def test_baselines(env, u, episodes: int) -> None:
    print("\n== baselines ==")
    zero = run_baseline(env, u, "zero", episodes)
    scripted = run_baseline(env, u, "scripted", episodes)
    check(scripted["success"] > 0.9, f"scripted pilot gets all four landed > 90% ({scripted['success']:.0%})")
    check(scripted["illegal"] == 0.0, f"scripted pilot never enters the sphere illegally ({scripted['illegal']:.0%})")
    check(zero["success"] == 0.0, f"zero actions never succeed ({zero['success']:.0%})")
    check(scripted["return"] > zero["return"] + 100.0,
          f"reward separates them: scripted {scripted['return']:.1f} vs zero {zero['return']:.1f}")


def main():
    env, u = make_env(args_cli.num_envs)
    if args_cli.test in ("geometry", "all"):
        test_geometry(u)
    if args_cli.test in ("controller", "all"):
        test_controller(env, u)
    if args_cli.test in ("baselines", "all"):
        test_baselines(env, u, args_cli.episodes)
    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} CHECK(S) FAILED:\n  " + "\n  ".join(FAILURES)))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Kit can shut down before an uncaught traceback reaches the terminal
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

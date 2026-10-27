"""Sanity checks for AS-Tello-Room-v0: four Tellos form up 1 m in front of tag 10.

  geometry   every mapped tag is in the USD stage at its map.yaml pose (in every env); the
             four start in the 2 m box, 6-7 m from tag 10, >= 0.5 m apart; the four spots
             are 1 m out, 0.5 m apart, assigned at random per episode
  sensing    from the start every drone gets tag fixes; published positions are accurate;
             a drone turned away from every tag goes lost (its position freezes) while its
             own sticks keep flying it, and publishes again once it turns back
  commands   the policy sends integer rc in -100..100 on every channel plus the SDK land;
             land puts every drone on the floor
  motion     one policy step = one rc packet at the measured rate; link loss as configured;
             integer rc steps through PhysX give the speeds and response times in
             control/tello_motion_fit.yaml (the fit of the real flights)
  baselines  zero sticks never succeed, and the per-step -0.5 landing / no-tag penalties
             add up exactly (no hand-written pilot: the policy is trained end to end)
  kalman     the laptop's filter (drone_deployment_infra/core/positional_kalman_filter.py)
             gives the same numbers on numpy (laptop) and torch (sim), and in flight the
             filtered position the policy sees is closer to the truth than the raw fix
  central    one row per env: all drones' blocks in, all their commands out, team reward
  single     (--task AS-Tello-Room-Single-v0) one drone from the floor: the firmware takeoff
             climbs to 0.8-1.0 m ignoring the sticks, takeoff steps pay no closer / no-tag
             terms, then the sticks fly it; it gets any of the four spots
             (runs single + kalman + baselines + central; the other tests assume four hovering drones)

    python -u scripts/check_room_env.py --headless --test all > outputs/check_room.log 2>&1
    python -u scripts/check_room_env.py --headless --task AS-Tello-Room-Single-v0 > outputs/check_room_single.log 2>&1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", choices=["AS-Tello-Room-v0", "AS-Tello-Room-Single-v0", "AS-Tello-Room-Team-v0", "AS-Tello-Room-Peer-v0"], default="AS-Tello-Room-v0")
parser.add_argument("--test", choices=["geometry", "sensing", "commands", "motion", "baselines", "kalman", "central", "single", "team", "curriculum", "precision", "search", "all"],
                    default="all")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import action_space_kit.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from action_space_kit.control import load_motion_fit  # noqa: E402
from action_space_kit.tasks.direct.tello_room.tello_room_env import _yaw_of  # noqa: E402
from action_space_kit.tags.sensing import blur_loss, camera_axes, decode_rate, sighting  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

FAILURES: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  [ok]   " if ok else "  [FAIL] ") + msg)
    if not ok:
        FAILURES.append(msg)


def fresh_start(u) -> None:
    u._reset_idx(torch.arange(u.num_envs, device=u.device))
    u.episode_length_buf[:] = 0


def sticks(u, rc, land: bool = False) -> dict:
    """Same integer rc for every drone (clipped to the command envelope) -> actions; the 5th
    output asks for land or not."""
    a = u.rc.from_rc(torch.tensor(rc, dtype=torch.float32, device=u.device))
    a = torch.cat([a, torch.tensor([1.0 if land else -1.0], device=u.device)])
    return {k: a.expand(u.num_envs, 5) for k in u.cfg.possible_agents}


# -- geometry -------------------------------------------------------------------


def test_geometry(u) -> None:
    import omni.usd
    from pxr import Usd, UsdGeom

    cfg = u.cfg
    print("\n== geometry ==")
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    origins = u.scene.env_origins.cpu().numpy()
    for env in (0, u.num_envs - 1):
        worst = 0.0
        missing = []
        for tid, tag in u.room_tags.items():
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env}/Room/tag_{tid}")
            if not prim.IsValid():
                missing.append(tid)
                continue
            m = cache.GetLocalToWorldTransform(prim)
            pos = np.array(m.ExtractTranslation()) - origins[env]
            normal = np.array(m.TransformDir((0, 0, 1)))
            worst = max(worst, float(np.linalg.norm(pos - tag.pos)), float(np.linalg.norm(normal - tag.normal)))
        check(not missing, f"env {env}: all {len(u.room_tags)} mapped tags spawned {sorted(u.room_tags)}" + (f", missing {missing}" if missing else ""))
        check(worst < 1e-3, f"env {env}: tag poses match map.yaml (worst error {worst:.2e})")
    t10 = u.room_tags[10]
    check(np.allclose(t10.normal, [1, 0, 0], atol=1e-6) and abs(t10.pos[2] - cfg.tag10_height) < 1e-6,
          "tag 10 is the origin: on the X=0 wall, facing +X, at tag10_height")

    fresh_start(u)
    pos = u._positions()
    cx, cy = cfg.spawn_box_center
    half = 0.5 * cfg.spawn_box_size + 1e-4
    in_box = ((pos[..., 0] - cx).abs() <= half) & ((pos[..., 1] - cy).abs() <= half)
    r = torch.linalg.norm(pos[..., :2], dim=-1)
    check(bool(in_box.all()), f"every start is inside the {cfg.spawn_box_size:.0f} m box centred {cfg.spawn_box_center}")
    check(bool(((r >= cfg.spawn_range[0] - 1e-4) & (r <= cfg.spawn_range[1] + 1e-4)).all()),
          f"every start is {cfg.spawn_range[0]:.0f}-{cfg.spawn_range[1]:.0f} m from tag 10 (range {r.min():.2f}-{r.max():.2f} m)")
    sep = u._pairwise_distance(pos[..., :2].contiguous()).amin().item()
    for _ in range(49):  # 50 resets of every env
        fresh_start(u)
        sep = min(sep, u._pairwise_distance(u._positions()[..., :2].contiguous()).amin().item())
    check(sep >= cfg.spawn_min_separation - 1e-4, f"starts are >= {cfg.spawn_min_separation} m apart over 50 resets x {u.num_envs} envs (closest {sep:.2f} m)")

    spots = u._spots
    check(bool(torch.allclose(spots[..., 0], torch.full_like(spots[..., 0], cfg.spot_distance)))
          and bool((spots[..., 2] == 0).all()), f"landing spots are on the floor, {cfg.spot_distance:.1f} m in front of the tag-10 wall")
    check(abs(u._wp_offset[2].item() - cfg.waypoint_height) < 1e-6,
          f"each waypoint is {cfg.waypoint_height} m above its landing spot, radius {cfg.waypoint_radius} m")
    ys = torch.sort(spots[0, :, 1]).values
    check(bool(torch.allclose(ys[1:] - ys[:-1], torch.full_like(ys[1:], cfg.spot_spacing))),
          f"spots are {cfg.spot_spacing} m apart: Y = {[round(v, 2) for v in ys.tolist()]}")
    perm = (torch.sort(spots[..., 1], -1).values == ys.unsqueeze(0)).all().item()
    check(bool(perm), "every env assigns each spot to exactly one drone")
    if cfg.random_assignment:
        crossed = (torch.argsort(pos[..., 1], -1) != torch.argsort(spots[..., 1], -1)).any(-1).float().mean().item()
        kinds = len({tuple(r) for r in torch.argsort(spots[..., 1], -1).tolist()})
        check(kinds >= 20 and crossed > 0.5, f"assignment varies: {kinds} different drone->spot orders over {u.num_envs} envs, "
              f"{crossed:.0%} not left-to-right")
    else:
        same = (torch.argsort(pos[..., 1], -1) == torch.argsort(spots[..., 1], -1)).all().item()
        check(bool(same), "drone k-th from the left takes spot k (paths don't cross)")
    obs = u._get_observations()
    dim = cfg.observation_spaces[cfg.possible_agents[0]]
    check(all(o.shape == (u.num_envs, dim) for o in obs.values()), f"observation dim {dim}")


# -- sensing --------------------------------------------------------------------


def test_sensing(env, u) -> None:
    print("\n== sensing (positions only from AprilTags; no fix -> lost) ==")
    fresh_start(u)
    errs, fresh = [], []
    for k in range(int(3.0 / u.step_dt)):
        u.step(sticks(u, (0, 0, 0, 0)))
        if k * u.step_dt > 1.0:
            m = u.fresh
            errs.append(torch.linalg.norm(u.pub_pos - u._positions(), dim=-1)[m])
            fresh.append(m.float().mean().item())
    acq = u.acquired.float().mean().item()
    err = torch.cat(errs)
    check(acq > 0.95, f"{acq:.0%} of drones get a tag fix within 3 s of the start (6-7 m out)")
    check(err.median().item() < 0.15, f"published position error at 6-7 m: median {err.median():.3f} m, p95 {err.quantile(0.95):.3f} m")
    print(f"  fix on {np.mean(fresh):.0%} of hovering steps")

    # motion blur: a fast-moving / turning camera loses whole frames (tags.sensing blur_*)
    ts = u.cfg.tag_sensor
    quat = u._stack("root_quat_w").reshape(-1, 4)
    R = matrix_from_quat(quat)
    cam_pos = u._positions().reshape(-1, 3) + R @ u._cam_offset
    cam_rot = camera_axes(R, ts.camera_pitch)
    rates = {}
    n = cam_pos.shape[0]
    for v, w in ((0.0, 0.0), (2.0, 0.0), (0.0, 0.35), (0.0, 1.0), (1.0, 1.0)):
        rates[(v, w)] = np.mean([sighting(ts, cam_pos, cam_rot, u._tag_pos, u._tag_normal, u._tag_size,
                                          speed=torch.full((n,), v, device=u.device),
                                          yaw_rate=torch.full((n,), w, device=u.device))[1].float().mean().item()
                                 for _ in range(50)])
    want = {k: rates[(0.0, 0.0)] * (1 - blur_loss(ts, torch.tensor(k[0]), torch.tensor(k[1])).item()) for k in rates}
    check(all(abs(rates[k] - want[k]) < 0.03 for k in rates),
          "motion blur drops whole frames, mostly when turning: frames with a tag at "
          + ", ".join(f"{v:.0f} m/s + {w:.2f} rad/s {rates[(v, w)]:.0%}" for v, w in rates)
          + f" (expected {', '.join(f'{want[k]:.0%}' for k in rates)})")

    # view angle: tag 10 seen from 3 m at growing angles off its face
    j10 = u._tag_ids.index(10)
    got, exp = [], []
    for deg in (0.0, 30.0, 45.0, 55.0, 65.0):
        a = math.radians(deg)
        cam = torch.tensor([[3.0 * math.cos(a), 3.0 * math.sin(a), u.cfg.tag10_height]], device=u.device).expand(4096, 3)
        yaw = torch.full((4096,), math.atan2(-math.sin(a), -math.cos(a)), device=u.device)
        c, s_, z, o = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
        Rb = torch.stack([torch.stack([c, -s_, z], -1), torch.stack([s_, c, z], -1), torch.stack([z, z, o], -1)], -2)
        m = sighting(ts, cam, camera_axes(Rb, ts.camera_pitch), u._tag_pos, u._tag_normal, u._tag_size)[2][:, j10]
        got.append(m.float().mean().item())
        exp.append(decode_rate(ts, torch.tensor(math.cos(a))).item() if a < ts.max_view_angle else 0.0)
    check(all(abs(g - e) < 0.03 for g, e in zip(got, exp)) and got[0] > got[2] > got[3] and got[-1] == 0.0,
          "tag 10 decodes less the further off its face: "
          + ", ".join(f"{d:.0f} deg {g:.0%}" for d, g in zip((0, 30, 45, 55, 65), got))
          + " (flights: 89% at 0-20, 77% at 20-35, 41% at 35-50, 16% at 50-60, 6% at 60-70)")

    # turn away from every tag: the feed stops and nothing takes over the sticks
    fresh_start(u)
    went_lost = torch.zeros_like(u.lost)
    frozen_ok, policy_flies = True, True
    for _ in range(int(9.0 / u.step_dt)):  # rc yaw 20 cw ~ 12 deg/s: ~110 deg, past every tag
        before = u.pub_pos.clone()
        was_lost = u.lost.clone()
        u.step(sticks(u, (0, 0, 0, 20)))
        stay = was_lost & u.lost
        frozen_ok &= bool(torch.allclose(u.pub_pos[stay], before[stay]))
        policy_flies &= bool((u.last_rc[..., 3] == 20).all())
        went_lost |= u.lost
    check(went_lost.float().mean().item() > 0.5, f"{went_lost.float().mean():.0%} of drones turned away from the tags went lost")
    check(frozen_ok, "a lost drone publishes nothing new (its last position stays frozen)")
    check(policy_flies, "the policy's sticks keep flying a lost drone (no scripted search takes over)")
    back = torch.zeros_like(went_lost)
    for _ in range(int(12.0 / u.step_dt)):  # turn back the other way
        u.step(sticks(u, (0, 0, 0, -20)))
        back |= went_lost & ~u.lost
    rate = (back.sum() / went_lost.sum().clamp(min=1)).item()
    check(rate > 0.9, f"{rate:.0%} of lost drones publish again once they turn back to the tags")


# -- commands -------------------------------------------------------------------


def test_commands(env, u) -> None:
    if u.cfg.reward_version == 1:
        test_commands_legacy(env, u)
        return
    print("\n== version 2 commands and rewards ==")
    fresh_start(u)
    finite, integer = True, True
    for _ in range(40):
        actions = {k: torch.randn(u.num_envs, 5, device=u.device).clamp(-1, 1) for k in u.cfg.possible_agents}
        _, reward, _, _, _ = env.step(actions)
        finite &= all(bool(torch.isfinite(r).all()) for r in reward.values())
        finite &= all(bool(torch.isfinite(r).all()) for r in u._ep_rew.values())
        integer &= bool(((u.last_rc == u.last_rc.round()) & (u.last_rc.abs() <= 100)).all())
    check(finite, "all reward terms stay finite, including outside the slowdown zone")
    check(integer, "all actions map to integer RC inside the configured range")
    fresh_start(u)
    for _ in range(u.cfg.land_confirm_steps):
        env.step(sticks(u, (0, 0, 0, 0), land=True))
    expected = u.cfg.rew_early_land * u.cfg.num_drones
    check(bool(torch.allclose(u._ep_rew["early_land_loss"],
                              torch.full_like(u._ep_rew["early_land_loss"], expected))),
          "confirmed pre-waypoint land requests retain their penalty")
    fresh_start(u)
    u.at_waypoint[:] = True  # historical entry must NOT authorize land back at the start
    for _ in range(10):
        env.step(sticks(u, (0, 0, 0, 0), land=True))
    check(not bool(u.landing.any()), "latched waypoint plus current off-target pose blocks landing")
    check(bool((u._ep_rew["early_land_loss"] == 0).all()),
          "post-waypoint requests can wait for the gate without repeated request penalties")
    for auto in (False, True):
        previous_auto = u.cfg.auto_land
        u.cfg.auto_land = auto
        fresh_start(u)
        u.at_waypoint[:] = True
        u.pub_pos[:] = u._spots + u._wp_offset
        u.acquired[:] = u.fresh[:] = True
        u.lost[:] = u.taking_off[:] = False
        u.fix_age[:] = 0
        u.kf.v[:] = 0
        u.kf.a[:] = .005
        u.kf.c[:] = .04
        for i in range(int(math.ceil(u.cfg.land_dwell_s / u.step_dt))):
            u._pre_physics_step(sticks(u, (0, 0, 0, 0), land=True))
            if (i+1)*u.step_dt < u.cfg.land_dwell_s:
                check(not bool(u.landing.any()), f"auto_land={auto}: no land before dwell, tick {i+1}")
        # Confirmed requests can have had their vote counter reset while the gate was closed.
        for _ in range(u.cfg.land_confirm_steps):
            u._pre_physics_step(sticks(u, (0, 0, 0, 0), land=True))
        check(bool(u.landing.all()), f"auto_land={auto}: current settled, confident pose permits landing")
        u.cfg.auto_land = previous_auto
    fresh_start(u)


def test_commands_legacy(env, u) -> None:
    print("\n== commands (real Tello SDK: rc -100..100 per channel, land) ==")
    lo, hi = u.rc.lo.tolist(), u.rc.hi.tolist()
    print("  range: " + ", ".join(f"{ch} {int(a):+d}..{int(b):+d}" for ch, a, b in zip(("right", "forward", "up", "yaw"), lo, hi)))
    fresh_start(u)
    inside = True
    for _ in range(60):  # random policy outputs, any magnitude
        acts = {k: (2.5 * torch.randn(u.num_envs, 5, device=u.device)).clamp(-1, 1) for k in u.cfg.possible_agents}
        for k in acts:
            acts[k][:, 4] = -1.0  # no land in this part
        env.step(acts)
        rc = u.last_rc
        inside &= bool(((rc >= u.rc.lo - 1e-6) & (rc <= u.rc.hi + 1e-6)).all()) and bool((rc == rc.round()).all())
    check(inside, f"random policy outputs only ever produce integer rc inside {int(lo[0])}..{int(hi[0])}")
    ext = u.rc.to_rc(torch.tensor([[1.0] * 4, [-1.0] * 4], device=u.device))
    check(ext[0].tolist() == [100.0] * 4 and ext[1].tolist() == [-100.0] * 4,
          f"action +-1 is rc +-100 on every channel: +1 -> {ext[0].tolist()}, -1 -> {ext[1].tolist()}")

    # land gate: before the waypoint, land is blocked and each request costs rew_early_land
    # (auto_land: the request is ignored, free)
    auto = u.cfg.auto_land
    fresh_start(u)
    n = 3 * u.cfg.land_confirm_steps
    for _ in range(n):
        env.step(sticks(u, (0, 0, 0, 0), land=True))
    pen = u._ep_rew["early_land_loss"].mean().item() / u.cfg.num_drones
    want = 0.0 if auto else 3 * u.cfg.rew_early_land
    check(not u.landing.any().item() and abs(pen - want) < 1e-3,
          f"before the waypoint land is {'ignored (auto_land)' if auto else 'blocked and penalized'} "
          f"({n} steps of land: {pen:.0f} per drone, no land sent)")

    fresh_start(u)
    u.at_waypoint[:] = True  # inside the bubble: land goes out (auto_land: without asking), no penalty
    for _ in range(int(6.0 / u.step_dt)):
        env.step(sticks(u, (0, 0, 0, 0), land=not auto))
        if u.outcome_valid.all():
            break
    landed = u.outcome_landed.mean().item()
    check(landed == u.cfg.num_drones, f"land: all {u.cfg.num_drones} drones touch down ({landed:.2f} on average) with no crash "
          f"({u.outcome_crash.float().mean():.0%})" if landed == u.cfg.num_drones else f"land: only {landed:.2f} drones touched down")
    full = u.cfg.rew_time * (u.max_episode_length - 1) * u.cfg.num_drones
    got = u.outcome_rew["landing_loss"].mean().item()
    check(abs(got - full) < 0.02 * abs(full), f"landing off the spot early still pays the whole episode's landing loss ({got:.0f} vs {full:.0f})")
    lr = u.outcome_rew["land_reward"].mean().item() / u.cfg.num_drones
    el = u.outcome_rew["early_land_loss"].abs().max().item()
    check(abs(lr - u.cfg.rew_land) < 1e-3 and el < 1e-6,
          f"after the waypoint land {'is sent automatically' if auto else 'is free'} and touchdown pays rew_land ({lr:.0f} per drone, early-land loss {el:.0f})")

    # waypoint phase: closing in pays until the waypoint is reached, then nothing
    for reached in (False, True):
        fresh_start(u)
        u.at_waypoint[:] = reached
        before = u._ep_rew["closer_reward"].clone()
        for _ in range(int(2.0 / u.step_dt)):
            env.step(sticks(u, (0, 40, 30, 0)))  # forward and up: toward the waypoint
            u.at_waypoint[:] = reached
        gained = (u._ep_rew["closer_reward"] - before).mean().item()
        if reached:
            check(abs(gained) < 1e-6, f"after reaching its waypoint a drone earns no closer reward ({gained:.3f})")
        else:
            check(gained > 1.0, f"before reaching its waypoint, closing on it pays (+{gained:.1f} per env in 2 s)")
    # entering the waypoint pays rew_waypoint once per drone
    fresh_start(u)
    E = torch.arange(u.num_envs, device=u.device)
    target = u._spots + u._wp_offset + u.scene.env_origins.unsqueeze(1)
    for i, drone in enumerate(u._drones):
        pose = drone.data.root_state_w[:, :7].clone()
        pose[:, :3] = target[:, i]
        drone.write_root_pose_to_sim(pose, E)
        drone.write_root_velocity_to_sim(torch.zeros(u.num_envs, 6, device=u.device), E)
    u._prev_dist[:] = 0.0
    for _ in range(int(4.0 / u.step_dt)):
        env.step(sticks(u, (0, 0, 0, 0)))
        if u.at_waypoint.all():
            break
    for _ in range(int(1.0 / u.step_dt)):  # stay inside: no second payment
        env.step(sticks(u, (0, 0, 0, 0)))
    live = ~u.outcome_valid
    paid = u._ep_rew["waypoint_reward"][live] / u.cfg.num_drones
    reached = u.at_waypoint[live].float().mean().item()
    # (facing the wall, a drone at a middle spot's waypoint sees no tag: tag 10 is too big to fit
    # in the frame from 1.1 m and tags 8/9 are outside the field of view, so it registers only
    # once it turns toward one; that's the sensing geometry, not the reward)
    check(reached > 0.3 and torch.allclose(paid, u.at_waypoint[live].float().mean(-1) * u.cfg.rew_waypoint),
          f"entering the waypoint pays {u.cfg.rew_waypoint:.0f} once per drone that registers ({reached:.0%} "
          f"registered facing the wall; {paid.mean():.0f} per drone after 1 s more inside)")

    # the waypoint counts only when slow: the filter's speed estimate under waypoint_max_speed
    registered, speeds = {}, []
    fast_ok, n_fast = True, 0
    for limit in (0.0, u.cfg.waypoint_max_speed):  # limit 0: nothing is ever slow enough
        u.cfg.waypoint_max_speed = limit
        fresh_start(u)
        for i, drone in enumerate(u._drones):
            pose = drone.data.root_state_w[:, :7].clone()
            pose[:, :3] = target[:, i]
            drone.write_root_pose_to_sim(pose, E)
            drone.write_root_velocity_to_sim(torch.zeros(u.num_envs, 6, device=u.device), E)
        ever = torch.zeros_like(u.at_waypoint)
        seen = torch.zeros_like(u.at_waypoint)  # had a fresh fix inside the waypoint
        for _ in range(int(2.0 / u.step_dt)):  # (auto_land: registered drones land, episodes end)
            env.step(sticks(u, (0, 0, 0, 0)))
            ever |= u._waypoint_event
            if limit == 0.0 and u._fast_entry.any():  # too fast: -500 per 0.3 m/s over, episode over
                ev = u._fast_entry.any(-1)
                want = (u.cfg.rew_fast_entry_per_mps * u.est_speed * u._fast_entry.float()).sum(-1)[ev]
                got = u.outcome_rew["fast_entry_loss"][ev]
                fast_ok &= bool(torch.allclose(got, want, atol=1e-3)) and bool(u.outcome_fast_entry[ev].all())
                n_fast += int(ev.sum())
            inside = torch.linalg.norm(u.pub_pos - (u._spots + u._wp_offset), dim=-1) < u.cfg.waypoint_radius
            seen |= inside & u.fresh & ~u.landing
            if limit > 0:
                speeds.append(u.est_speed[u.fresh])
        registered[limit] = (ever & seen).sum().item() / max(1, seen.sum().item())
    sp = torch.cat(speeds)
    check(registered[0.0] == 0.0 and registered[limit] > 0.9,
          f"the waypoint counts only below waypoint_max_speed: of the drones with a fix inside it, hovering "
          f"registers {registered[limit]:.0%} "
          f"with {limit} m/s, {registered[0.0]:.0%} with 0 (hover speed estimate median {sp.median():.2f}, "
          f"p90 {sp.quantile(0.9):.2f} m/s)")
    u.cfg.waypoint_max_speed = limit
    check(fast_ok and n_fast > 0, f"entering faster than waypoint_max_speed fails the episode and costs "
          f"{-u.cfg.rew_fast_entry_per_mps * 0.3:.0f} per 0.3 m/s over ({n_fast} episodes checked)")

    # braking within slow_zone_radius of tag 10 pays rew_slowdown per m/s of new low speed
    fresh_start(u)
    for i, drone in enumerate(u._drones):
        pose = drone.data.root_state_w[:, :7].clone()
        pose[:, :3] = u.scene.env_origins + torch.tensor([1.0, 0.9 - 0.6 * i, 1.0], device=u.device)  # in the zone, below the waypoints
        vel = torch.zeros(u.num_envs, 6, device=u.device)
        vel[:, 0] = 1.0  # 1 m/s away from the wall, sticks centred: the firmware brakes it
        drone.write_root_pose_to_sim(pose, E)
        drone.write_root_velocity_to_sim(vel, E)
    for _ in range(int(2.0 / u.step_dt)):
        env.step(sticks(u, (0, 0, 0, 0)))
    paid = u._ep_rew["slowdown_reward"][~u.outcome_valid] / u.cfg.num_drones
    check(bool((paid > 0.5 * u.cfg.rew_slowdown).all() & (paid <= 1.05 * u.cfg.rew_slowdown).all()),
          f"braking from 1 m/s inside the slow zone pays rew_slowdown per m/s shed ({paid.mean():.0f} per drone, "
          f"at most {u.cfg.rew_slowdown:.0f})")

    # two drones within collision_distance: each pays rew_collision (not the crash loss), episode ends
    if u.cfg.num_drones > 1:
        fresh_start(u)
        for _ in range(3):
            env.step(sticks(u, (0, 0, 0, 0)))
        pose = u._drones[0].data.root_state_w[:, :7].clone()
        pose[:, 0] += 0.1
        u._drones[1].write_root_pose_to_sim(pose, E)
        u._drones[1].write_root_velocity_to_sim(torch.zeros(u.num_envs, 6, device=u.device), E)
        env.step(sticks(u, (0, 0, 0, 0)))
        col, cr = u.outcome_rew["collision_loss"], u.outcome_rew["crash_loss"]
        check(bool(u.outcome_collision.all()) and torch.allclose(col, torch.full_like(col, 2 * u.cfg.rew_collision))
              and float(cr.abs().max()) < 1e-6,
              f"two drones 0.1 m apart: collision ends the episode and each pays rew_collision "
              f"({col.mean() / 2:.0f} per drone, crash loss {cr.abs().max():.0f})")

    # moving away costs rew_away_ratio of the closer rate
    fresh_start(u)
    d0 = torch.linalg.norm(u._positions() - (u._spots + u._wp_offset), dim=-1).sum(-1)
    before = u._ep_rew["closer_reward"].clone()
    for _ in range(int(1.5 / u.step_dt)):
        env.step(sticks(u, (0, -30, 0, 0)))  # backward, away from tag 10 and the waypoints
    d1 = torch.linalg.norm(u._positions() - (u._spots + u._wp_offset), dim=-1).sum(-1)
    live = ~u.outcome_valid
    paid = (u._ep_rew["closer_reward"] - before)[live]
    lost_m = (d1 - d0)[live]
    want = -u.cfg.rew_away_ratio * u.cfg.rew_closer_scale * lost_m
    ok = bool(live.any()) and (lost_m > 0.1).all().item() and torch.allclose(paid, want, rtol=0.05, atol=0.05)
    check(ok, f"moving away costs {u.cfg.rew_away_ratio} x the closer rate ({paid.mean():.2f} for {lost_m.mean():.2f} m lost per env, "
              f"expected {want.mean():.2f})")


# -- motion ---------------------------------------------------------------------


def step_response(u, rc, seconds: float):
    fresh_start(u)
    for _ in range(int(0.5 / u.step_dt)):  # settle
        u.step(sticks(u, (0, 0, 0, 0)))
    traces = []
    for _ in range(int(seconds / u.step_dt)):
        u.step(sticks(u, rc))
        v = u._stack("root_lin_vel_w")
        yaw = _yaw_of(u._stack("root_quat_w"))
        c, s = torch.cos(yaw), torch.sin(yaw)
        fwd = c * v[..., 0] + s * v[..., 1]
        right = s * v[..., 0] - c * v[..., 1]
        yaw_rate_cw = -torch.rad2deg(u._stack("root_ang_vel_w")[..., 2])
        traces.append(torch.stack([right, fwd, v[..., 2], yaw_rate_cw], -1).mean(dim=(0, 1)))
    return torch.stack(traces).cpu().numpy()  # (T, 4): right, forward, up m/s; yaw deg/s (cw)


def test_motion(env, u) -> None:
    fit = load_motion_fit(u.cfg.rc.fit_path)
    print("\n== command timing (one policy step = one rc packet) ==")
    check(abs(u.step_dt - 1.0 / u.cfg.command_rate_hz) < 1e-9,
          f"policy step {u.step_dt * 1000:.1f} ms = 1 / command_rate_hz ({u.cfg.command_rate_hz:.2f} Hz, "
          f"{u.cfg.decimation} physics steps of {u.physics_dt * 1000:.0f} ms)")
    fresh_start(u)
    lost = []
    for _ in range(100):
        u.step(sticks(u, (0, 0, 0, 0)))
        lost.append(u.rc_lost.float().mean().item())
    want = 0.5 * sum(u.cfg.link_loss_range)
    check(abs(np.mean(lost) - want) < 0.02, f"rc packets lost {np.mean(lost):.1%} (link_loss_range mean {want:.1%})")

    print(f"\n== motion vs fit ({fit['source']}) ==")
    names = ("right", "forward", "up", "yaw")
    for ch in ("forward", "right", "up", "yaw"):
        k = names.index(ch)
        # 3 s at rc 50 stays inside the room; strafe left, away from the tag-5 wall at Y = 1.7 m
        rc_val = {"yaw": 100, "right": -50}.get(ch, 50)
        rc = [0, 0, 0, 0]
        rc[k] = rc_val
        trace = step_response(u, rc, 3.0)[:, k]
        gain = fit[ch]["gain"] * (180 / math.pi if ch == "yaw" else 1.0)
        want = gain * rc_val
        got = float(np.mean(trace[-10:]))
        sgn = 1.0 if want >= 0 else -1.0
        rise = sgn * trace >= 0.63 * abs(want)
        t63 = (np.argmax(rise) + 1) * u.step_dt if rise.any() else float("inf")
        want_t = fit[ch]["tau"] + fit[ch]["delay"]
        unit = "deg/s" if ch == "yaw" else "m/s"
        check(abs(got - want) <= 0.15 * abs(want), f"rc {ch} {rc_val}: steady {got:.3f} {unit} (fit {want:.3f})")
        check(t63 <= 2.0 * want_t + 0.15, f"rc {ch} {rc_val}: 63% rise in {t63:.2f} s (fit tau + delay {want_t:.2f} s)")
    db = int(fit["forward"]["deadband"])
    if db > 1 and u.rc.hi[1] >= db - 1:
        trace = step_response(u, (0, db - 1, 0, 0), 1.5)[:, 1]
        check(abs(trace[-5:].mean()) < 0.02, f"rc forward {db - 1} (inside the deadband) does not move ({trace[-5:].mean():.3f} m/s)")


# -- baselines ------------------------------------------------------------------


def test_baselines(env, u, episodes: int) -> None:
    """Zero sticks: never succeeds, and its landing / no-tag losses equal -0.5 x the off-spot
    and tag-less drone-steps it accumulated."""
    print("\n== baselines (zero sticks) ==")
    fresh_start(u)
    off_steps = torch.zeros(u.num_envs, device=u.device)
    blind_steps = torch.zeros(u.num_envs, device=u.device)
    totals = {"success": 0.0, "landing": 0.0, "no_tag": 0.0, "crash": 0.0, "time": 0.0, "off": 0.0, "blind": 0.0}
    done_count = 0
    while done_count < episodes:
        env.step(sticks(u, (0, 0, 0, 0)))
        off_steps += (~u.land_ok).float().sum(-1)
        blind_steps += (~u.fresh & ~u.landed & ~u.taking_off).float().sum(-1)
        fin = u.outcome_valid
        if fin.any():
            idx = fin.nonzero().flatten()
            done_count += len(idx)
            totals["success"] += u.outcome_success[idx].float().sum().item()
            totals["landing"] += u.outcome_rew["landing_loss"][idx].sum().item()
            totals["no_tag"] += u.outcome_rew["no_tag_loss"][idx].sum().item()
            totals["crash"] += u.outcome_rew["crash_loss"][idx].sum().item()
            parts = sum(u.outcome_rew[k][idx] for k in u.REWARD_TERMS if k != "total")
            # relative: float32 sums of ~15 terms over ~1000 steps drift ~1e-5 of the episode total
            err = (parts - u.outcome_rew["total"][idx]).abs() / (u.outcome_rew["total"][idx].abs() + 1.0)
            err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))  # NaN: a failure, not 0
            totals["terms_err"] = max(totals.get("terms_err", 0.0), err.max().item())
            totals["time"] += u.outcome_time[idx].sum().item()
            totals["off"] += off_steps[idx].sum().item()
            totals["blind"] += blind_steps[idx].sum().item()
            off_steps[idx] = 0.0
            blind_steps[idx] = 0.0
    m = {k: v / done_count for k, v in totals.items()}
    print(f"  zero     : success {m['success']:.0%}, time {m['time']:.1f} s, landing loss {m['landing']:.0f}, "
          f"no-tag loss {m['no_tag']:.0f}, crash loss {m['crash']:.0f}  ({done_count} episodes)")
    check(m["success"] == 0.0, f"zero sticks never succeed ({m['success']:.0%})")
    check(totals.get("terms_err", 0.0) < 1e-4, f"the {len(u.REWARD_TERMS) - 1} logged reward terms add up to the total reward "
          f"(worst episode mismatch {totals.get('terms_err', 0.0):.1e} of the episode total)")
    # counts are read after the auto-reset, so an episode's last step may differ by 1 per drone
    slack = 0.5 * u.cfg.num_drones + 1e-3
    check(abs(m["landing"] - u.cfg.rew_time * m["off"]) <= slack,
          f"landing loss = -0.5 x off-spot drone-steps ({m['landing']:.0f} vs {u.cfg.rew_time * m['off']:.0f})")
    check(abs(m["no_tag"] - u.cfg.rew_no_fix * m["blind"]) <= slack,
          f"no-tag loss = -0.5 x drone-steps without a tag ({m['no_tag']:.0f} vs {u.cfg.rew_no_fix * m['blind']:.0f})")


def test_central(env, u) -> None:
    from action_space_kit.rl.sb3_shared import Sb3CentralVecEnv

    print(f"\n== central controller (one policy for all {u.cfg.num_drones} drone(s)) ==")
    venv = Sb3CentralVecEnv(env)
    obs = venv.reset()
    D, dim = u.cfg.num_drones, u.cfg.observation_spaces[u.cfg.possible_agents[0]]
    check(obs.shape == (u.num_envs, D * dim) and venv.action_space.shape == (D * 5,),
          f"one row per env: observation {obs.shape[1]} (= {D} x {dim}), action {venv.action_space.shape[0]} (= {D} x 5)")
    acts = np.random.uniform(-1, 1, (u.num_envs, D * 5)).astype(np.float32)
    acts[:, 4::5] = -1.0  # no land
    venv.step_async(acts)
    obs, rew, dones, _ = venv.step_wait()
    sent = u._actions.reshape(u.num_envs, -1).cpu().numpy()
    live = ~dones  # an env that finished this step reset and cleared its stored actions
    check(np.allclose(sent[live], acts[live], atol=1e-6),
          f"action slice i drives drone i (all {D * 5} numbers reach their drones; {int(live.sum())} running envs)")
    team = u._ep_rew["total"].cpu().numpy()  # this episode's sum over drones, one step in
    check(np.allclose(rew[~dones], team[~dones], atol=1e-3), f"the central reward is the team reward (sum over the {D} drone(s))")


def test_kalman(env, u) -> None:
    from action_space_kit.tags.kalman import FILTER_PATH, KalmanCfg, KalmanCV

    print(f"\n== Kalman filter ({FILTER_PATH.name}, shared with the laptop) ==")
    # 1) one implementation: numpy (laptop) and torch (sim) agree on the same fixes
    rng = np.random.default_rng(1)
    kcfg = KalmanCfg()
    k_np = KalmanCV(kcfg, (2,))
    k_t = KalmanCV(kcfg, (2,), xp=torch, device=u.device)
    worst = 0.0
    for k in range(150):
        z = np.stack([[5.0 - 0.03 * k, 0.1, 1.0], [6.0, -0.2 + 0.02 * k, 1.2]]) + rng.normal(0, 0.15, (2, 3))
        if k == 90:
            z[0, 1] += 0.9  # a pose flip
        seen = rng.random(2) > 0.2
        k_np.predict(u.step_dt)
        k_t.predict(u.step_dt)
        k_np.update(z, mask=seen)
        k_t.update(torch.as_tensor(z, dtype=torch.float32, device=u.device), mask=torch.as_tensor(seen, device=u.device))
        worst = max(worst, float(np.abs(k_t.p.cpu().double().numpy() - k_np.p).max()))
    check(worst < 1e-3, f"numpy (laptop) and torch (sim) run the same filter (max difference {worst * 1000:.3f} mm over 150 ticks)")

    # 2) in flight: the published (filtered) position is closer to the truth than the raw fix
    fresh_start(u)
    for _ in range(int(1.0 / u.step_dt)):  # settle into takeoff / hover
        env.step(sticks(u, (0, 0, 0, 0)))
    fly = sticks(u, (0, 35, 0, 0))
    err_raw, err_kf, n = 0.0, 0.0, 0
    for _ in range(int(4.0 / u.step_dt)):
        env.step(fly)
        ok = u.kf.ready & ~u.lost & ~u.landed & ~u.taking_off & ~u.outcome_valid.unsqueeze(-1)
        truth = u._positions()
        err_raw += (torch.linalg.norm(u.raw_pos - truth, dim=-1) * ok).sum().item()
        err_kf += (torch.linalg.norm(u.pub_pos - truth, dim=-1) * ok).sum().item()
        n += ok.sum().item()
    raw_cm, kf_cm = 100 * err_raw / max(n, 1), 100 * err_kf / max(n, 1)
    check(n > 0 and kf_cm < raw_cm, f"flying at rc 35: filtered position {kf_cm:.1f} cm from the truth vs raw fix {raw_cm:.1f} cm "
          f"(noise unchanged, {n} drone-steps)")


def test_search(env, u, prob: float, spawn_sep: float) -> None:
    """Starts turned away from the tags (spawn_search_prob) and the 0.2 m spawn spacing."""
    cfg = u.cfg
    other_sep, cfg.spawn_min_separation = cfg.spawn_min_separation, spawn_sep
    print("\n== search starts (turned away from the tags) and close spawns ==")
    cfg.spawn_search_prob = prob
    away, close, limit_ok = [], 0.0, True
    for _ in range(20):
        fresh_start(u)
        pos = u._positions()
        q = u._stack("root_quat_w")
        yaw = torch.atan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))
        off = torch.remainder(yaw - torch.atan2(-pos[..., 1], -pos[..., 0]) + math.pi, 2 * math.pi) - math.pi
        away.append((off.abs() > cfg.spawn_search_yaw[0] - 1e-3).float().flatten())
        if cfg.num_drones > 1:
            pair = u._pairwise_distance(pos)
            close = max(close, (pair < cfg.collision_distance).any().float().item())
            want = torch.minimum(torch.full_like(pair, cfg.collision_distance), pair - cfg.start_contact_margin)
            limit_ok &= bool(torch.allclose(u._pair_limit, want, atol=1e-4))
    frac = torch.cat(away).mean().item()
    check(abs(frac - prob) < 0.05, f"{frac:.0%} of drones start turned away from the tags (spawn_search_prob {prob:.0%}, "
          f"yaw {math.degrees(cfg.spawn_search_yaw[0]):.0f}-{math.degrees(cfg.spawn_search_yaw[1]):.0f} deg off tag 10)")
    if cfg.num_drones > 1:
        check(close > 0 and limit_ok, "drones may spawn closer than collision_distance; each pair's collision limit is "
              f"min({cfg.collision_distance}, start distance - {cfg.start_contact_margin}) m")
        fresh_start(u)
        u.step(sticks(u, (0, 0, 0, 0)))
        check(not u._collision.any().item(), "no collision on the first step after a reset")
        start = u._pairwise_distance(u._positions()).amin(dim=(-1, -2))
        hit = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
        for _ in range(int(5.0 / u.step_dt)):
            u.step(sticks(u, (0, 0, 0, 0)))
            hit |= u._ep_collision
        near = start < cfg.collision_distance
        print(f"  zero sticks for 5 s: {hit[near].float().mean():.0%} of envs whose closest pair started < "
              f"{cfg.collision_distance} m collide (hover drift); {hit[~near].float().mean():.0%} of the rest")

    cfg.spawn_search_prob = 1.0
    fresh_start(u)
    for _ in range(int(4.0 / u.step_dt)):  # takeoff (floor starts) and hover, no yaw
        u.step(sticks(u, (0, 0, 0, 0)))
    blind = 1.0 - u.acquired.float().mean().item()
    check(blind > 0.95, f"{blind:.0%} of drones turned away see no tag in 4 s without turning")
    for _ in range(int(20.0 / u.step_dt)):  # rc yaw 40 ~ 24 deg/s: a full turn in 15 s
        u.step(sticks(u, (0, 0, 0, 40)))
    found = u.acquired.float().mean().item()
    check(found > 0.9, f"{found:.0%} of them get a tag fix after turning (rc yaw 40 for 20 s)")
    cfg.spawn_search_prob, cfg.spawn_min_separation = 0.0, other_sep


def test_single(env, u) -> None:
    cfg = u.cfg
    print("\n== single drone from the floor (takeoff, then the sticks) ==")
    D = cfg.num_drones
    dim = cfg.observation_spaces[cfg.possible_agents[0]]
    obs = env.reset()[0]["drone_0"]
    fresh_start(u)
    pos0 = u._positions()
    check(D == 1 and dim == 25, f"one drone, observation {dim} (= 24 + taking-off flag)")
    check(((pos0[..., 2] - cfg.rest_height).abs() < 0.01).all().item() and u.taking_off.all().item(),
          f"starts on the floor (z {pos0[..., 2].mean():.3f} m), takeoff sent")
    xy = pos0[..., :2]
    r = torch.linalg.norm(xy, dim=-1)
    cx, cy = cfg.spawn_box_center
    h = 0.5 * cfg.spawn_box_size + 1e-3
    check(((xy[..., 0] - cx).abs() <= h).all().item() and ((xy[..., 1] - cy).abs() <= h).all().item()
          and (r >= cfg.spawn_range[0] - 1e-3).all().item() and (r <= cfg.spawn_range[1] + 1e-3).all().item(),
          f"start in the {cfg.spawn_box_size:.0f} m box, {r.min():.2f}-{r.max():.2f} m from tag 10")
    spot_y = torch.round(u._spots[:, 0, 1] / cfg.spot_spacing * 2).unique().numel()
    check(spot_y == cfg.spot_count, f"the drone gets every one of the {cfg.spot_count} spots across envs ({spot_y} seen)")

    # full sticks + land during takeoff: ignored, the firmware climbs straight up
    push = sticks(u, (100, 100, 100, 100), land=True)
    for _ in range(5):
        obs = env.step(push)[0]["drone_0"]
    check(torch.allclose(obs[:, -1], u.taking_off[:, 0].float()), "the observation's last number is the taking-off flag")
    ep = u._ep_rew
    check((cfg.reward_version >= 2 or ep["closer_reward"].abs().max().item() < 1e-6)
          and ep["no_tag_loss"].abs().max().item() < 1e-6,
          "takeoff pays no no-tag loss (version 2 retains consistent potential shaping)")
    check(torch.allclose(ep["landing_loss"], torch.full_like(ep["landing_loss"], 5 * cfg.rew_time)),
          f"... but the landing loss runs ({ep['landing_loss'].mean():.1f} after 5 steps)")
    check(not u.landing.any().item(), "land asked for during takeoff is not sent")
    steps, t_up = 0, torch.full((u.num_envs,), float("nan"), device=u.device)
    z_up = torch.zeros(u.num_envs, device=u.device)
    crashed = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    hold = sticks(u, (0, 0, 0, 0))  # after its takeoff a drone flies the sticks: hold it
    while u.taking_off.any() and steps < int(6.0 / u.step_dt):
        was = u.taking_off[:, 0].clone()
        env.step(push if steps < 3 else hold)
        steps += 1
        crashed |= u.outcome_valid
        up = was & ~u.taking_off[:, 0]
        t_up = torch.where(up, torch.full_like(t_up, steps * u.step_dt), t_up)
        z_up = torch.where(up, u._positions()[:, 0, 2], z_up)
    drift = torch.linalg.norm(u._positions()[..., :2] - xy, dim=-1).max().item()
    check(not u.taking_off.any().item() and not crashed.any().item(),
          f"every takeoff completes without a crash ({(5 * u.step_dt + t_up).mean():.1f} s mean, {int(crashed.sum())} ended)")
    check((z_up > cfg.takeoff_height[0] - 0.02).all().item() and (z_up < cfg.takeoff_height[1] + 0.1).all().item(),
          f"takeoff ends at {z_up.min():.2f}-{z_up.max():.2f} m (takeoff_height {cfg.takeoff_height})")
    check(drift < 0.15, f"full sticks during takeoff are ignored (max horizontal drift {drift * 100:.0f} cm)")
    # now the sticks fly it: forward toward tag 10
    fwd = sticks(u, (0, 50, 0, 0))
    before = u._positions()[..., 0].clone()
    for _ in range(int(1.5 / u.step_dt)):
        env.step(fwd)
    moved = (before - u._positions()[..., 0]).mean().item()
    check(moved > 0.2, f"after takeoff the sticks fly it (rc forward 50 for 1.5 s: {moved:.2f} m toward tag 10)")


def test_curriculum(env, u):
    print("\n== mixed-start curriculum ==")
    cfg = u.cfg
    cfg.spawn_short_fraction, cfg.spawn_hover_fraction = 0.30, 0.15
    try:
        fresh_start(u)
        pos, kinds = u._positions().clone(), u._start_kind.clone()
        full, short, hover = kinds == 0, kinds == 1, kinds == 2
        check(all(bool(mask.any()) for mask in (full, short, hover)), "all three start modes are sampled")
        r = torch.linalg.norm(pos[full, :, :2], dim=-1)
        check(bool(((r >= 6) & (r <= 7)).all()), "configured episodes retain the full 6-7 m start")
        check(bool((u.taking_off[~hover]).all()) and not bool(u.taking_off[hover].any()),
              "full and short approaches start with takeoff; hover replay starts airborne")
        check(bool(((pos[hover, :, :2] - u._spots[hover, :, :2]).abs() <= .10001).all()),
              "hover replay starts within 10 cm per axis of its own assigned pad")
        check(bool(((pos[short, :, 0] >= 1.5) & (pos[short, :, 0] <= 5.5)).all()),
              "short approaches cover the configured intermediate distances")
        initial = u._start_positions.clone()
        u.episode_length_buf[:] = u.max_episode_length - 2
        obs, _, _, _, info = env.step(sticks(u, (0, 0, 0, 0)))
        records = info["episode_records"]
        names = ("configured", "short_approach", "hover_landing")
        check(len(records) == u.num_envs and all(
            row["start_kind"] == names[int(kinds[row["env_id"]])]
            and np.allclose(row["start_positions"], initial[row["env_id"]].cpu().numpy())
            for row in records), "terminal records retain the finished episode's start mode and pose")
        check(obs["drone_0"].shape == (u.num_envs, 25), "mixed starts preserve the 25-number observation")
    finally:
        cfg.spawn_short_fraction = cfg.spawn_hover_fraction = 0.0
        fresh_start(u)


def test_precision(env, u):
    print("\n== continuous precision shaping ==")
    cfg = u.cfg
    fields = {"rew_precision_potential": 80., "rew_settle_position": 0., "rew_settle_speed": 0.,
              "spawn_short_fraction": .3, "spawn_hover_fraction": .15}
    original = {key: getattr(cfg, key) for key in fields}
    try:
        for key, value in fields.items():
            setattr(cfg, key, value)
        fresh_start(u)
        initial = u._potential(u._positions())
        check(torch.allclose(u._prev_potential, initial, atol=1e-4),
              "reset potential matches the actual state for all start modes")
        u.at_waypoint[:] = True
        check(torch.equal(initial, u._potential(u._positions())),
              "latching the waypoint introduces no new precision cost")
        finite = True
        for _ in range(40):
            obs, reward, _, _, _ = env.step(sticks(u, (0, 0, 0, 0)))
            finite &= all(bool(torch.isfinite(v).all()) for v in (*obs.values(), *reward.values()))
        check(finite, "precision shaping produces finite observations and rewards")
        check(bool((u._ep_rew["settling_loss"] == 0).all()), "milestone-triggered settling costs are disabled")
        ids = torch.arange(0, u.num_envs, 2, device=u.device)
        u._reset_idx(ids)
        check(torch.allclose(u._prev_potential[ids], u._potential(u._positions())[ids], atol=1e-4),
              "partial resets initialize precision potential from reset pose and zero velocity")
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)
        fresh_start(u)


def test_team(env, u):
    print("\n== simultaneous four-drone floor starts ==")
    fresh_start(u)
    check(u.cfg.num_drones == 4 and u.taking_off.all().item(), "all four drones begin takeoff together")
    initial = u._positions().clone()
    check(torch.allclose(initial[..., 2], torch.full_like(initial[..., 2], u.cfg.rest_height)),
          "all drones start on the floor")
    width = 48 if u.cfg.peer_observations else 25
    check(all(value.shape == (u.num_envs, width) for value in u._get_observations().values()),
          f"each drone has its expected {width}-feature observation")
    ended = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    for _ in range(int(5.0 / u.step_dt)):
        env.step(sticks(u, (0, 0, 0, 0)))
        ended |= u.outcome_valid
    check(not u.taking_off.any().item() and not ended.any().item(),
          "all four simultaneous takeoffs finish without an episode ending")
    check((torch.linalg.norm(u._positions()[..., :2] - initial[..., :2], dim=-1) < .15).all().item(),
          "simultaneous takeoff retains horizontal separation")
    # Deliberately place one pair too close in room 0: collision termination must
    # affect that whole room, but not neighboring vectorized simulation rooms.
    pose = u._drones[1].data.root_state_w[:1, :7].clone()
    pose[0, :3] = u._drones[0].data.root_pos_w[0] + torch.tensor([.20, 0., 0.], device=u.device)
    u._drones[1].write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=u.device))
    done, _ = u._get_dones()
    check(all(value[0].item() for value in done.values()) and u.outcome_collision[0].item(),
          "a pair inside 25 cm terminates the whole joint episode")
    check(not u.outcome_collision[1:].any().item(), "collision is confined to its own simulation room")
    fresh_start(u)
    records = u.extras["episode_records"]
    check(len(records) == 1 and abs(records[0]["metrics"]["Episode/min_separation_m"] - .20) < .001,
          "the completed collision episode retains its actual minimum separation")
    check(records[0]["metrics"]["Episode/timeout_rate"] == 0., "collision termination is not counted as a timeout")
    u.episode_length_buf[:] = u.max_episode_length - 2
    env.step(sticks(u, (0, 0, 0, 0)))
    records = u.extras["episode_records"]
    check(len(records) == u.num_envs and all(row["metrics"]["Episode/timeout_rate"] == 1. for row in records),
          "every forced timeout is retained in the completed episode records")
    fresh_start(u)
    if u.cfg.collision_substeps:
        # A pair passes inside the limit and separates before the next RC update.
        # The per-physics-step detector must retain that violation and minimum.
        pose = u._drones[1].data.root_state_w[:1, :7].clone()
        original = pose.clone()
        pose[0, :3] = u._drones[0].data.root_pos_w[0] + torch.tensor([.20, 0., 0.], device=u.device)
        ids = torch.tensor([0], device=u.device)
        u._drones[1].write_root_pose_to_sim(pose, env_ids=ids)
        u._apply_action()
        u._drones[1].write_root_pose_to_sim(original, env_ids=ids)
        done, _ = u._get_dones()
        check(all(value[0].item() for value in done.values()) and u.outcome_collision[0].item(),
              "a collision between RC updates remains a joint failure after the pair separates")
        check(abs(float(u.outcome_min_sep[0]) - .20) < .001,
              "minimum separation includes physics substeps between RC updates")
        fresh_start(u)
        check(not u._substep_collision.any().item(), "substep collision state clears on reset")


def main():
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    # every other check assumes the drones start facing tag 10 and >= 0.5 m apart (zero sticks
    # never collide); geometry checks the configured spacing, test_search the rest
    search_prob, u.cfg.spawn_search_prob = u.cfg.spawn_search_prob, 0.0
    spawn_sep = u.cfg.spawn_min_separation
    if args_cli.task in ("AS-Tello-Room-Team-v0", "AS-Tello-Room-Peer-v0"):
        if args_cli.test in ("geometry", "all"):
            test_geometry(u)
        u.cfg.spawn_min_separation = max(spawn_sep, 0.5)
        if args_cli.test in ("search", "all"):
            test_search(env, u, search_prob, spawn_sep)
        if args_cli.test in ("team", "all"):
            test_team(env, u)
        if args_cli.test in ("central", "all"):
            test_central(env, u)
        print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} CHECK(S) FAILED:\n  " + "\n  ".join(FAILURES)))
        env.close()
        return
    if cfg.num_drones == 1:
        if args_cli.test in ("search", "all"):
            test_search(env, u, search_prob, spawn_sep)
        if args_cli.test in ("single", "all"):
            test_single(env, u)
        if args_cli.test in ("curriculum", "all"):
            test_curriculum(env, u)
        if args_cli.test in ("precision", "all"):
            test_precision(env, u)
        if args_cli.test in ("kalman", "all"):
            test_kalman(env, u)
        if args_cli.test in ("baselines", "all"):
            test_baselines(env, u, args_cli.episodes)
        if args_cli.test in ("central", "all"):
            test_central(env, u)
        print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} CHECK(S) FAILED:\n  " + "\n  ".join(FAILURES)))
        env.close()
        return
    if args_cli.test in ("geometry", "all"):
        test_geometry(u)
    u.cfg.spawn_min_separation = max(spawn_sep, 0.5)
    if args_cli.test in ("sensing", "all"):
        test_sensing(env, u)
    if args_cli.test in ("search", "all"):
        test_search(env, u, search_prob, spawn_sep)
    if args_cli.test in ("commands", "all"):
        test_commands(env, u)
    if args_cli.test in ("motion", "all"):
        test_motion(env, u)
    if args_cli.test in ("baselines", "all"):
        test_baselines(env, u, args_cli.episodes)
    if args_cli.test in ("kalman", "all"):
        test_kalman(env, u)
    if args_cli.test in ("central", "all"):
        test_central(env, u)
    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} CHECK(S) FAILED:\n  " + "\n  ".join(FAILURES)))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

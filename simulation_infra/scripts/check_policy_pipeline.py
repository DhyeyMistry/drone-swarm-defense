"""Isaac integration check: inference parity, timeout terminals, and gated touchdown."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", choices=("AS-Tello-Room-Single-v0", "AS-Tello-Room-Team-v0", "AS-Tello-Room-Peer-v0"),
                    help="Defaults to the checkpoint's saved task")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import json

import action_space_kit.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
torch.set_num_threads(1)  # small batch CPU parity checks do not benefit from large thread pools
from action_space_kit.rl.policy_export import norm_path_for
from action_space_kit.rl.run_config import restore_run_config
from action_space_kit.rl.sb3_shared import Sb3CentralVecEnv, Sb3SharedPolicyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drone_deployment_infra"))
from core.room_policy import RoomPolicy


def feed(u):
    fields = {
        "position": u.pub_pos,
        "velocity": u.kf.v,
        "position_variance": u.kf.a,
        "velocity_variance": u.kf.c,
        "target": u._spots,
        "yaw": u.pub_yaw,
        "fix_age": u.fix_age,
        "fresh": u.fresh,
        "acquired": u.acquired,
        "lost": u.lost,
        "height": u.height_meas,
        "taking_off": u.taking_off,
        "landed": u.landed,
    }
    return {k: v[0].detach().cpu().tolist() for k, v in fields.items()}


def main():
    task = args.task or json.loads((Path(args.checkpoint).parent / "params/run.json").read_text())["task"]
    if task not in ("AS-Tello-Room-Single-v0", "AS-Tello-Room-Team-v0", "AS-Tello-Room-Peer-v0"):
        raise ValueError("Pipeline check supports the floor-start single and team room tasks")
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "sb3_cfg_entry_point")
    restore_run_config(args.checkpoint, cfg, agent_cfg)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 2.0  # force multiple timeout/reset cases during firmware takeoff
    central = bool(agent_cfg.get("central", False))
    base = (Sb3CentralVecEnv if central else Sb3SharedPolicyVecEnv)(gym.make(task, cfg=cfg))
    u = base.unwrapped_env
    normal = VecNormalize.load(str(norm_path_for(args.checkpoint)), base)
    normal.training = False
    normal.norm_reward = False
    env = VecFrameStack(normal, n_stack=agent_cfg["frame_stack"])
    model = PPO.load(args.checkpoint, env, device="cpu")
    adapter = RoomPolicy(args.checkpoint)
    obs = env.reset()
    u.episode_length_buf[:] = 0
    records, resets = [], 0
    done = np.array([True])
    for i in range(90):
        sample = feed(u)
        sample["reset"] = i == 0 or bool(done[0])
        if sample["reset"]:
            adapter.reset()
        result = adapter.step(sample)
        per_drone = u._get_observations()
        raw = torch.cat([per_drone[name][0] for name in cfg.possible_agents]).cpu().numpy()
        np.testing.assert_allclose(result["observation"], raw, atol=2e-6)
        action, _ = model.predict(obs, deterministic=True)
        shaped_action = action.reshape(cfg.num_drones, 5)
        np.testing.assert_allclose(result["action"], shaped_action, rtol=1e-4, atol=2e-5)
        sample["expected_observation"] = raw.tolist()
        sample["expected_action"] = shaped_action.tolist()
        records.append(sample)
        obs, reward, done, infos = env.step(action)
        assert np.isfinite(reward).all()
        np.testing.assert_array_equal(result["rc"], u.last_rc[0].cpu().numpy()) if not done[0] else None
        if done[0]:
            resets += 1
            assert infos[0]["TimeLimit.truncated"]
            assert not np.array_equal(infos[0]["terminal_observation"], obs[0])
    assert resets >= 2
    replay = ("outputs/shared_peer_policy_replay.jsonl" if cfg.peer_observations and not central else
              "outputs/peer_policy_replay.jsonl" if cfg.peer_observations else
              "outputs/team_policy_replay.jsonl" if cfg.num_drones > 1 else "outputs/policy_replay.jsonl")
    Path(replay).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"[ok] 90 simulator/adapter observation, action and RC comparisons for {cfg.num_drones} drone(s); {resets} timeout resets")

    # Controlled touchdown: the policy requests land while zero sticks hold over the target.
    # Remove measurement/link randomness to isolate the gate+firmware integration.
    cfg.episode_length_s = 75.0
    cfg.fix_noise = cfg.fix_bias = 0.0
    cfg.tag_sensor.pixel_noise = cfg.tag_sensor.dropout = cfg.tag_sensor.blur_max_loss = 0.0
    cfg.link_loss_range = (0.0, 0.0)
    cfg.spawn_yaw_noise = 0.0
    successes = 0
    for target_index in range(4):
        cfg.fixed_spot_index = target_index if cfg.num_drones == 1 else -1
        env.reset()
        u.episode_length_buf[:] = 0
        indices = (torch.arange(cfg.num_drones, device=u.device) + target_index) % 4
        u._spots[0] = u._spot_table[indices]
        u.taking_off[:] = False
        u._fix_bias[:] = 0
        u._link_loss[:] = 0
        # Teleport orientation as well as position: the far-start heading may hide all
        # tags over this pad. A zero-stick gate fixture cannot turn to reacquire them.
        for drone_index, drone in enumerate(u._drones):
            pose = drone.data.root_state_w.clone()
            pose[:, :3] = u._spots[:, drone_index] + u._wp_offset + u.scene.env_origins
            yaw = torch.atan2(-u._spots[:, drone_index, 1], -u._spots[:, drone_index, 0])
            zero = torch.zeros_like(yaw)
            pose[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
            pose[:, 7:] = 0
            drone.write_root_pose_to_sim(pose[:, :7])
            drone.write_root_velocity_to_sim(pose[:, 7:])
            u.pub_yaw[:, drone_index] = yaw
        u.pub_pos[:] = u._spots + u._wp_offset
        hold_land = np.tile(np.array([[0., 0., 0., 0., 1.]], np.float32),
                            (1, cfg.num_drones) if central else (cfg.num_drones, 1))
        for tick in range(int(12 / u.step_dt)):
            _, _, done, _ = env.step(hold_land)
            if tick < int(cfg.land_dwell_s / u.step_dt):
                assert not u.landing.any(), "Land sent before dwell completed"
            if done[0]:
                successes += int(u.outcome_success[0])
                break
        if not (bool(done[0]) and bool(u.outcome_success[0])):
            diagnostic = {
                "pad": target_index,
                "done": bool(done[0]),
                "feed": feed(u),
                "true_position": u._positions()[0].cpu().tolist(),
                "at_waypoint": u.at_waypoint[0].cpu().tolist(),
                "landing": u.landing[0].cpu().tolist(),
                "dwell": u.land_dwell[0].cpu().tolist(),
                "outcome_success": bool(u.outcome_success[0]) if done[0] else None,
                "outcome_crash": bool(u.outcome_crash[0]) if done[0] else None,
            }
            path = Path(f"outputs/pipeline_pad_{target_index}_failure.json")
            path.write_text(json.dumps(diagnostic, indent=2) + "\n")
            print(f"[FAIL] Controlled landing diagnostic: {path}")
        assert bool(done[0]) and bool(u.outcome_success[0]), f"Controlled landing failed for pad {target_index}"
    print(f"[ok] Gated, on-target touchdown for {cfg.num_drones} drone(s) across all {successes} pad assignments")
    env.close()
    print("POLICY PIPELINE PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Print before shutdown: Kit can stall in close(), hiding the actual assertion.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.app.close()

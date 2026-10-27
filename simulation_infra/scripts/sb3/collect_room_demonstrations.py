"""Collect successful full-room policy trajectories and mirrored imitation examples."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=2027)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import action_space_kit.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from action_space_kit.rl.policy_export import norm_path_for  # noqa: E402
from action_space_kit.rl.run_config import restore_run_config  # noqa: E402
from action_space_kit.rl.sb3_shared import Sb3CentralVecEnv  # noqa: E402
from action_space_kit.rl.symmetry import mirror_room_action, mirror_room_observation  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    task = "AS-Tello-Room-Single-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "sb3_cfg_entry_point")
    restore_run_config(args.checkpoint, cfg, agent_cfg)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.spawn_short_fraction = cfg.spawn_hover_fraction = cfg.spawn_near_target_radius = 0.
    cfg.spawn_box_center, cfg.spawn_box_size, cfg.spawn_range = (6.5, 0.), 2., (6., 7.)
    cfg.takeoff_at_reset, cfg.episode_length_s, cfg.fixed_spot_index = True, 75., -1
    base = Sb3CentralVecEnv(gym.make(task, cfg=cfg))
    u = base.unwrapped_env
    normalizer = VecNormalize.load(str(norm_path_for(args.checkpoint)), base)
    normalizer.training = normalizer.norm_reward = False
    frames = agent_cfg["frame_stack"]
    env = VecFrameStack(normalizer, n_stack=frames)
    model = PPO.load(args.checkpoint, env)
    model.set_random_seed(args.seed)
    obs = env.reset()
    u.episode_length_buf[:] = 0
    raw = np.zeros((args.num_envs, frames, 25), dtype=np.float32)
    valid = np.zeros((args.num_envs, frames), dtype=bool)
    raw[:, -1], valid[:, -1] = normalizer.get_original_obs(), True
    mean = normalizer.obs_rms.mean
    std = np.sqrt(normalizer.obs_rms.var + normalizer.epsilon)

    def normalize(history):
        result = np.clip((history - mean) / std, -normalizer.clip_obs, normalizer.clip_obs).astype(np.float32)
        result[~valid] = 0
        return result.reshape(args.num_envs, -1)

    trajectories = [[] for _ in range(args.num_envs)]
    records = {}
    kept = []
    max_normalization_error = 0.
    for _ in range(math.ceil(cfg.episode_length_s / u.step_dt) + 2):
        error = float(np.max(np.abs(normalize(raw) - obs)))
        max_normalization_error = max(max_normalization_error, error)
        if error > 1e-5:
            raise AssertionError(f"Raw-history reconstruction disagrees with VecFrameStack: {error}")
        reflected = normalize(mirror_room_observation(raw.reshape(args.num_envs, -1)).reshape(raw.shape))
        with torch.inference_mode():
            means = model.policy.get_distribution(torch.as_tensor(obs, device=model.device)).distribution.mean.cpu().numpy()
        reflected_means = mirror_room_action(means)
        for index in range(args.num_envs):
            if index not in records:
                trajectories[index].append((obs[index].copy(), reflected[index].copy(),
                                             means[index].copy(), reflected_means[index].copy()))
        obs, _, dones, _ = env.step(np.clip(means, -1., 1.))
        for record in base.episode_buffer:
            index = record["env_id"]
            if index in records:
                continue
            records[index] = record
            if record["metrics"]["Episode/success_rate"] == 1:
                kept.append(index)
            else:
                trajectories[index].clear()
        base.episode_buffer.clear()
        base.log_buffer.clear()
        if len(records) == args.num_envs:
            break
        raw, valid = np.roll(raw, -1, axis=1), np.roll(valid, -1, axis=1)
        raw[dones], valid[dones] = 0., False
        raw[:, -1], valid[:, -1] = normalizer.get_original_obs(), True
    if len(records) != args.num_envs or not kept:
        raise RuntimeError(f"Incomplete/empty collection: {len(records)} episodes, {len(kept)} successes")
    xs, ys, groups, mirrored, targets = [], [], [], [], []
    for index in kept:
        steps = trajectories[index]
        pad = records[index]["drones"][0]["target_index"]
        for reflection in (0, 1):
            xs.extend(row[reflection] for row in steps)
            ys.extend(row[reflection + 2] for row in steps)
            groups.extend([index] * len(steps))
            mirrored.extend([reflection] * len(steps))
            targets.extend([cfg.spot_count - 1 - pad if reflection else pad] * len(steps))
    np.savez_compressed(out / "demonstrations.npz", observations=np.asarray(xs), actions=np.asarray(ys),
                        groups=np.asarray(groups), mirrored=np.asarray(mirrored), targets=np.asarray(targets))
    metadata = {"teacher": str(Path(args.checkpoint).resolve()), "seed": args.seed,
                "teacher_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "normalizer_sha256": hashlib.sha256(norm_path_for(args.checkpoint).read_bytes()).hexdigest(),
                "episodes": len(records), "successful_episodes": len(kept), "samples": len(xs),
                "max_normalization_error": max_normalization_error,
                "mirroring": "training prior only; validation uses the actual asymmetric room",
                "records": list(records.values())}
    (out / "collection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"COLLECTION COMPLETE: {len(kept)}/{len(records)} successful trajectories; {len(xs)} examples", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.app.close()

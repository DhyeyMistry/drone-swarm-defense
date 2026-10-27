"""Warm-start a four-drone joint policy from the selected single-drone checkpoint.

Runs without Isaac Sim and sends no hardware commands. This is an initialization,
not a trained coordination policy: deterministic per-drone outputs initially match
the parent, then ordinary centralized PPO can learn the cross-drone connections.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

REPO = Path(__file__).resolve().parents[3]
RL = REPO / "simulation_infra/source/action_space_kit/action_space_kit/rl"
# Import the pure CPU helper without importing the Isaac extension package.
sys.path.insert(0, str(RL))
from team_transfer import expand_normalizer, transfer_policy  # noqa: E402


class SpaceOnlyEnv(gym.Env):
    """Declare checkpoint spaces without loading a simulator."""

    def __init__(self, observation_dim, action_dim):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (action_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        raise RuntimeError("This environment only declares spaces; it cannot train or evaluate flights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoint, out = Path(args.checkpoint).resolve(), Path(args.out).resolve()
    parent = checkpoint.parent
    run = json.loads((parent / "params/run.json").read_text())
    if run["task"] != "AS-Tello-Room-Single-v0" or run["env"]["num_drones"] != 1:
        raise ValueError("Expected a single-drone room checkpoint")
    if not run["env"]["start_on_floor"] or not run["env"]["land_gate_enabled"]:
        raise ValueError("Expected the floor-start, gated-landing checkpoint")
    if run["env"]["spawn_short_fraction"] or run["env"]["spawn_hover_fraction"]:
        raise ValueError("Select a full-mission checkpoint before expanding the team")
    if out.exists():
        raise FileExistsError(out)
    match = re.fullmatch(r"model_(\d+)_steps", checkpoint.stem)
    norm_name = f"model_vecnormalize_{match[1]}_steps.pkl" if match else "model_vecnormalize.pkl"
    norm_path = parent / norm_name
    # As with PPO.load, this pickle must be from the trusted local training run.
    with norm_path.open("rb") as handle:
        old_norm = pickle.load(handle)
    source = PPO.load(str(checkpoint), device="cpu")
    source.policy.set_training_mode(False)
    frames, drones, width = run["agent"]["frame_stack"], 4, 25
    if source.action_space.shape != (5,):
        raise ValueError("Expected the five-action room policy")
    base = DummyVecEnv([lambda: SpaceOnlyEnv(width * drones, 5 * drones)])
    norm, value_scale = expand_normalizer(old_norm, base, drones)
    env = VecFrameStack(norm, n_stack=frames)
    kwargs = copy.deepcopy(source.policy_kwargs)
    arch = kwargs["net_arch"]
    if isinstance(arch, dict):
        kwargs["net_arch"] = {key: [n * drones for n in sizes] for key, sizes in arch.items()}
    else:
        kwargs["net_arch"] = [n * drones for n in arch]
    kwargs["ortho_init"] = False  # all network parameters are replaced below
    target = PPO("MlpPolicy", env, policy_kwargs=kwargs, learning_rate=source.learning_rate,
                 n_steps=source.n_steps, batch_size=source.n_steps, n_epochs=source.n_epochs,
                 gamma=source.gamma, gae_lambda=source.gae_lambda, ent_coef=source.ent_coef,
                 vf_coef=source.vf_coef, max_grad_norm=source.max_grad_norm,
                 target_kl=source.target_kl, seed=source.seed, device="cpu")
    transfer_policy(source.policy, target.policy, observation_dim=width, frame_stack=frames,
                    num_drones=drones, value_scale=value_scale)
    target.policy.set_training_mode(False)

    # Verify complete frame ordering, normalization and RAW actor means (not just
    # clipped RC actions, which could hide a bad transfer). Include reset padding.
    rng = np.random.default_rng(9307)
    raw = rng.normal(size=(128, frames, drones, width)).astype(np.float32)
    single = old_norm.normalize_obs(raw.reshape(-1, width)).reshape(raw.shape)
    joint = norm.normalize_obs(raw.reshape(-1, drones * width)).reshape(raw.shape)
    np.testing.assert_array_equal(single, joint)
    single[:8, :-1] = joint[:8, :-1] = 0.
    own = torch.from_numpy(single.transpose(0, 2, 1, 3).reshape(-1, frames * width))
    together = torch.from_numpy(joint.reshape(-1, frames * drones * width))
    with torch.no_grad():
        expected = source.policy.get_distribution(own).distribution.mean.reshape(128, -1)
        actual = target.policy.get_distribution(together).distribution.mean
        expected_value = source.policy.predict_values(own).reshape(128, drones).sum(1) * value_scale
        actual_value = target.policy.predict_values(together).flatten()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_value, expected_value, rtol=1e-5, atol=1e-5)

    out.mkdir(parents=True)
    params = out / "params"
    params.mkdir()
    shutil.copytree(parent / "assets", out / "assets")
    shutil.copy2(parent / "params/run.json", params / "parent_run.json")
    if (parent / "params/source.tar.gz").exists():
        shutil.copy2(parent / "params/source.tar.gz", params / "parent_source.tar.gz")
    target.save(out / "model.zip")
    norm.save(out / "model_vecnormalize.pkl")
    metadata = {
        "method": "independent actor initialization of a dense joint PPO policy",
        "parent_checkpoint": str(checkpoint),
        "parent_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parent_normalizer_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "parent_training_transitions": source.num_timesteps,
        "new_training_transitions": target.num_timesteps,
        "num_drones": drones, "frame_stack": frames, "observation_dim_per_drone": width,
        "observation_order": "frame, drone, feature",
        "critic_initial_value": "mean of parent values" if old_norm.norm_reward else "sum of parent values",
        "max_actor_mean_error": float((actual - expected).abs().max()),
        "max_critic_error": float((actual_value - expected_value).abs().max()),
        "qualification": "untrained initialization; requires shared-room training and evaluation",
    }
    run["task"] = "AS-Tello-Room-Team-v0"
    run["env"]["num_drones"] = drones
    run["env"]["possible_agents"] = [f"drone_{i}" for i in range(drones)]
    run["env"]["fixed_spot_index"] = -1
    run["agent"]["central"] = True
    run["agent"]["freeze_normalization"] = True
    run["agent"]["policy_kwargs"]["net_arch"] = kwargs["net_arch"]
    run["agent"]["policy_kwargs"]["ortho_init"] = False
    run["team_transfer"] = metadata
    run["git_commit"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    run["argv"] = sys.orig_argv
    paths = [Path(__file__), RL / "team_transfer.py",
             RL.parent / "tasks/direct/tello_room/tello_room_team_env_cfg.py",
             RL.parent / "tasks/direct/tello_room/__init__.py"]
    run["source_sha256"] = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in paths}
    for path in paths:
        shutil.copy2(path, params / (path.stem + "_source.py"))
    (params / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    (params / "transfer.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (params / "env.yaml").write_text(yaml.safe_dump(run["env"], sort_keys=False))
    (params / "agent.yaml").write_text(yaml.safe_dump(run["agent"], sort_keys=False))
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"TEAM TRANSFER PASSED: {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()

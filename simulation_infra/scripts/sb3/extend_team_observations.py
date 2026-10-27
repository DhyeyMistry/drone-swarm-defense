"""Add estimated peer-motion inputs without changing an existing team policy's outputs."""

import argparse
import copy
import hashlib
import json
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from expand_room_policy import REPO, RL, SpaceOnlyEnv

sys.path.insert(0, str(RL))
from team_transfer import extend_observation_normalizer, extend_policy_observations


def snapshot_transfer_source(params, run):
    """Keep current transfer code separate from the preserved parent's source snapshot."""
    files = subprocess.check_output(
        ["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines()
    source = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in files
              if (REPO / name).is_file() and Path(name).suffix in (".py", ".yaml", ".toml")}
    run["source_sha256"] = source
    with tarfile.open(params / "source.tar.gz", "w:gz") as archive:
        for name in sorted(source):
            archive.add(REPO / name, arcname=name)
    (params / "working.diff").write_text(subprocess.check_output(["git", "-C", str(REPO), "diff", "HEAD"], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shared", action="store_true",
                        help="Initialize a shared per-drone peer policy from the original single-drone model")
    args = parser.parse_args()
    checkpoint, out = Path(args.checkpoint).resolve(), Path(args.out).resolve()
    run = json.loads((checkpoint.parent / "params/run.json").read_text())
    expected_task = "AS-Tello-Room-Single-v0" if args.shared else "AS-Tello-Room-Team-v0"
    drones = 1 if args.shared else 4  # observation/action blocks in each policy row
    if run["task"] != expected_task or run["env"]["num_drones"] != drones:
        raise ValueError(f"Expected {expected_task} with {drones} drone(s) per policy row")
    if not run["env"]["start_on_floor"] or run["env"]["spawn_short_fraction"] or run["env"]["spawn_hover_fraction"]:
        raise ValueError("Expected a full-room floor-start checkpoint")
    if out.exists():
        raise FileExistsError(out)
    source = PPO.load(checkpoint, device="cpu")
    frames, old, new = run["agent"]["frame_stack"], 25, 48
    match = re.fullmatch(r"model_(\d+)_steps", checkpoint.stem)
    norm_path = checkpoint.with_name(f"model_vecnormalize_{match[1]}_steps.pkl" if match else "model_vecnormalize.pkl")
    with norm_path.open("rb") as handle:
        source_norm = pickle.load(handle)
    base = DummyVecEnv([lambda: SpaceOnlyEnv(new * drones, 5 * drones)])
    norm = extend_observation_normalizer(source_norm, base, old_width=old, new_width=new, num_drones=drones)
    env = VecFrameStack(norm, n_stack=frames)
    target = PPO("MlpPolicy", env, policy_kwargs=copy.deepcopy(source.policy_kwargs),
                 learning_rate=source.learning_rate, n_steps=source.n_steps, batch_size=source.n_steps,
                 n_epochs=source.n_epochs, gamma=source.gamma, gae_lambda=source.gae_lambda,
                 ent_coef=source.ent_coef, vf_coef=source.vf_coef, max_grad_norm=source.max_grad_norm,
                 target_kl=source.target_kl, seed=source.seed, device="cpu")
    extend_policy_observations(source.policy, target.policy, old_width=old, new_width=new,
                               frame_stack=frames, num_drones=drones)
    target.num_timesteps = 0 if args.shared else source.num_timesteps
    raw = np.random.default_rng(737).normal(size=(128, frames, drones, new)).astype(np.float32)
    original = source_norm.normalize_obs(raw[..., :old].reshape(-1, old * drones)).reshape(128, frames, drones, old)
    expanded = norm.normalize_obs(raw.reshape(-1, new * drones)).reshape(raw.shape)
    np.testing.assert_array_equal(original, expanded[..., :old])
    original[:8, :-1] = expanded[:8, :-1] = 0.
    x, y = torch.from_numpy(original.reshape(128, -1)), torch.from_numpy(expanded.reshape(128, -1))
    with torch.no_grad():
        a = source.policy.get_distribution(x).distribution.mean
        b = target.policy.get_distribution(y).distribution.mean
        va, vb = source.policy.predict_values(x), target.policy.predict_values(y)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(va, vb, atol=1e-5, rtol=1e-5)
    out.mkdir(parents=True)
    shutil.copytree(checkpoint.parent / "assets", out / "assets")
    shutil.copytree(checkpoint.parent / "params", out / "parent_params")
    params = out / "params"
    params.mkdir()
    metadata = {"parent_checkpoint": str(checkpoint),
                "parent_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "parent_normalizer_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                "parent_training_transitions": source.num_timesteps,
                "new_training_transitions": target.num_timesteps,
                "control_mode": "shared per-drone" if args.shared else "central joint",
                "max_actor_error": float((a-b).abs().max()), "max_critic_error": float((va-vb).abs().max()),
                "old_features_per_drone": old, "features_per_drone": new,
                "qualification": "Output-preserving initialization; peer inputs are not yet trained"}
    run.update(task="AS-Tello-Room-Peer-v0", peer_transfer=metadata, argv=sys.orig_argv)
    run["env"].update(num_drones=4, possible_agents=[f"drone_{i}" for i in range(4)], fixed_spot_index=-1,
                      peer_observations=True, peer_sort_by_distance=args.shared, peer_order_version=2, collision_substeps=True,
                      rew_predictive_separation=-100., separation_lookahead_s=1.5, separation_prediction_margin=.5)
    run["agent"].update(central=not args.shared, freeze_normalization=True)
    snapshot_transfer_source(params, run)
    (params / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (params / "transfer.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for key in ("env", "agent"):
        (params / f"{key}.yaml").write_text(yaml.safe_dump(run[key], sort_keys=False))
    target.save(out / "model.zip")
    norm.save(out / "model_vecnormalize.pkl")
    print(json.dumps(metadata, indent=2), flush=True)
    print("PEER TRANSFER PASSED", flush=True)
    env.close()


if __name__ == "__main__":
    main()

"""Create a residual candidate with stable peer ordering and/or bounded action means.

This changes deterministic commands. Score the migration before training or selecting it.
The base pilot, correction weights, critic and normalization are retained exactly.
"""

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from expand_room_policy import REPO, SpaceOnlyEnv
from extend_team_observations import snapshot_transfer_source

sys.path.insert(0, str(REPO / "drone_deployment_infra"))
from core import residual_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bound-mean", action="store_true")
    parser.add_argument("--stable-peer-order", action="store_true")
    args = parser.parse_args()
    if not (args.bound_mean or args.stable_peer_order):
        raise ValueError("Select a candidate change explicitly")
    checkpoint, out = Path(args.checkpoint).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    match = re.fullmatch(r"model_(\d+)_steps", checkpoint.stem)
    normalizer = checkpoint.with_name(f"model_vecnormalize_{match[1]}_steps.pkl" if match else "model_vecnormalize.pkl")
    if not normalizer.is_file():
        raise FileNotFoundError(normalizer)
    run = json.loads((checkpoint.parent / "params/run.json").read_text())
    if run["task"] != "AS-Tello-Room-Peer-v0" or run["agent"]["central"] or run["env"]["num_drones"] != 4:
        raise ValueError("Expected a shared four-drone peer policy")
    torch.set_num_threads(1)
    source = PPO.load(checkpoint, device="cpu")
    if getattr(source.policy.action_net, "bound_mean", False):
        raise ValueError("Use an unbounded source so the migration comparison is explicit")
    kwargs = {**copy.deepcopy(source.policy_kwargs), "bound_mean": args.bound_mean}
    env = DummyVecEnv([lambda: SpaceOnlyEnv(source.observation_space.shape[0], source.action_space.shape[0])])
    target = PPO(residual_policy.ResidualActorCriticPolicy, env, policy_kwargs=kwargs,
                 learning_rate=source.learning_rate, n_steps=source.n_steps, batch_size=source.n_steps,
                 n_epochs=source.n_epochs, gamma=source.gamma, gae_lambda=source.gae_lambda,
                 ent_coef=source.ent_coef, vf_coef=source.vf_coef, max_grad_norm=source.max_grad_norm,
                 target_kl=source.target_kl, seed=source.seed, device="cpu")
    residual_policy.copy_residual_state(source.policy, target.policy)
    for key, value in source.policy.state_dict().items():
        torch.testing.assert_close(target.policy.state_dict()[key], value, rtol=0, atol=0)
    target.num_timesteps = source.num_timesteps
    x = torch.from_numpy(np.random.default_rng(498).normal(size=(512, *source.observation_space.shape)).astype(np.float32))
    with torch.no_grad():
        old_mean = source.policy.get_distribution(x).distribution.mean
        new_mean = target.policy.get_distribution(x).distribution.mean
        va, vb = source.policy.predict_values(x), target.policy.predict_values(x)
    torch.testing.assert_close(new_mean, old_mean.tanh() if args.bound_mean else old_mean, rtol=0, atol=0)
    torch.testing.assert_close(va, vb, rtol=0, atol=0)
    if args.bound_mean:
        assert new_mean.abs().max() <= 1.
    out.mkdir(parents=True)
    shutil.copytree(checkpoint.parent / "assets", out / "assets")
    shutil.copytree(checkpoint.parent / "params", out / "parent_params")
    shutil.copy2(normalizer, out / "model_vecnormalize.pkl")
    params = out / "params"
    params.mkdir()
    metadata = {"parent_checkpoint": str(checkpoint), "parent_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "parent_transitions": source.num_timesteps, "all_parameter_values_preserved": True,
                "max_critic_error": float((va-vb).abs().max()), "base_frozen": True, "bound_mean": args.bound_mean,
                "peer_order_version": 2 if args.stable_peer_order else run["env"].get("peer_order_version", 1),
                "max_probe_command_change": float((old_mean.clamp(-1., 1.)-new_mean.clamp(-1., 1.)).abs().max()),
                "qualification": "Untrained contract migration; requires complete evaluation"}
    run.update(residual_candidate_transfer=metadata, argv=sys.orig_argv)
    run["agent"]["policy_kwargs"]["bound_mean"] = args.bound_mean
    run["env"]["peer_order_version"] = metadata["peer_order_version"]
    snapshot_transfer_source(params, run)
    (params / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (params / "transfer.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for key in ("env", "agent"):
        (params / f"{key}.yaml").write_text(yaml.safe_dump(run[key], sort_keys=False))
    cloudpickle.register_pickle_by_value(residual_policy)
    target.save(out / "model.zip")
    print(json.dumps(metadata, indent=2), flush=True)
    print("RESIDUAL CANDIDATE TRANSFER PASSED", flush=True)
    env.close()


if __name__ == "__main__":
    main()

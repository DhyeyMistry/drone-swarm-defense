"""Keep an existing shared learned pilot and add a zero-initialized learned correction."""

import argparse
import copy
import hashlib
import json
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
    args = parser.parse_args()
    checkpoint, out = Path(args.checkpoint).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if checkpoint.name != "model.zip":
        raise ValueError("Use a final/init model.zip with its matching model_vecnormalize.pkl")
    run = json.loads((checkpoint.parent / "params/run.json").read_text())
    if run["task"] != "AS-Tello-Room-Peer-v0" or run["agent"]["central"] or run["env"]["num_drones"] != 4:
        raise ValueError("Expected a four-drone shared peer policy")
    torch.set_num_threads(1)
    source = PPO.load(checkpoint, device="cpu")
    kwargs = {**copy.deepcopy(source.policy_kwargs), "residual_width": 128}
    env = DummyVecEnv([lambda: SpaceOnlyEnv(source.observation_space.shape[0], source.action_space.shape[0])])
    target = PPO(residual_policy.ResidualActorCriticPolicy, env, policy_kwargs=kwargs,
                 learning_rate=source.learning_rate, n_steps=source.n_steps, batch_size=source.n_steps,
                 n_epochs=source.n_epochs, gamma=source.gamma, gae_lambda=source.gae_lambda,
                 ent_coef=source.ent_coef, vf_coef=source.vf_coef, max_grad_norm=source.max_grad_norm,
                 target_kl=source.target_kl, seed=source.seed, device="cpu")
    residual_policy.copy_learned_pilot(source.policy, target.policy)
    x = torch.from_numpy(np.random.default_rng(847).normal(size=(512, *source.observation_space.shape)).astype(np.float32))
    with torch.no_grad():
        original = source.policy.get_distribution(x).distribution.mean.clamp(-1., 1.)
        initial = target.policy.get_distribution(x).distribution.mean
        va, vb = source.policy.predict_values(x), target.policy.predict_values(x)
    torch.testing.assert_close(original, initial, rtol=0, atol=0)
    torch.testing.assert_close(va, vb, rtol=0, atol=0)
    out.mkdir(parents=True)
    shutil.copytree(checkpoint.parent / "assets", out / "assets")
    shutil.copytree(checkpoint.parent / "params", out / "parent_params")
    shutil.copy2(checkpoint.parent / "model_vecnormalize.pkl", out / "model_vecnormalize.pkl")
    params = out / "params"
    params.mkdir()
    metadata = {"parent_checkpoint": str(checkpoint), "parent_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "parent_transitions": source.num_timesteps, "max_command_error": float((original-initial).abs().max()),
                "max_critic_error": float((va-vb).abs().max()), "base_frozen": True, "residual_width": 128,
                "qualification": "Deterministic command-preserving initialization, not trained collision avoidance"}
    run.update(residual_transfer=metadata, argv=sys.orig_argv)
    run["agent"]["policy_kwargs"]["residual_width"] = 128
    snapshot_transfer_source(params, run)
    (params / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (params / "transfer.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for key in ("env", "agent"):
        (params / f"{key}.yaml").write_text(yaml.safe_dump(run[key], sort_keys=False))
    # Embed the small pure-Python policy classes so clean CPU inference/export does
    # not require importing Isaac Lab or this source directory to load the artifact.
    cloudpickle.register_pickle_by_value(residual_policy)
    target.save(out / "model.zip")
    print(json.dumps(metadata, indent=2), flush=True)
    print("RESIDUAL TRANSFER PASSED", flush=True)
    env.close()


if __name__ == "__main__":
    main()

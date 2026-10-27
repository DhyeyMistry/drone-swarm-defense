"""Play (and score) a Stable-Baselines3 checkpoint of an Action Space task.

Adapted from Isaac Lab's scripts/reinforcement_learning/sb3/play.py, using the same
shared-policy wrapper as training. Prints episode statistics as episodes finish.

    python scripts/sb3/play.py --task AS-Tello-Approach-v0 --num_envs 16 --checkpoint logs/sb3/AS-Tello-Approach-v0/<run>/model.zip
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (default: latest run).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--episodes", type=int, default=0, help="Stop after this many finished episodes (0 = run forever).")
parser.add_argument("--config-source", choices=("checkpoint", "current"), default="checkpoint")
parser.add_argument("--report", type=str, default=None, help="Write exact episode cohort and summary to JSON.")
parser.add_argument("--steps", type=int, default=0, help="Optional diagnostic step cap; incomplete cohorts remain failures.")
parser.add_argument("--action-diagnostics", action="store_true",
                    help="Measure raw Gaussian action saturation during room flights without changing commands.")
parser.add_argument("--pilot", choices=("policy", "hold"), default="policy",
                    help="hold: room gate baseline with zero RC and continuous land request; not a policy score.")
parser.add_argument("--no_dr", action="store_true", default=False, help="Fly with nominal dynamics (no randomization).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import hashlib
import json
import os
import re
import time

import action_space_kit.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from action_space_kit.rl.evaluation import EpisodeCohort
from action_space_kit.rl.run_config import restore_run_config
from action_space_kit.rl.sb3_shared import Sb3CentralVecEnv, Sb3SharedPolicyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path


def vecnormalize_path(checkpoint_path: str) -> Path:
    """model.zip -> model_vecnormalize.pkl; model_N_steps.zip -> model_vecnormalize_N_steps.pkl."""
    p = Path(checkpoint_path)
    m = re.match(r"model_(\d+)_steps$", p.stem)
    name = f"model_vecnormalize_{m.group(1)}_steps.pkl" if m else "model_vecnormalize.pkl"
    return p.with_name(name)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with a Stable-Baselines3 agent."""
    if args_cli.checkpoint is None:
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", "model.zip", sort_alpha=False)
    else:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
    log_dir = os.path.dirname(checkpoint_path)
    if args_cli.config_source == "checkpoint":
        restore_run_config(checkpoint_path, env_cfg, agent_cfg, hydra_args)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "debug_vis"):
        env_cfg.debug_vis = not args_cli.headless
    if args_cli.no_dr and hasattr(env_cfg, "dynamics"):
        env_cfg.dynamics.domain_randomization = False
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    shared = isinstance(env.unwrapped, DirectMARLEnv)
    wrapper = Sb3CentralVecEnv if agent_cfg.get("central", False) else Sb3SharedPolicyVecEnv
    env = wrapper(env) if shared else Sb3VecEnvWrapper(env)
    base_env = env

    norm_path = vecnormalize_path(checkpoint_path)
    if norm_path.exists():
        print(f"Loading saved normalization: {norm_path}")
        env = VecNormalize.load(str(norm_path), env)
        env.training = False
        env.norm_reward = False
    elif agent_cfg.get("normalize_input", False):
        raise FileNotFoundError(f"Checkpoint requires normalization statistics: {norm_path}")
    frame_stack = agent_cfg.get("frame_stack", 1)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)

    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)
    # PPO.load restores the training seed; seed again AFTER loading and BEFORE first reset.
    agent.set_random_seed(env_cfg.seed)
    print(f"[eval] effective_seed={env_cfg.seed} pilot={args_cli.pilot}")
    if args_cli.pilot == "hold" and not args_cli.task.startswith("AS-Tello-Room"):
        raise ValueError("The hold/land baseline is only defined for room tasks")
    dt = base_env.unwrapped_env.step_dt if shared else base_env.unwrapped.step_dt

    slots = base_env.num_isaac_envs if shared else env.num_envs
    cohort = EpisodeCohort(slots, args_cli.episodes or 10**12,
                           retain_records=bool(args_cli.episodes or args_cli.report))
    if args_cli.action_diagnostics and (not shared or not args_cli.task.startswith("AS-Tello-Room")):
        raise ValueError("Action diagnostics require a room task")
    action_samples = {"all_flying": [], "first_8_seconds": [], "neighbor_within_0_8m": []}
    obs = env.reset()
    if shared:
        # a full reset staggers episode clocks for training; scored episodes must get the whole
        # episode, or the first one in every env is cut short and counted as a failure
        base_env.unwrapped_env.episode_length_buf[:] = 0
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if args_cli.pilot == "policy":
                actions, _ = agent.predict(obs, deterministic=True)
                if args_cli.action_diagnostics:
                    tensor_obs, _ = agent.policy.obs_to_tensor(obs)
                    means = agent.policy.get_distribution(tensor_obs).distribution.mean.cpu().numpy()
                    np.testing.assert_allclose(actions, np.clip(means, -1., 1.), atol=1e-6)
                    u = base_env.unwrapped_env
                    means = means.reshape(slots, u.cfg.num_drones, 5)
                    active = np.asarray(cohort.counts) < np.asarray(cohort.quotas)
                    flying = (~(u.taking_off | u.landing | u.landed)).cpu().numpy() & active[:, None]
                    early = (u.episode_length_buf * dt < 8.).cpu().numpy()[:, None]
                    near = (u._pairwise_distance(u._positions()).amin(-1) < .8).cpu().numpy()
                    for name, mask in (("all_flying", flying), ("first_8_seconds", flying & early),
                                       ("neighbor_within_0_8m", flying & near)):
                        action_samples[name].append(means[mask, :4])
            else:
                actions = np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32)
                actions[:, 4::5] = 1.0
            obs, _, dones, infos = env.step(actions)
        changed = False
        if shared:
            for record in base_env.episode_buffer:
                changed |= cohort.add(record)
            base_env.episode_buffer.clear()
            base_env.log_buffer.clear()
        else:
            for i, done in enumerate(dones):
                if done:
                    changed |= cohort.add({"env_id": i, "episode_id": timestep + 1,
                                           "metrics": infos[i].get("episode", {})})
        if changed:
            summary = cohort.summary()
            values = ", ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in summary["metrics"].items())
            print(f"[{summary['episodes']} episodes] " + values)
        if args_cli.episodes and (timestep + 1) % 100 == 0:
            pending = np.flatnonzero(np.asarray(cohort.counts) < np.asarray(cohort.quotas))
            print(f"[eval-progress] steps={timestep + 1} simulated_s={(timestep + 1) * dt:.3f} "
                  f"completed={sum(cohort.counts)}/{args_cli.episodes} "
                  f"pending_slots={len(pending)} first_pending={pending[:8].tolist()}", flush=True)
        if args_cli.episodes and cohort.complete:
            break
        if args_cli.steps and timestep + 1 >= args_cli.steps:
            break
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    summary = cohort.summary()
    diagnostics = None
    if args_cli.action_diagnostics:
        diagnostics = {"channels": ["right", "forward", "up", "yaw"],
                       "scope": "Raw policy means before command clipping; counts are drone-steps, not trials",
                       "buckets": {}}
        for name, samples in action_samples.items():
            values = np.concatenate(samples, axis=0) if samples else np.empty((0, 4))
            diagnostics["buckets"][name] = {"drone_steps": len(values)}
            if len(values):
                diagnostics["buckets"][name].update(
                    quantiles=np.quantile(values, [0., .1, .5, .9, 1.], axis=0).tolist(),
                    abs_mean_gt_1=(np.abs(values) > 1.).mean(axis=0).tolist(),
                    abs_mean_gt_1_3=(np.abs(values) > 1.3).mean(axis=0).tolist())
        print("[action-diagnostics] " + json.dumps(diagnostics), flush=True)
    if args_cli.report:
        report = Path(args_cli.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_hash = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
        norm_hash = hashlib.sha256(norm_path.read_bytes()).hexdigest() if norm_path.exists() else None
        report.write_text(json.dumps({"checkpoint": checkpoint_path,
                                     "checkpoint_sha256": checkpoint_hash,
                                     "normalizer_sha256": norm_hash,
                                     "seed": env_cfg.seed,
                                     "pilot": args_cli.pilot,
                                     "config_source": args_cli.config_source, "overrides": hydra_args,
                                     "diagnostic_step_cap": args_cli.steps, "action_diagnostics": diagnostics,
                                     "summary": summary, "records": cohort.records}, indent=2, allow_nan=False) + "\n")
        print(f"[eval] report={report} complete={cohort.complete}")
    env.close()
    if args_cli.episodes and not cohort.complete:
        raise RuntimeError(f"Incomplete evaluation: {summary['episodes']}/{args_cli.episodes} episodes")


if __name__ == "__main__":
    main()
    simulation_app.close()

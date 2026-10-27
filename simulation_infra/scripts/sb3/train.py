"""Train an Action Space task with Stable-Baselines3 PPO.

Adapted from Isaac Lab's scripts/reinforcement_learning/sb3/train.py. The difference: a
multi-agent task is wrapped with ``Sb3SharedPolicyVecEnv``, so all drones train one shared
policy that each real drone can run on its own, instead of Isaac Lab's centralized
``multi_agent_to_single_agent`` conversion.

    python scripts/sb3/train.py --task AS-Tello-Approach-v0 --headless --num_envs 512 --max_iterations 500
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import copy
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--action-std", type=float, default=None,
                    help="Explicitly reset Gaussian exploration std on warm start; deterministic actions are unchanged.")
parser.add_argument("--value-scale", type=float, default=None,
                    help="On warm start, scale critic predictions to new reward units; actor is unchanged.")
parser.add_argument("--critic-warmup-iterations", type=int, default=0,
                    help="Fit the critic for this many initial rollouts before updating a resumed actor.")
parser.add_argument("--config-source", choices=("checkpoint", "current"), default="checkpoint",
                    help="On resume, restore the saved configuration or explicitly use the current task.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (rollout + update cycles).")
parser.add_argument("--save_interval", type=int, default=50, help="Save a checkpoint every n iterations.")
parser.add_argument("--curriculum", choices=("landing", "approach", "full"), default=None,
                    help="Single-drone stage, preserving the same observation/action shapes.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if (args_cli.action_std is not None or args_cli.value_scale is not None) and not args_cli.checkpoint:
    parser.error("Warm-start adjustments require --checkpoint")
if args_cli.critic_warmup_iterations < 0 or (args_cli.critic_warmup_iterations and not args_cli.checkpoint):
    parser.error("Critic warmup requires --checkpoint and a nonnegative iteration count")
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import os
import random
import time
from datetime import datetime

import action_space_kit.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
from action_space_kit.rl.finetune import prepare_finetune
from action_space_kit.rl.policy_export import norm_path_for
from action_space_kit.rl.run_config import restore_run_config, snapshot_run
from action_space_kit.rl.sb3_shared import LogExtrasCallback, Sb3CentralVecEnv, Sb3SharedPolicyVecEnv
from action_space_kit.rl.team_transfer import CriticWarmupCallback
from action_space_kit.rl.training_control import GracefulStopCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

stop_callback = GracefulStopCallback()
signal.signal(signal.SIGINT, stop_callback.request_stop)
signal.signal(signal.SIGTERM, stop_callback.request_stop)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with a Stable-Baselines3 agent."""
    if args_cli.checkpoint and args_cli.config_source == "checkpoint":
        restore_run_config(args_cli.checkpoint, env_cfg, agent_cfg, hydra_args)
    if args_cli.curriculum:
        if getattr(env_cfg, "num_drones", None) != 1:
            raise ValueError("Validate the single-drone curriculum before expanding the team")
        stages = {"landing": ((1.25, 0.), .8, (.8, 1.9), False),
                  "approach": ((2.5, 0.), 1.5, (1.8, 3.4), True),
                  "full": ((6.5, 0.), 2., (6., 7.), True)}
        fields = dict(zip(("spawn_box_center", "spawn_box_size", "spawn_range", "takeoff_at_reset"),
                          stages[args_cli.curriculum]))
        fields["spawn_height"] = ((env_cfg.waypoint_height - .05, env_cfg.waypoint_height + .05)
                                  if args_cli.curriculum == "landing" else (0.9, 1.4))
        fields["spawn_near_target_radius"] = 0.10 if args_cli.curriculum == "landing" else 0.0
        explicit = {arg.split("=", 1)[0].lstrip("+") for arg in hydra_args}
        for name, value in fields.items():
            if "env." + name not in explicit:
                setattr(env_cfg, name, value)
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # shared policy: every agent of a MARL env is one SB3 row, so rollouts are that many times
    # larger; central policy (agent cfg ``central: true``): one row per env, all agents at once
    central = bool(agent_cfg.get("central", False))
    num_agents = len(env_cfg.possible_agents) if isinstance(env_cfg, DirectMARLEnvCfg) and not central else 1
    num_rows = env_cfg.scene.num_envs * num_agents
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * num_rows

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = os.path.join(log_root_path, run_info)
    Path(log_dir).mkdir(parents=True, exist_ok=False)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    (Path(log_dir) / "command.txt").write_text(" ".join(sys.orig_argv))
    saved_agent_cfg = copy.deepcopy(agent_cfg)
    agent_cfg.pop("central", None)
    if getattr(env_cfg, "reward_version", 1) >= 2 and env_cfg.reward_gamma != agent_cfg["gamma"]:
        raise ValueError("env.reward_gamma must equal agent.gamma for potential shaping")

    agent_cfg = process_sb3_cfg(agent_cfg, num_rows)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # Construction applies the motion fit: save the effective dynamics, not pre-fit defaults.
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    snapshot_run(log_dir, env_cfg, saved_agent_cfg, args_cli.task, sys.orig_argv)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = Sb3CentralVecEnv(env) if central else Sb3SharedPolicyVecEnv(env)
        print(f"[INFO] {'central policy: one controller for all agents' if central else 'shared per-agent policy'}")
        callbacks = [LogExtrasCallback(env)]
    else:
        env = Sb3VecEnvWrapper(env)
        callbacks = []

    frame_stack = agent_cfg.pop("frame_stack", 1)
    norm_args = {key: agent_cfg.pop(key) for key in
                 ("normalize_input", "normalize_value", "clip_obs", "freeze_normalization") if key in agent_cfg}
    if norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        if norm_args.get("freeze_normalization") and not args_cli.checkpoint:
            raise ValueError("Freezing normalization requires fitted statistics from a checkpoint")
        if args_cli.checkpoint:
            norm_path = norm_path_for(args_cli.checkpoint)
            if not norm_path.is_file():
                raise FileNotFoundError(f"Resume requires the matching normalizer: {norm_path}")
            env = VecNormalize.load(str(norm_path), env)
            env.training = not norm_args.get("freeze_normalization", False)
            env.norm_reward = norm_args.get("normalize_value", False)
            env.clip_obs = norm_args.get("clip_obs", env.clip_obs)
            env.gamma = agent_cfg["gamma"]
            print(f"[resume] Restored normalization from {norm_path}; count={env.obs_rms.count}")
        else:
            env = VecNormalize(
                env,
                training=True,
                norm_obs=norm_args["normalize_input"],
                norm_reward=norm_args.get("normalize_value", False),
                clip_obs=norm_args.get("clip_obs", 100.0),
                gamma=agent_cfg["gamma"],
                clip_reward=np.inf,
            )
    vec_norm = env if isinstance(env, VecNormalize) else None
    if frame_stack > 1:
        # the policy sees each drone's last `frame_stack` (normalized) observations
        print(f"Stacking the last {frame_stack} observations")
        env = VecFrameStack(env, n_stack=frame_stack)

    if args_cli.checkpoint is not None:
        agent = PPO.load(args_cli.checkpoint, env, print_system_info=True, tensorboard_log=log_dir, **agent_cfg)
        print(f"[resume] Starting at {agent.num_timesteps} transitions")
    else:
        agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    agent.verbose = 1  # a transferred artifact may have inherited silent inference verbosity
    if args_cli.action_std is not None or args_cli.value_scale is not None:
        prepare_finetune(agent, args_cli.action_std, args_cli.value_scale)
        print(f"[warm-start] action_std={args_cli.action_std}; value_scale={args_cli.value_scale}")
    if args_cli.critic_warmup_iterations:
        callbacks.append(CriticWarmupCallback(args_cli.critic_warmup_iterations))

    # CheckpointCallback counts vec-env steps (one per iteration step), not transitions
    callbacks.append(
        CheckpointCallback(
            save_freq=args_cli.save_interval * agent_cfg["n_steps"],
            save_path=log_dir,
            name_prefix="model",
            save_vecnormalize=True,
            verbose=2,
        )
    )
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(total_timesteps=n_timesteps, callback=[stop_callback, *callbacks], progress_bar=True, log_interval=1,
                    reset_num_timesteps=args_cli.checkpoint is None)

    agent.save(os.path.join(log_dir, "model"))
    print(f"Saving to:\n{os.path.join(log_dir, 'model.zip')}")
    if vec_norm is not None:
        print("Saving normalization")
        vec_norm.save(os.path.join(log_dir, "model_vecnormalize.pkl"))
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Fly one episode in every env at once with an SB3 checkpoint and log it for render_grid.py.

Each env runs until its first episode ends (success, violation or time-out); after that its
trajectory is frozen. Saves positions, gate/landing state and each env's outcome to an .npz.

    python scripts/rollout_grid.py --checkpoint logs/sb3/AS-Tello-Approach-v0/<run>/model.zip \
        --out /mnt/data/isaac/tmp/grid_it1.npz [env.approach_azimuth_spread=3.14159 ...]
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Roll out one episode per env and log it for a grid video.")
parser.add_argument("--task", type=str, default="AS-Tello-Approach-v0")
parser.add_argument("--agent", type=str, default="sb3_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from isaaclab.envs import DirectMARLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import action_space_kit.tasks  # noqa: F401
from action_space_kit.rl.sb3_shared import Sb3SharedPolicyVecEnv



def vecnormalize_path(checkpoint_path: str) -> Path:
    """model.zip -> model_vecnormalize.pkl; model_N_steps.zip -> model_vecnormalize_N_steps.pkl (as play.py)."""
    p = Path(checkpoint_path)
    m = re.match(r"model_(\d+)_steps$", p.stem)
    return p.with_name(f"model_vecnormalize_{m.group(1)}_steps.pkl" if m else "model_vecnormalize.pkl")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    checkpoint = os.path.abspath(args_cli.checkpoint)

    base_env = Sb3SharedPolicyVecEnv(gym.make(args_cli.task, cfg=env_cfg))
    u = base_env.unwrapped_env
    env = base_env
    norm_path = vecnormalize_path(checkpoint)
    if norm_path.exists():
        env = VecNormalize.load(str(norm_path), env)
        env.training = False
        env.norm_reward = False
    if agent_cfg.get("frame_stack", 1) > 1:
        env = VecFrameStack(env, n_stack=agent_cfg["frame_stack"])
    agent = PPO.load(checkpoint, env)

    obs = env.reset()
    # a full reset staggers episode clocks so training envs don't finish in lockstep; here every
    # env should get the whole episode
    u.episode_length_buf[:] = 0
    E, D = u.num_envs, u.cfg.num_drones
    waypoints = u._waypoints.cpu().numpy().copy()
    spots = u._landing_spots.cpu().numpy().copy()
    finished = np.zeros(E, dtype=bool)
    end_step = np.full(E, -1)
    outcome = {k: np.zeros(E, dtype=bool) for k in ("success", "illegal_entry", "collision", "crash")}
    pos_log, gate_log, landed_log = [], [], []

    for t in range(int(u.max_episode_length)):
        # state before stepping: a finishing env is reset inside step(), so this is its last frame
        pos_log.append(u._positions().cpu().numpy().astype(np.float16))
        gate_log.append(u._gate_passed.cpu().numpy())
        landed_log.append(u._landed.cpu().numpy())
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        done_now = u.outcome_valid.cpu().numpy() & ~finished
        for k in outcome:
            outcome[k][done_now] = getattr(u, f"outcome_{k}").cpu().numpy()[done_now]
        end_step[done_now] = t
        finished |= done_now
        if finished.all():
            break
    n = len(pos_log)
    end_step[~finished] = n - 1

    np.savez_compressed(
        args_cli.out,
        pos=np.stack(pos_log),  # (T, E, D, 3), env-local
        gate_passed=np.stack(gate_log),
        landed=np.stack(landed_log),
        waypoints=waypoints,
        spots=spots,
        end_step=end_step,
        timeout=~(outcome["success"] | outcome["illegal_entry"] | outcome["collision"] | outcome["crash"]),
        keepout_radius=u.cfg.keepout_radius,
        step_dt=u.step_dt,
        checkpoint=checkpoint,
        **outcome,
    )
    print(f"[grid] {E} envs, {n} steps, success {outcome['success'].sum()}/{E} -> {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

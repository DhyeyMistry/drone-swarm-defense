"""Live view of the room task over Isaac Sim's WebRTC livestream: the newest policy, flying now.

Opens the room headless with the livestream on and flies the newest checkpoint of the newest
run (or --checkpoint) in real time, every env overlapped in one room with the markers on. Every
--reload_min minutes it looks for a newer checkpoint and swaps it in without restarting, so the
view keeps up with a training run. Watch it with NVIDIA's "Isaac Sim WebRTC Streaming Client"
app on another computer, connecting to this desktop's address (over Tailscale from anywhere:
the desktop's 100.x address; a Cloudflare quick tunnel can't carry WebRTC).

    python -u scripts/stream_room.py > outputs/stream_room.log 2>&1        # then connect the client
    python -u scripts/stream_room.py --task AS-Tello-Room-v0 --num_envs 8

Checkpoints are exported to TorchScript in a clean interpreter first (rl/policy_export.py): the
livestream loads Kit's bundled numpy 1.26, which can't unpickle the numpy-2 checkpoint files,
so sb3/play.py can't be used with --livestream.
"""

import argparse
import sys

parser = argparse.ArgumentParser(description="Stream the newest room-task policy live.")
parser.add_argument("--task", choices=["AS-Tello-Room-v0", "AS-Tello-Room-Single-v0"], default="AS-Tello-Room-Single-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="fixed checkpoint (default: newest, reloaded)")
parser.add_argument("--num_envs", type=int, default=16, help="drones' envs on screen, all in one room")
parser.add_argument("--reload_min", type=float, default=5.0, help="minutes between checks for a newer checkpoint")
parser.add_argument("--livestream", type=int, default=2, help="1: public-IP WebRTC, 2: private network (LAN / Tailscale)")
parser.add_argument("--set", nargs="*", default=[], help="extra env cfg overrides, key=python_literal (dots for nesting)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "livestream": args.livestream})
simulation_app = app_launcher.app

import ast  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.vec_env import VecFrameStack  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
from action_space_kit.rl.policy_export import export_policy, latest_checkpoint  # noqa: E402
from action_space_kit.rl.sb3_shared import Sb3CentralVecEnv  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

TASK = args.task
OUT = Path("outputs/stream")


def kalman_of(ckpt: Path) -> bool | None:
    """kalman.enabled the checkpoint's run was trained with (params/env.yaml), if recorded."""
    p = ckpt.parent / "params" / "env.yaml"
    if not p.exists():
        return None
    m = re.search(r"^kalman:\s*\n\s+enabled:\s*(true|false)", p.read_text(), re.M)
    return None if m is None else m.group(1) == "true"


def newest() -> Path:
    return Path(args.checkpoint).resolve() if args.checkpoint else latest_checkpoint(Path("logs", "sb3", TASK)).resolve()


def load(ckpt: Path, n_stack: int):
    OUT.mkdir(parents=True, exist_ok=True)
    pt = export_policy(ckpt, n_stack, OUT / "policy.pt")
    print(f"[stream] policy: {ckpt}", flush=True)
    return torch.jit.load(str(pt), map_location="cuda:0")


def main() -> int:
    ckpt = newest()
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"
    cfg.debug_vis = True
    k = kalman_of(ckpt)
    if k is not None:
        cfg.kalman.enabled = k
    for kv in args.set:
        key, v = kv.split("=", 1)
        obj = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, ast.literal_eval(v))
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    n_stack = load_cfg_from_registry(TASK, "sb3_cfg_entry_point").get("frame_stack", 1)
    venv = Sb3CentralVecEnv(env)
    if n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack)
    policy = load(ckpt, n_stack)
    print(f"[stream] {TASK}: {args.num_envs} envs in one room, kalman {cfg.kalman.enabled}; "
          f"streaming (livestream={args.livestream}). Connect the WebRTC client to this machine.", flush=True)

    obs = venv.reset()
    next_check = time.time() + 60 * args.reload_min
    episodes, wp, on_spot = 0, 0.0, 0.0
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            action = policy(torch.as_tensor(obs, dtype=torch.float32, device="cuda:0")).cpu().numpy()
        obs, _, dones, _ = venv.step(action)
        if dones.any():
            idx = torch.as_tensor(np.nonzero(dones)[0], device=u.device)
            episodes += len(idx)
            wp += (u.outcome_waypoints[idx] / cfg.num_drones).sum().item()
            on_spot += (u.outcome_land_ok[idx] / cfg.num_drones).sum().item()
            print(f"[stream] {episodes} episodes: waypoint reached {100 * wp / episodes:.0f}%, "
                  f"landed on spot {100 * on_spot / episodes:.0f}%", flush=True)
        if not args.checkpoint and time.time() >= next_check:
            next_check = time.time() + 60 * args.reload_min
            latest = newest()
            if latest != ckpt:
                ckpt, policy = latest, load(latest, n_stack)
                episodes, wp, on_spot = 0, 0.0, 0.0
        time.sleep(max(0.0, u.step_dt - (time.time() - t0)))  # real time: one rc command per 74 ms
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(code)

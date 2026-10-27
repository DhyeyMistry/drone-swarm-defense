"""Record a clip of the defend scenario by capturing the Kit viewport.

Replicator's render-variable path is broken on this machine, which takes out camera
sensors and Isaac Lab's built-in video recording. Kit viewport rendering works -- it is
what feeds the WebRTC livestream -- so this captures viewport frames and stitches them
into an mp4 with ffmpeg.

Captures are asynchronous: each one needs the app to tick a few times before the file
lands, which is why the tick counts below are generous.

    python scripts/record_clip.py --frames 150 --out /mnt/data/isaac/videos/clip.mp4
"""

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Record a defend-scenario clip.")
parser.add_argument("--frames", type=int, default=150, help="Frames to capture.")
parser.add_argument("--capture_every", type=int, default=2, help="Sim steps between captures.")
parser.add_argument("--ticks_per_capture", type=int, default=3, help="App ticks after each capture.")
parser.add_argument("--out", type=str, default="/mnt/data/isaac/videos/defend_chase.mp4")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--frame_dir", type=str, default="/mnt/data/isaac/tmp/clip_frames")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

# livestream mode is not optional here. In plain headless mode Isaac Lab sets
# /isaaclab/render/active_viewport to false and leaves the viewport context unset, so
# there is nothing to capture -- captures silently produce no files. Turning livestream
# on keeps the viewport pipeline alive, which is what makes these captures work.
app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

os.makedirs(args.frame_dir, exist_ok=True)
for stale in os.listdir(args.frame_dir):
    os.remove(os.path.join(args.frame_dir, stale))

env_cfg = parse_env_cfg("AS-Defend-v0", device="cuda:0", num_envs=1)
env_cfg.seed = 42
# frame the protected site so interceptions are visible
env_cfg.viewer.eye = (9.0, 9.0, 6.0)
env_cfg.viewer.lookat = (0.0, 0.0, 1.0)

env = gym.make("AS-Defend-v0", cfg=env_cfg).unwrapped
obs, _ = env.reset()

viewport = get_active_viewport()
print(f"[clip] viewport: {viewport}", flush=True)

# let the renderer settle before the first capture
for _ in range(30):
    simulation_app.update()


def capture(index: int) -> str:
    path = os.path.join(args.frame_dir, f"frame_{index:05d}.png")
    # wait_for_result is a coroutine, so there is nothing useful to call synchronously;
    # ticking the app is what actually flushes the capture to disk
    capture_viewport_to_file(viewport, path)
    for _ in range(args.ticks_per_capture):
        simulation_app.update()
    return path


num_attackers = env.cfg.num_attackers
captured = 0
step = 0

print(f"[clip] capturing {args.frames} frames", flush=True)
while captured < args.frames:
    # greedy pursuit: each defender flies at the nearest attacker
    actions = {}
    for agent in env.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * num_attackers].reshape(env.num_envs, num_attackers, 3)
        distance = torch.linalg.norm(rel, dim=-1)
        target = rel[torch.arange(env.num_envs, device=env.device), distance.argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)

    obs, _, _, _, _ = env.step(actions)
    env.sim.render()
    step += 1

    if step % args.capture_every == 0:
        path = capture(captured)
        captured += 1
        if captured == 1:
            # fail fast rather than simulating for minutes and writing nothing
            for _ in range(60):
                simulation_app.update()
            if not os.path.exists(path):
                print("[clip] first capture produced no file -- aborting", flush=True)
                env.close()
                simulation_app.close()
                sys.exit(1)
            print(f"[clip] first frame OK ({os.path.getsize(path)} bytes)", flush=True)

# let the trailing captures flush
for _ in range(120):
    simulation_app.update()

frames = sorted(f for f in os.listdir(args.frame_dir) if f.endswith(".png"))
print(f"[clip] wrote {len(frames)} frames", flush=True)

if not frames:
    print("[clip] no frames captured", flush=True)
    sys.exit(1)

import imageio_ffmpeg  # noqa: E402

os.makedirs(os.path.dirname(args.out), exist_ok=True)
cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(),
    "-y",
    "-framerate",
    str(args.fps),
    "-i",
    os.path.join(args.frame_dir, "frame_%05d.png"),
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-vf",
    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    args.out,
]
proc = subprocess.run(cmd, capture_output=True, text=True)
ok = proc.returncode == 0 and os.path.exists(args.out)
if ok:
    print(f"[clip] wrote {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
else:
    print(f"[clip] ffmpeg failed: {proc.stderr[-600:]}", flush=True)

# close last: simulation_app.close() exits the process, so nothing after it would run
env.close()
simulation_app.close()
sys.exit(0 if ok else 1)

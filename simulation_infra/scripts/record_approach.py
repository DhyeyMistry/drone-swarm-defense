"""Record a video of AS-Tello-Approach-v0, flown by a trained SB3 policy or the scripted pilot.

Camera sensors (Replicator) are broken on this server, so like ``record_clip.py`` this
captures Kit viewport frames and stitches them with ffmpeg. The camera is re-aimed at
the start of every episode, side-on to the approach. In frame: the keep-out sphere (red
dot shell), the touchdown zone (green disc), and per drone one color shared by its
beacon, its gate ring and its landing pad, so the assignment is visible.

    # trained policy (defaults to the latest run's model.zip)
    python scripts/record_approach.py --pilot policy --episodes 3 --out /mnt/data/isaac/videos/approach_policy.mp4
    # scripted reference pilot
    python scripts/record_approach.py --pilot scripted --episodes 3 --out /mnt/data/isaac/videos/approach_scripted.mp4
"""

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Record an AS-Tello-Approach-v0 video.")
parser.add_argument("--pilot", choices=["policy", "scripted"], default="policy")
parser.add_argument("--checkpoint", type=str, default=None, help="SB3 model .zip (default: latest run's model.zip).")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--max_seconds", type=float, default=20.0, help="Cap on each recorded episode.")
parser.add_argument("--no_dr", action="store_true", help="Nominal dynamics instead of randomized.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--spawn_azimuth_range", type=float, default=None, help="Override cfg.spawn_azimuth_range [rad].")
parser.add_argument("--legacy_geometry", action="store_true", help="Pre-it5 geometry (20 ft out, 10 ft box) for it1-it4 policies.")
parser.add_argument("--set", nargs="*", default=[], help="Extra cfg overrides, key=python_literal.")
parser.add_argument("--label", type=str, default="", help="Tag printed with each episode result.")
parser.add_argument("--ticks_per_capture", type=int, default=3, help="App ticks after each capture.")
parser.add_argument("--out", type=str, default="/mnt/data/isaac/videos/approach.mp4")
parser.add_argument("--frame_dir", type=str, default="/mnt/data/isaac/tmp/approach_frames")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

# Livestream mode keeps the viewport pipeline alive in headless mode; without it captures
# silently produce no files (see record_clip.py).
app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
from action_space_kit.rl.policy_export import export_policy  # noqa: E402
from action_space_kit.tasks.direct.tello_approach.scripted_pilot import scripted_actions  # noqa: E402
from action_space_kit.tasks.direct.tello_approach.tello_approach_env_cfg import TelloApproachEnvCfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

TASK = "AS-Tello-Approach-v0"


def aim_camera(u) -> None:
    """Side-on view of env 0: looking across the approach, from above and to the left."""
    a = float(u._spawn_azimuth[0])
    origin = u.scene.env_origins[0].tolist()
    mid = 0.45 * float(u.cfg.spawn_distance)  # between the base and the spawn line
    back = 0.95 * float(u.cfg.spawn_distance)
    look = [origin[0] + mid * math.cos(a), origin[1] + mid * math.sin(a), origin[2] + 0.6]
    side = [-math.sin(a), math.cos(a)]
    eye = [look[0] + back * side[0], look[1] + back * side[1], origin[2] + 0.65 * back]
    u.sim.set_camera_view(eye=eye, target=look)


def main() -> int:
    os.makedirs(args.frame_dir, exist_ok=True)
    for stale in os.listdir(args.frame_dir):
        os.remove(os.path.join(args.frame_dir, stale))

    cfg = TelloApproachEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.sim.device = "cuda:0"
    cfg.debug_vis = True
    if args.spawn_azimuth_range is not None:
        cfg.spawn_azimuth_range = args.spawn_azimuth_range
    if args.legacy_geometry:
        FT = 0.3048
        cfg.spawn_distance, cfg.spawn_square, cfg.spawn_spacing = 20.0 * FT, 10.0 * FT, (0.6, 0.9)
        cfg.approach_elevation = (30.0 * math.pi / 180.0, 60.0 * math.pi / 180.0)
        cfg.max_height, cfg.max_xy, cfg.separation_distance, cfg.obs_clip_pos = 4.0, 10.0, 0.5, 10.0
    import ast
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(cfg, k, ast.literal_eval(v))
    if args.no_dr:
        cfg.dynamics.domain_randomization = False
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped

    if args.pilot == "policy":
        from stable_baselines3.common.vec_env import VecFrameStack

        from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry

        from action_space_kit.rl.sb3_shared import Sb3SharedPolicyVecEnv

        ckpt = args.checkpoint or get_checkpoint_path(
            os.path.abspath(os.path.join("logs", "sb3", TASK)), ".*", "model.zip", sort_alpha=False
        )
        ckpt = os.path.abspath(ckpt)
        agent_cfg = load_cfg_from_registry(TASK, "sb3_cfg_entry_point")
        n_stack = agent_cfg.get("frame_stack", 1)
        policy = torch.jit.load(str(export_policy(ckpt, n_stack, os.path.join(args.frame_dir, "policy.pt"))), map_location="cuda:0")
        venv = Sb3SharedPolicyVecEnv(env)
        if n_stack > 1:
            venv = VecFrameStack(venv, n_stack=n_stack)
        print(f"[rec] policy: {ckpt}", flush=True)
        obs = venv.reset()

        def step():
            nonlocal obs
            with torch.inference_mode():
                action = policy(torch.as_tensor(obs, dtype=torch.float32, device="cuda:0")).cpu().numpy()
            obs, _, dones, _ = venv.step(action)
            return bool(dones.any())

        def restart():
            nonlocal obs
            obs = venv.reset()

    else:
        env.reset()
        phase = torch.zeros(1, cfg.num_drones, dtype=torch.long, device=u.device)

        def step():
            _, _, term, trunc, _ = env.step(scripted_actions(u, phase))
            done = bool(term["drone_0"].any() or trunc["drone_0"].any())
            if done:
                phase.zero_()
            return done

        def restart():
            env.reset()
            phase.zero_()

    viewport = get_active_viewport()
    aim_camera(u)
    for _ in range(30):
        u.sim.render()  # renders without stepping physics

    frame = 0

    # Ticks go through sim.render(), which pauses physics around the app update. A bare
    # simulation_app.update() also steps physics without the drones' controller forces,
    # so the drones free-fall between policy steps.
    def capture():
        nonlocal frame
        capture_viewport_to_file(viewport, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
        for _ in range(args.ticks_per_capture):
            u.sim.render()  # renders without stepping physics
        frame += 1

    max_steps = int(args.max_seconds / u.step_dt)
    results = []
    spans = []  # (first frame, last frame, headline, outcome) per episode, for the overlay
    for ep in range(args.episodes):
        aim_camera(u)
        for _ in range(5):
            u.sim.render()  # renders without stepping physics
        first, steps, done = frame, 0, False
        while steps < max_steps:
            done = step()
            steps += 1
            if done:
                break  # the env has already auto-reset: this frame would show the next episode's start
            u.sim.render()
            capture()
        success = bool(u.outcome_success[0])
        outcome = (
            f"episode {ep + 1}: {'SUCCESS' if success else 'not complete'} -- "
            f"gates {int(u.outcome_gates[0])}/4, landed {int(u.outcome_landed[0])}/4, "
            f"illegal entry {bool(u.outcome_illegal_entry[0])}, collision {bool(u.outcome_collision[0])}, "
            f"crash {bool(u.outcome_crash[0])}, near_base {float(u.outcome_near_base[0]):.2f}, "
            f"{steps * u.step_dt:.1f} s"
        )
        short = (
            f"{'SUCCESS' if success else 'FAIL'}  gates {int(u.outcome_gates[0])}/4  landed {int(u.outcome_landed[0])}/4  "
            f"near base {float(u.outcome_near_base[0]) * 100:.0f}%"
            + ("  illegal entry" if bool(u.outcome_illegal_entry[0]) else "")
            + ("  collision" if bool(u.outcome_collision[0]) else "")
        )
        if not done:
            outcome = f"episode {ep + 1}: cut at {args.max_seconds:.0f} s"
            short = f"cut at {args.max_seconds:.0f} s"
            restart()
        results.append(outcome)
        print(f"[rec] {args.label} {outcome}", flush=True)
        # hold this episode's last frame for 1.5 s with its result, so attempts are clearly separated
        if frame > first:
            import shutil

            last = os.path.join(args.frame_dir, f"frame_{frame - 1:05d}.png")
            for _ in range(int(1.5 / u.step_dt)):
                for _ in range(200):
                    if os.path.exists(last):
                        break
                    u.sim.render()  # the viewport capture is async; let it land before copying
                shutil.copyfile(last, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
                frame += 1
        spans.append((first, frame - 1, steps, short))

    for _ in range(120):
        u.sim.render()  # renders without stepping physics
    frames = sorted(f for f in os.listdir(args.frame_dir) if f.endswith(".png"))
    print(f"[rec] wrote {len(frames)} frames", flush=True)
    if not frames:
        return 1

    # overlay "attempt k/N", and the result over the held end frames
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    hold = int(1.5 / u.step_dt)
    for k, (a, b, steps, short) in enumerate(spans):
        for i in range(a, b + 1):
            path = os.path.join(args.frame_dir, f"frame_{i:05d}.png")
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB")
            d = ImageDraw.Draw(img)
            lines = [f"{args.label}  attempt {k + 1}/{len(spans)}"]
            if i > b - hold:
                lines.append(short)
            y = 14
            for line in lines:
                d.rectangle(d.textbbox((14, y), line, font=font), fill=(0, 0, 0))
                d.text((14, y), line, font=font, fill=(255, 255, 255))
                y += 38
            img.save(path)

    import imageio_ffmpeg

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fps = round(1.0 / u.step_dt)  # one frame per policy step: real time
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps),
        "-i", os.path.join(args.frame_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.out,
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(args.out):
        print(f"[rec] ffmpeg failed: {proc.stderr[-600:]}", flush=True)
        return 1
    print(f"[rec] wrote {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        # Exit directly: with livestream on, simulation_app.close() can hang in shutdown
        # after the video is already written.
        sys.stdout.flush()
        os._exit(code)

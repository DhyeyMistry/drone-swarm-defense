"""Render a rollout_grid.py log as a top-down video of every env at once.

One tile per env: the keep-out sphere (red circle), each drone's gate (ring in its color)
and the drones with short trails. When an env's episode ends its tile turns green
(all four landed) or red (illegal entry, collision, crash) or grey (time-out) and freezes.
The header keeps a running count.

    python scripts/render_grid.py /mnt/data/isaac/tmp/grid_it1.npz /mnt/data/isaac/videos/grid_it1.mp4
"""

import argparse
import math

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("log")
parser.add_argument("out")
parser.add_argument("--title", default=None)
parser.add_argument("--trail", type=int, default=30, help="Trail length [steps].")
parser.add_argument("--half", type=float, default=8.0, help="Half-width of a tile [m].")
parser.add_argument("--hold", type=float, default=3.0, help="Seconds to hold the final frame.")
args = parser.parse_args()

DRONE_COLORS = [(1.0, 0.5, 0.0), (0.0, 0.8, 1.0), (1.0, 0.2, 0.8), (0.55, 1.0, 0.2)]  # as the env's markers
TILE_BG = {"flying": "#1b1f24", "success": "#1f5c2e", "fail": "#6b1f1f", "timeout": "#4a4a4a"}

d = np.load(args.log)
pos = d["pos"].astype(np.float32)  # (T, E, D, 3)
T, E, D, _ = pos.shape
end = d["end_step"]
success, timeout = d["success"], d["timeout"]
fail = ~success & ~timeout
r_keep = float(d["keepout_radius"])
dt = float(d["step_dt"])

cols = int(math.ceil(math.sqrt(E * 16 / 9)))
rows = int(math.ceil(E / cols))
size = 2.0 * args.half
off = np.stack([(np.arange(E) % cols) * size, -(np.arange(E) // cols) * size], axis=-1)  # tile centres

W, H = 1920, 1080
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="black")
ax = fig.add_axes([0.0, 0.0, 1.0, 0.94])
ax.set_facecolor("black")
ax.set_xlim(-args.half, cols * size - args.half)
ax.set_ylim(-(rows - 1) * size - args.half, args.half)
ax.set_aspect("equal")
ax.axis("off")

tiles = PatchCollection(
    [Rectangle((x - args.half * 0.97, y - args.half * 0.97), size * 0.97, size * 0.97) for x, y in off],
    facecolors=TILE_BG["flying"], edgecolors="none",
)
ax.add_collection(tiles)
ax.add_collection(
    PatchCollection([Circle((x, y), r_keep) for x, y in off], facecolors="none", edgecolors="#e04040", linewidths=0.8)
)
wp = d["waypoints"][..., :2] + off[:, None, :]  # (E, D, 2)
gate_colors = np.tile(np.array(DRONE_COLORS), (E, 1))
ax.scatter(wp[..., 0].ravel(), wp[..., 1].ravel(), s=14, facecolors="none", edgecolors=gate_colors, linewidths=0.8)
trails = LineCollection([], linewidths=1.0)
ax.add_collection(trails)
dots = ax.scatter([], [], s=7, c=[], linewidths=0)
header = fig.text(0.01, 0.97, "", color="white", fontsize=18, family="monospace", va="center")

drone_colors = np.tile(np.array(DRONE_COLORS), (E, 1))  # row = env * D + drone
xy = pos[..., :2] + off[None, :, None, :]  # (T, E, D, 2)
title = args.title or str(d["checkpoint"]).split("/sb3/")[-1]

writer = imageio_ffmpeg.write_frames(args.out, (W, H), fps=int(round(1.0 / dt)), quality=8, macro_block_size=8)
writer.send(None)
n_frames = T + int(args.hold / dt)
for f in range(n_frames):
    t = min(f, T - 1)
    idx = np.minimum(t, end)  # frozen once the env is done
    now = xy[idx, np.arange(E)]  # (E, D, 2)
    lo = np.maximum(idx - args.trail, 0)
    segs = []
    for e in range(E):
        seg = xy[lo[e] : idx[e] + 1, e]  # (L, D, 2)
        segs.extend(seg[:, k] for k in range(D))
    trails.set_segments(segs)
    trails.set_color(drone_colors)
    dots.set_offsets(now.reshape(-1, 2))
    dots.set_color(drone_colors)

    done = end <= t
    bg = np.array([TILE_BG["flying"]] * E, dtype=object)
    bg[done & success] = TILE_BG["success"]
    bg[done & fail] = TILE_BG["fail"]
    bg[done & timeout] = TILE_BG["timeout"]
    tiles.set_facecolors(list(bg))
    header.set_text(
        f"{title}   t={t * dt:5.1f}s   "
        f"success {int((done & success).sum()):3d}   failed {int((done & fail).sum()):3d}   "
        f"timed out {int((done & timeout).sum()):3d}   flying {int((~done).sum()):3d}   / {E}"
    )
    fig.canvas.draw()
    writer.send(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
writer.close()

names = ("illegal_entry", "collision", "crash")
print(
    f"[grid] success {success.sum()}/{E}, failed {fail.sum()} ("
    + ", ".join(f"{k} {int((d[k] & fail).sum())}" for k in names)
    + f"), timed out {timeout.sum()} -> {args.out}",
    flush=True,
)

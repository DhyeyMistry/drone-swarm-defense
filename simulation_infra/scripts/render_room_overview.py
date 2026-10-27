"""One annotated overview image of AS-Tello-Room-v0, to judge the scene by eye.

Shows env 0 from behind and above the drones: the tag-10 wall with tags 8/9/10, the
floor stands with tags 6/7, the side wall with tag 5, both drones at their spawn
(sphere = true position, cube = the tag/odometry estimate) and each lane's landing spot
(disc at the 1.3 m standoff). Every mapped tag's label is drawn at its map.yaml position
projected through the render camera, so a label sitting on its tag means the placement is right.
Below it, a strip of head-on close-ups of every mapped tag, each run through pupil-apriltags.

    python -u scripts/render_room_overview.py --out outputs/room_overview.png > outputs/room_overview.log 2>&1
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default="outputs/room_overview.png")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: E402, F401
from action_space_kit.tasks.direct.tello_room.tello_room_env_cfg import TelloRoomEnvCfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

W, H, FOCAL, H_AP = 1600, 1000, 12.0, 20.955
EYE = np.array([10.2, -4.6, 4.4])
TARGET = np.array([3.4, 0.0, 0.7])


def main() -> int:
    cfg = TelloRoomEnvCfg()
    cfg.scene.num_envs = 1
    cfg.debug_vis = True
    env = gym.make("AS-Tello-Room-v0", cfg=cfg)
    env.reset()
    u = env.unwrapped
    u._reset_idx(torch.arange(1, device=u.device))
    zero = {a: torch.zeros(1, 5, device=u.device) for a in cfg.possible_agents}
    for _ in range(20):  # hover 1 s so the tag fixes arrive and the estimate settles
        env.step(zero)

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    viewport = get_active_viewport()
    viewport.resolution = (W, H)
    cam = UsdGeom.Camera.Define(stage, "/World/OverviewCam")
    cam.GetHorizontalApertureAttr().Set(H_AP)
    cam.GetVerticalApertureAttr().Set(H_AP * H / W)
    cam.GetFocalLengthAttr().Set(FOCAL)
    cam.GetClippingRangeAttr().Set((0.01, 100.0))
    origin = u.scene.env_origins[0].cpu().numpy()
    cam_xf = UsdGeom.Xformable(cam).AddTransformOp()
    viewport.camera_path = "/World/OverviewCam"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def basis(eye, target):
        f = (target - eye) / np.linalg.norm(target - eye)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        return f, r, np.cross(r, f)

    def shoot(eye, target, path):
        """Render env 0 from eye toward target (room frame) and return the image."""
        f, r, up_ = basis(eye, target)
        rows = [list(r) + [0.0], list(up_) + [0.0], list(-f) + [0.0], list(eye + origin) + [1.0]]
        cam_xf.Set(Gf.Matrix4d(*[float(v) for row in rows for v in row]))
        for _ in range(30):
            u.sim.render()
        if os.path.exists(path):
            os.remove(path)
        capture_viewport_to_file(viewport, path)
        for _ in range(300):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                break
            u.sim.render()
        for _ in range(5):
            u.sim.render()
        frame = cv2.imread(path)
        os.remove(path)
        return frame

    raw = os.path.splitext(args.out)[0] + "_raw.png"
    img = shoot(EYE, TARGET, raw)
    if img is None:
        print("[overview] no frame captured", flush=True)
        return 1
    fwd, right, up = basis(EYE, TARGET)

    # close-up of every mapped tag, head-on from ~3.5 printed widths (drones and markers hidden)
    import pupil_apriltags

    detector = pupil_apriltags.Detector(families="tag36h11")
    u.set_debug_vis(False)
    for i in range(cfg.num_drones):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_0/Drone_{i}")).MakeInvisible()
    tiles = []
    tile = W // len(u.room_tags)
    for tid, tag in sorted(u.room_tags.items()):
        eye = tag.pos + tag.normal * 3.5 * tag.printed_size_m
        shot = shoot(eye, tag.pos, raw)
        c = min(H, W) // 2
        crop = shot[H // 2 - c // 2:H // 2 + c // 2, W // 2 - c // 2:W // 2 + c // 2]
        ids = [d.tag_id for d in detector.detect(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))]
        ok = ids == [tid]
        crop = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.rectangle(crop, (0, 0), (tile, 30), (0, 0, 0), -1)
        cv2.putText(crop, f"tag {tid}: decodes {ids}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (120, 255, 120) if ok else (80, 80, 255), 1, cv2.LINE_AA)
        p = tag.pos
        cv2.putText(crop, f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m", (6, tile - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m", (6, tile - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        print(f"[overview] tag {tid} close-up decodes as {ids} {'ok' if ok else 'MISMATCH'}", flush=True)
        tiles.append(crop)

    f_px = FOCAL / H_AP * W

    def project(p):
        d = np.asarray(p) - EYE
        z = d @ fwd
        return int(W / 2 + f_px * (d @ right) / z), int(H / 2 - f_px * (d @ up) / z)

    def label(text, p, color, dy=-14):
        x, y = project(p)
        cv2.putText(img, text, (x - 4 * len(text), y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x - 4 * len(text), y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    for tid, tag in u.room_tags.items():
        x, y = project(tag.pos)
        cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        label(f"tag {tid}", tag.pos + np.array([0, 0, 0.5 * tag.printed_size_m]), (255, 255, 255))
    pos = u._positions()[0].cpu().numpy()
    pub = u.pub_pos[0].cpu().numpy()
    spots = u._spots[0].cpu().numpy()
    for i in range(cfg.num_drones):
        err = np.linalg.norm(pub[i] - pos[i])
        label(f"drone {i}  fix err {err * 100:.0f} cm", pos[i] + np.array([0, 0, 0.45]), (0, 255, 255))
        cv2.line(img, project(pos[i] + np.array([0, 0, 0.4])), project(pos[i] + np.array([0, 0, 0.08])), (0, 255, 255), 1)
        cv2.line(img, project(pos[i]), project(spots[i]), (120, 200, 120), 1, cv2.LINE_AA)  # drone -> its spot
        foot = pos[i] * np.array([1.0, 1.0, 0.0])  # drop line to the floor, so the start box reads right
        cv2.line(img, project(pos[i]), project(foot), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.circle(img, project(foot), 4, (0, 200, 255), -1, cv2.LINE_AA)
        label(f"spot {i}", spots[i] + np.array([0, 0, 0.3]), (200, 255, 200))
    # the start box on the floor
    cx, cy = cfg.spawn_box_center
    h = 0.5 * cfg.spawn_box_size
    corners = [np.array([cx + a, cy + b, 0.0]) for a, b in ((-h, -h), (-h, h), (h, h), (h, -h))]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        cv2.line(img, project(a), project(b), (0, 200, 255), 2, cv2.LINE_AA)
    label(f"start box {cfg.spawn_box_size:.0f} x {cfg.spawn_box_size:.0f} m, {cfg.spawn_range[0]:.0f}-{cfg.spawn_range[1]:.0f} m from tag 10",
          corners[0], (0, 200, 255), dy=30)
    label("origin (floor under tag 10)", np.zeros(3), (255, 200, 120), dy=26)
    banner = (f"AS-Tello-Room-v0  |  map.yaml tags {sorted(u.room_tags)}  |  tag 10 at {cfg.tag10_height} m  |  "
              "red + = map position  |  sphere = true drone, cube = published fix, ball = spot")
    cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(img, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    strip = np.concatenate(tiles, axis=1)
    strip = cv2.copyMakeBorder(strip, 0, 0, 0, W - strip.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.imwrite(args.out, np.concatenate([img, strip], axis=0))
    print(f"[overview] wrote {args.out}", flush=True)
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

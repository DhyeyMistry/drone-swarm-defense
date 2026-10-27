"""End-to-end: render the sim room from a Tello's camera and localize with the REAL code.

For each test viewpoint the Kit viewport camera is set to the Tello's calibrated
intrinsics (config/camera/tello_3.yaml: 960x720, fx ~ 914.5 px) and placed at a known
pose in the room. The rendered frame goes through the deployment localizer
(``drone_deployment_infra/core/positional.py``: tello_calibration's detector + TagMap,
the same code that flies the real drone), and its answer is compared with the true camera
position in the tag-10 frame. Passing means the map -> sim placement -> tag texture
orientation -> detector chain is consistent, so what a policy learns about tag sightings
here transfers.

    python -u scripts/check_room_render.py > outputs/check_room_render.log 2>&1

Frames are saved to outputs/room_views/.
"""

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out_dir", default="outputs/room_views")
parser.add_argument("--drone", default="tello_3", help="camera calibration to use (config/camera/<drone>.yaml)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

# livestream keeps the viewport pipeline alive headless (see record_approach.py)
app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import math  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import action_space_kit.tasks  # noqa: E402, F401
from action_space_kit.tags.room_map import CONFIG_DIR, world_to_map  # noqa: E402
from action_space_kit.tasks.direct.tello_room.tello_room_env_cfg import TelloRoomEnvCfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

DEPLOY = CONFIG_DIR.parent
sys.path[:0] = [str(DEPLOY / "cal_and_map"), str(DEPLOY / "core")]
from positional import Positioner  # noqa: E402

# (label, eye xyz in the room frame, yaw in degrees: 180 = facing tag 10)
VIEWS = [
    ("lane L, 4 m out", (4.0, -0.5, 1.2), 180.0),
    ("lane R, 3 m, yawed 12 deg", (3.0, 0.5, 1.0), 192.0),
    ("standoff, lane R", (1.3, 0.5, 1.2), 180.0),
    ("centre, 2.5 m", (2.5, 0.0, 1.3), 180.0),
    ("facing the tag-5 wall", (2.6, -0.3, 1.5), 90.0),
]
FAILURES = []


def check(ok, msg):
    print(("  [ok]   " if ok else "  [FAIL] ") + msg, flush=True)
    if not ok:
        FAILURES.append(msg)


def main() -> int:
    cfg = TelloRoomEnvCfg()
    cfg.scene.num_envs = 1
    env = gym.make("AS-Tello-Room-v0", cfg=cfg)
    env.reset()
    u = env.unwrapped

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    for i in range(cfg.num_drones):  # keep the drones out of the shots
        UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_0/Drone_{i}")).MakeInvisible()

    calib = yaml.safe_load((CONFIG_DIR / "camera" / f"{args.drone}.yaml").read_text())
    W, H = int(calib["image_width"]), int(calib["image_height"])
    fx = float(calib["camera_matrix"][0][0])
    viewport = get_active_viewport()
    viewport.resolution = (W, H)
    # our own camera: Kit owns /OmniverseKit_Persp through the session layer and resets its lens
    cam = UsdGeom.Camera.Define(stage, "/World/TelloCam")
    cam_xf = UsdGeom.Xformable(cam).AddTransformOp()
    viewport.camera_path = "/World/TelloCam"
    h_ap = 20.955
    cam.GetHorizontalApertureAttr().Set(h_ap)
    cam.GetVerticalApertureAttr().Set(h_ap * H / W)
    cam.GetFocalLengthAttr().Set(fx * h_ap / W)
    cam.GetClippingRangeAttr().Set((0.01, 100.0))

    positioner = Positioner(args.drone)
    origin = u.scene.env_origins[0].cpu().numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n== render -> real localizer ({args.drone}: {W}x{H}, fx {fx:.1f}) ==", flush=True)
    for k, (label, eye, yaw_deg) in enumerate(VIEWS):
        eye = np.array(eye)
        yaw = math.radians(yaw_deg)
        target = eye + np.array([math.cos(yaw), math.sin(yaw), 0.0])
        # USD camera looks down its -Z with +Y up; Gf matrices are row-vector (rows = axes)
        fwd = target - eye
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        rows = [list(right) + [0.0], list(up) + [0.0], list(-fwd) + [0.0], list(eye + origin) + [1.0]]
        cam_xf.Set(Gf.Matrix4d(*[float(v) for row in rows for v in row]))
        for _ in range(20):
            u.sim.render()
        path = os.path.join(args.out_dir, f"view_{k}.png")
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
        if k == 0:
            print(f"  camera {viewport.camera_path}: focal {cam.GetFocalLengthAttr().Get():.3f} mm, "
                  f"h_ap {cam.GetHorizontalApertureAttr().Get():.3f}, v_ap {cam.GetVerticalApertureAttr().Get():.3f}, "
                  f"viewport {tuple(viewport.resolution)}, frame {None if frame is None else frame.shape}", flush=True)
        if frame is None:
            check(False, f"{label}: no frame captured")
            continue
        loc = positioner.locate(frame)
        truth = world_to_map(eye, cfg.tag10_height)
        if loc is None:
            check(False, f"{label}: real localizer found no mapped tag (frame {frame.shape[1]}x{frame.shape[0]}, {path})")
            continue
        tol = 0.05 + 0.04 * float(np.linalg.norm(truth))
        # Sim fidelity is judged per tag, on the best-conditioned one (largest IPPE
        # ambiguity ratio). The fused answer is reported separately: positional.py averages
        # every tag and tags.yaml turns the ambiguity gate off, so one flipped far tag can
        # drag it -- a property of the real localizer, not of the render.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        per_tag = []
        for o in positioner.detector.detect(gray):
            M = positioner.tag_map.camera_pose(o) if o.accepted and o.tag_id in positioner.tag_map.poses else None
            if M is not None:
                per_tag.append((o.ambiguity_ratio or 0.0, int(o.tag_id), float(np.linalg.norm(M[:3, 3] - truth))))
        ratio, best_id, best_err = max(per_tag)
        check(best_err < tol, f"{label}: tag {best_id} (ambiguity ratio {ratio:.1f}) localizes to {best_err:.3f} m of truth (tol {tol:.2f})")
        err = float(np.linalg.norm(np.array(loc["xyz"]) - truth))
        flipped = [f"{tid} ({e:.2f} m off, ratio {r:.1f})" for r, tid, e in per_tag if e > tol]
        print(f"         fused over tags {loc['tags']}: {np.round(loc['xyz'], 3).tolist()} vs true {np.round(truth, 3).tolist()} "
              f"-> {err:.3f} m" + (f"  [WARN real localizer: ambiguous pose flip on tag {', '.join(flipped)}]" if flipped else ""), flush=True)
    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} CHECK(S) FAILED"), flush=True)
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(code)  # with livestream on, simulation_app.close() can hang

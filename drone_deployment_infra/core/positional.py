"""Where is the drone? Position in the tag-10 origin frame from any mapped AprilTag.

Given a camera frame, this detects AprilTags and, from any tag that is in the saved
room map (config/map.yaml, built by cal_and_map/scripts/05_map.py), returns the drone's
position relative to the origin tag -- no matter which mapped tag is in view. It wraps
the cal_and_map package (tello_calibration), which must be installed:

    pip install -e drone_deployment_infra/cal_and_map

Use it from a flight loop:

    pos = Positioner("tello_3")
    loc = pos.locate(frame_bgr)        # {"tag": 6, "xyz": [x, y, z]} in metres, or None
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tello_calibration.calib.io import camera_from_dict, load_yaml
from tello_calibration.tags.detect import TagDetector, load_tag_config
from tello_calibration.tags.mapping import TagMap
from tello_calibration.utils.config import CONFIG_DIR


class Positioner:
    def __init__(self, drone: str, map_path: Path | None = None,
                 camera_path: Path | None = None, tags_path: Path | None = None) -> None:
        map_path = Path(map_path) if map_path else CONFIG_DIR / "map.yaml"
        if not map_path.exists():
            raise FileNotFoundError(f"no map at {map_path}; build it with cal_and_map 05_map.py")
        if camera_path is None:
            camera_path = CONFIG_DIR / "camera" / f"{drone}.yaml"
            if not camera_path.exists():       # not calibrated yet: fall back to a shared default
                camera_path = CONFIG_DIR / "camera" / "default.yaml"
        if not Path(camera_path).exists():
            raise FileNotFoundError(f"no camera calib for {drone} and no camera/default.yaml")
        K, dist = camera_from_dict(load_yaml(Path(camera_path)))
        self.tag_map = TagMap.load(map_path)
        cfg = load_tag_config(tags_path) if tags_path else load_tag_config()
        self.detector = TagDetector(cfg, K, dist)

    def locate(self, frame_bgr: np.ndarray) -> dict | None:
        """Where the drone is, from ANY mapped tag(s) in view, or None if none are.

        Returns a dict with:
          tags            ids of the mapped tags used this frame
          xyz             drone position in the origin (tag-10) frame, metres,
                          averaged over every visible mapped tag (robust to one being hidden)
          range           straight-line distance to the origin (tag 10)
          to_origin_cam   unit-ish direction from the drone to the origin, in the CAMERA
                          frame (x right, y down, z forward) -- what a controller steers on
        Covering any single tag (even tag 10) does not matter as long as one mapped tag is seen.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        seen = []  # (distance, tag_id, M_cam_to_origin, position)
        for obs in self.detector.detect(gray):
            if obs.accepted and obs.tag_id in self.tag_map.poses:
                M = self.tag_map.camera_pose(obs)
                if M is not None:
                    p = M[:3, 3]
                    d = obs.distance_m if obs.distance_m is not None else float(np.linalg.norm(p))
                    seen.append((d, int(obs.tag_id), M, p))
        if not seen:
            return None
        seen.sort(key=lambda s: s[0])
        xyz = np.mean([s[3] for s in seen], axis=0)      # fuse position over all visible tags
        rot = seen[0][2][:3, :3]                          # orientation from the nearest tag
        to_origin = rot.T @ (-xyz)                        # direction to origin, in camera frame
        norm = float(np.linalg.norm(to_origin)) or 1.0
        return {
            "tags": [s[1] for s in seen],
            "xyz": [round(float(v), 3) for v in xyz],
            "range": round(float(np.linalg.norm(xyz)), 3),
            "to_origin_cam": [float(v / norm) for v in to_origin],
            "R": [[float(rot[i, j]) for j in range(3)] for i in range(3)],  # cam -> origin
        }

    def direction_to(self, result: dict, target_xyz) -> list[float]:
        """Unit direction from the drone to an arbitrary origin-frame point, in the CAMERA
        frame (x right, y down, z forward). Lets a policy steer to a landing spot that is
        offset from the origin. `result` is a locate() dict (uses its R and xyz)."""
        R = np.asarray(result["R"], float)
        p = np.asarray(result["xyz"], float)
        v = R.T @ (np.asarray(target_xyz, float) - p)
        n = float(np.linalg.norm(v)) or 1.0
        return [float(x / n) for x in v]

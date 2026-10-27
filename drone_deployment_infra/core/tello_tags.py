"""AprilTag sizes and drone pose in the tag-10 world frame.

Tag 10 is the origin. Tags 5, 6, and 7 are small tags whose poses are
measured by tello_calibrate.py and stored in tag_map.json. A sighting of
any mapped tag is converted into the pose of tag 10 in the camera, so the
rest of the flight code can keep one frame.

pupil-apriltags tag_size is the black square, not the white quiet border.
The inch lengths below are used as that black-square side.
"""

import json
from pathlib import Path

import numpy as np

INCH = 0.0254
ORIGIN_ID = 10
# tag id -> black-square side, meters
TAG_SIZE_M = {
    5: 5.2 * INCH,
    6: 5.2 * INCH,
    7: 5.2 * INCH,
    10: 22.5 * INCH,
}
MAP_PATH = Path(__file__).with_name("tag_map.json")


def _as_rot(rows):
    rotation = np.asarray(rows, dtype=float).reshape(3, 3)
    return rotation


def _as_vec(values):
    return np.asarray(values, dtype=float).reshape(3)


def identity_pose():
    return np.eye(3), np.zeros(3)


def load_map(path=None):
    """Return {tag_id: (R, t)} with tag points mapped into the tag-10 frame.

    p_world = R @ p_tag + t. Tag 10 is always the identity. Tags missing
    from the file are omitted, so a sighting of them cannot be localized.
    """
    poses = {ORIGIN_ID: identity_pose()}
    map_path = Path(path) if path else MAP_PATH
    if not map_path.is_file():
        return poses
    raw = json.loads(map_path.read_text())
    for key, item in raw.get("tags", {}).items():
        tag_id = int(key)
        if tag_id == ORIGIN_ID or tag_id not in TAG_SIZE_M:
            continue
        poses[tag_id] = (_as_rot(item["R"]), _as_vec(item["t"]))
    return poses


def save_map(poses, samples, path=None):
    """Write linked tag poses. poses values are (R, t) in the tag-10 frame."""
    tags = {}
    for tag_id in sorted(poses):
        rotation, translation = poses[tag_id]
        tags[str(tag_id)] = {
            "size_m": round(TAG_SIZE_M[tag_id], 6),
            "R": [[round(float(rotation[i, j]), 6) for j in range(3)] for i in range(3)],
            "t": [round(float(v), 4) for v in translation],
            "samples": int(samples.get(tag_id, 0)),
        }
    body = {
        "origin": ORIGIN_ID,
        "frame": "tag 10: x right, y down, z negative in front of the tag",
        "tags": tags,
    }
    map_path = Path(path) if path else MAP_PATH
    map_path.write_text(json.dumps(body, indent=2) + "\n")
    return map_path


def camera_in_tag(rotation, translation):
    """Camera position in that tag's frame. x right, y down, z negative in front."""
    return -_as_rot(rotation).T @ _as_vec(translation)


def origin_in_camera(tag_id, rotation, translation, poses):
    """Pose of tag 10 in the camera, given one sighting of tag_id.

    rotation, translation are pupil pose_R, pose_t for that tag:
    p_camera = R @ p_tag + t. Returns (R, t) for tag 10 in the same camera,
    or None when tag_id has no measured pose.
    """
    if tag_id not in poses:
        return None
    rotation = _as_rot(rotation)
    translation = _as_vec(translation)
    world_r, world_t = poses[tag_id]
    origin_r = rotation @ world_r.T
    origin_t = translation - rotation @ world_r.T @ world_t
    return origin_r, origin_t


def camera_in_world(tag_id, rotation, translation, poses):
    """Camera position in the tag-10 frame from one sighting, or None."""
    origin = origin_in_camera(tag_id, rotation, translation, poses)
    if origin is None:
        return None
    return camera_in_tag(*origin)


def _self_check():
    # Tag 5 is 1 m to the right of tag 10, same axes. Camera faces tag 5
    # from 2 m, so the camera is 1 m right and 2 m in front of tag 10.
    poses = {ORIGIN_ID: identity_pose(), 5: (np.eye(3), np.array([1.0, 0.0, 0.0]))}
    rotation = np.eye(3)
    translation = np.array([0.0, 0.0, 2.0])
    world = camera_in_world(5, rotation, translation, poses)
    direct = camera_in_world(ORIGIN_ID, rotation, translation, poses)
    if not np.allclose(world, [1.0, 0.0, -2.0]):
        raise SystemExit(f"tag 5 world pose {world}")
    if not np.allclose(direct, [0.0, 0.0, -2.0]):
        raise SystemExit(f"tag 10 world pose {direct}")
    if camera_in_world(6, rotation, translation, poses) is not None:
        raise SystemExit("unmapped tag 6 returned a pose")
    print("tag map self-check ok")


if __name__ == "__main__":
    _self_check()

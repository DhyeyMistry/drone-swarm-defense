"""Build a map of AprilTags in the origin tag's frame, and localize the camera in it.

Every detection gives a tag -> camera pose (TagObservation.rvec/tvec, OpenCV camera
frame: x right, y down, z forward). Two tags seen in one frame give their relative
pose; chaining those from the origin tag (tag 10) places every tag in one frame:

    X_origin = M_origin_tag @ X_tag          (4x4 homogeneous, R|t)

The map stores M_origin_tag per tag. Localization inverts a single sighting:

    M_origin_cam = M_origin_tag @ inv(M_tag_cam)

and the camera's position in the origin frame (the drone's location relative to tag 10)
is the translation column of M_origin_cam. Coordinates are in the origin tag's frame
(x right, y up, z out of the printed face), as drawn by the tag convention in detect.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from tello_calibration.calib.io import load_yaml, save_yaml
from tello_calibration.tags.detect import TagObservation


def pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """4x4 homogeneous tag -> camera transform from a Rodrigues rvec and tvec."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return M


def invert(M: np.ndarray) -> np.ndarray:
    R = M[:3, :3]
    t = M[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _rotation_mean(rotations: list[np.ndarray]) -> np.ndarray:
    """Average rotation via SVD of the summed matrices (proper rotation out)."""
    left, _, right = np.linalg.svd(sum(rotations))
    mean = left @ right
    if np.linalg.det(mean) < 0:
        left[:, -1] *= -1
        mean = left @ right
    return mean


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = _rotation_mean([T[:3, :3] for T in transforms])
    M[:3, 3] = np.mean([T[:3, 3] for T in transforms], axis=0)
    return M


def frame_edges(observations: Iterable[TagObservation]) -> list[tuple[int, int, np.ndarray]]:
    """For every ordered pair of accepted, sized tags in one frame, the edge i -> j.

    edge (i, j, M) means X_i = M @ X_j, i.e. M is tag j expressed in tag i's frame:
    M_i_j = inv(M_cam_i) @ M_cam_j.
    """
    posed = [(o.tag_id, pose_matrix(o.rvec, o.tvec))
             for o in observations if o.accepted and o.tvec is not None]
    edges = []
    for i, (id_i, cam_i) in enumerate(posed):
        for id_j, cam_j in posed[i + 1:]:
            m_ij = invert(cam_i) @ cam_j
            edges.append((id_i, id_j, m_ij))
            edges.append((id_j, id_i, invert(m_ij)))
    return edges


class MapBuilder:
    """Accumulate co-visibility edges across frames, then solve for the origin frame."""

    def __init__(self, origin_tag: int) -> None:
        self.origin_tag = origin_tag
        self._edges: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        self._seen: dict[int, int] = defaultdict(int)

    def add_frame(self, observations: Iterable[TagObservation]) -> None:
        observations = list(observations)
        for o in observations:
            if o.accepted and o.tvec is not None:
                self._seen[o.tag_id] += 1
        for i, j, m in frame_edges(observations):
            self._edges[(i, j)].append(m)

    def build(self) -> "TagMap":
        """Breadth-first from the origin, attaching each tag by its best-sampled edge."""
        averaged = {key: _mean_transform(mats) for key, mats in self._edges.items()}
        counts = {key: len(mats) for key, mats in self._edges.items()}
        poses = {self.origin_tag: np.eye(4)}
        samples = {self.origin_tag: 0}
        while True:
            best = None
            for (i, j), m in averaged.items():
                if i in poses and j not in poses:
                    n = counts[(i, j)]
                    if best is None or n > best[0]:
                        best = (n, j, poses[i] @ m)
            if best is None:
                break
            n, j, m_origin_j = best
            poses[j] = m_origin_j
            samples[j] = n
        return TagMap(self.origin_tag, poses, dict(self._seen), samples)


@dataclass
class TagMap:
    origin_tag: int
    poses: dict[int, np.ndarray]     # tag_id -> M_origin_tag (4x4), X_origin = M @ X_tag
    seen: dict[int, int]             # frames each tag was accepted in
    samples: dict[int, int]          # edges used to place each tag in the origin frame

    def mapped(self) -> list[int]:
        return sorted(self.poses)

    def camera_in_origin(self, obs: TagObservation) -> np.ndarray | None:
        """Camera (drone) position in the origin frame from one mapped-tag sighting."""
        pose = self.camera_pose(obs)
        return None if pose is None else pose[:3, 3].copy()

    def camera_pose(self, obs: TagObservation) -> np.ndarray | None:
        """Full 4x4 camera -> origin transform from one sighting of a mapped tag."""
        if obs.tag_id not in self.poses or obs.tvec is None:
            return None
        m_origin_tag = self.poses[obs.tag_id]
        m_tag_cam = pose_matrix(obs.rvec, obs.tvec)
        return m_origin_tag @ invert(m_tag_cam)

    def to_dict(self) -> dict:
        tags = {}
        for tag_id in sorted(self.poses):
            M = self.poses[tag_id]
            tags[int(tag_id)] = {
                "R": [[float(M[i, j]) for j in range(3)] for i in range(3)],
                "t": [float(M[i, 3]) for i in range(3)],
                "seen": int(self.seen.get(tag_id, 0)),
                "samples": int(self.samples.get(tag_id, 0)),
            }
        return {"origin_tag": int(self.origin_tag),
                "frame": "origin tag: x right, y up, z out of the tag face; meters",
                "tags": tags}

    def save(self, path: Path) -> None:
        save_yaml(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "TagMap":
        data = load_yaml(path)
        poses = {}
        for key, item in data["tags"].items():
            M = np.eye(4)
            M[:3, :3] = np.asarray(item["R"], np.float64)
            M[:3, 3] = np.asarray(item["t"], np.float64)
            poses[int(key)] = M
        seen = {int(k): int(v.get("seen", 0)) for k, v in data["tags"].items()}
        samples = {int(k): int(v.get("samples", 0)) for k, v in data["tags"].items()}
        return cls(int(data["origin_tag"]), poses, seen, samples)


def _self_check() -> None:
    """Synthetic 3-tag scene: origin 10, tag 5 one metre to its right, tag 6 beyond 5."""
    from types import SimpleNamespace

    def obs(tag_id, M_cam_tag):
        rvec, _ = cv2.Rodrigues(M_cam_tag[:3, :3])
        return SimpleNamespace(tag_id=tag_id, rvec=rvec.reshape(3),
                               tvec=M_cam_tag[:3, 3].copy(), accepted=True)

    def trans(x, y, z):
        M = np.eye(4)
        M[:3, 3] = (x, y, z)
        return M

    # Camera 2 m in front of the origin (origin at cam z=+2). Tags share the origin's
    # orientation; tag 5 is +1 m x, tag 6 is +2 m x, all in the origin frame.
    cam_origin = trans(0, 0, 2)
    builder = MapBuilder(origin_tag=10)
    builder.add_frame([obs(10, cam_origin), obs(5, cam_origin @ trans(1, 0, 0))])
    builder.add_frame([obs(5, cam_origin @ trans(1, 0, 0)), obs(6, cam_origin @ trans(2, 0, 0))])
    tag_map = builder.build()
    assert tag_map.mapped() == [5, 6, 10], tag_map.mapped()
    assert np.allclose(tag_map.poses[5][:3, 3], [1, 0, 0], atol=1e-6), tag_map.poses[5][:3, 3]
    assert np.allclose(tag_map.poses[6][:3, 3], [2, 0, 0], atol=1e-6), tag_map.poses[6][:3, 3]
    # Localize from the tag-6 sighting: it must recover the same camera point as a
    # direct origin sighting -- both place the camera at origin (0, 0, -2).
    direct = -cam_origin[:3, :3].T @ cam_origin[:3, 3]
    cam = tag_map.camera_in_origin(obs(6, cam_origin @ trans(2, 0, 0)))
    assert np.allclose(cam, direct, atol=1e-6), (cam, direct)
    print("mapping self-check ok")


if __name__ == "__main__":
    _self_check()

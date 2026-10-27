"""Defender localization on rendered camera images: both tag print styles, stale-map check."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from upright_positional import UprightPositioner

OUR = Path(__file__).resolve().parents[1]
K = np.array(yaml.safe_load(open(OUR.parent / "config" / "camera" / "tello_2.yaml"))["camera_matrix"], float)
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
SIZES = {10: 0.5715, 6: 0.13208}


def render(tags, cam, official=True):
    """Camera at map position `cam`, level, looking at the tag-10 wall; tags = [(id, centre)]."""
    img = np.full((720, 960), 255, np.uint8)
    R_cm = np.diag([1.0, -1.0, -1.0])                      # camera axes in the map frame
    for tid, centre in tags:
        tag = cv2.aruco.generateImageMarker(DICT, tid, 800)
        if official:                                        # official prints = aruco rotated 180 deg
            tag = cv2.rotate(tag, cv2.ROTATE_180)
        h = SIZES[tid] / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]]) + centre
        pc = (R_cm.T @ (obj - cam).T).T
        uv = (K @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:]
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [800, 0], [800, 800], [0, 800]]), np.float32(uv))
        warped = cv2.warpPerspective(tag, H, (960, 720), borderValue=255)
        mask = cv2.warpPerspective(np.full_like(tag, 255), H, (960, 720))
        img[mask > 0] = warped[mask > 0]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def map_file(tmp_path):
    eye = np.eye(3).tolist()
    data = {"origin_tag": 10, "frame": "x right, y up, z out",
            "tags": {10: {"R": eye, "t": [0, 0, 0]}, 6: {"R": eye, "t": [0.8, 0, 0]}}}
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize("official", [True, False])
def test_position_is_right_for_either_print_orientation(map_file, official):
    cam = np.array([0.4, -0.3, 1.8])
    loc = UprightPositioner("tello_2", map_path=map_file).locate(render([(10, np.zeros(3))], cam, official))
    assert loc is not None
    assert np.linalg.norm(np.array(loc["xyz"]) - cam) < 0.1
    assert loc["R"][1][1] < 0                               # camera "down" points down in the map


def test_a_moved_tag_is_distrusted_and_does_not_corrupt_the_fix(map_file):
    cam = np.array([0.2, -0.1, 2.5])
    # the map says tag 6 is 0.8 m right of tag 10; in the "room" it was moved 0.8 m LEFT
    img = render([(10, np.zeros(3)), (6, np.array([-0.8, 0.0, 0.0]))], cam)
    pos = UprightPositioner("tello_2", map_path=map_file)
    loc = pos.locate(img)
    assert 6 in pos.distrusted and loc["tags"] == [10]
    assert np.linalg.norm(np.array(loc["xyz"]) - cam) < 0.15

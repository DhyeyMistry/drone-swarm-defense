#!/usr/bin/env python3
"""Measure tags 5, 6, and 7 in the tag-10 frame from one walkaround video.

The video has to show tag 10 and each other tag with some shared view:
two tags in one frame, or a chain of frames that overlaps (10 with 6, then
6 with 5). A tag that never shares a view with the origin stays unmapped.

Usage:
  .venv/bin/python tello_calibrate.py recordings/walk.avi
"""

import argparse
from collections import defaultdict

import cv2
import numpy as np
from pupil_apriltags import Detector

from tello_dual_video import CAL_F_AT_960
from tello_tags import ORIGIN_ID, TAG_SIZE_M, identity_pose, save_map

# Detect at 1 m, then scale translation by the real black-square side.
NOMINAL_SIZE_M = 1.0
MIN_SPAN_PX = 24


def _rot_mean(rotations):
    total = np.zeros((3, 3))
    for rotation in rotations:
        total += rotation
    left, _, right = np.linalg.svd(total)
    mean = left @ right
    if np.linalg.det(mean) < 0:
        left[:, -1] *= -1
        mean = left @ right
    return mean


def _tag_in_other(rotation_a, translation_a, rotation_b, translation_b):
    """Pose of tag B in tag A's frame. p_a = R @ p_b + t."""
    rotation = rotation_a.T @ rotation_b
    translation = rotation_a.T @ (translation_b - translation_a)
    return rotation, translation


def _span(tag):
    corners = np.asarray(tag.corners, dtype=float)
    return float(np.linalg.norm(corners[0] - corners[2]))


def detect_frame(detector, gray):
    height, width = gray.shape
    focal = CAL_F_AT_960 * (width / 960.0)
    found = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(focal, focal, width / 2.0, height / 2.0),
        tag_size=NOMINAL_SIZE_M,
    )
    poses = {}
    for tag in found:
        tag_id = int(tag.tag_id)
        if tag_id not in TAG_SIZE_M or tag.pose_t is None or _span(tag) < MIN_SPAN_PX:
            continue
        rotation = np.asarray(tag.pose_R, dtype=float).reshape(3, 3)
        translation = np.asarray(tag.pose_t, dtype=float).reshape(3) * TAG_SIZE_M[tag_id]
        margin = float(getattr(tag, "decision_margin", 1.0))
        poses[tag_id] = (rotation, translation, margin)
    return poses


def average_edges(raw):
    edges = {}
    for key, samples in raw.items():
        edges[key] = (
            _rot_mean([item[0] for item in samples]),
            np.mean([item[1] for item in samples], axis=0),
            len(samples),
        )
    return edges


def link_from_origin(edges):
    """Greedy attach: each new tag uses its strongest edge onto an already linked tag."""
    linked = {ORIGIN_ID: (*identity_pose(), 0)}
    while True:
        best = None
        for src, (rotation_s, translation_s, _) in linked.items():
            for (parent, child), (rotation, translation, count) in edges.items():
                if parent != src or child in linked:
                    continue
                if best is None or count > best[0]:
                    world_r = rotation_s @ rotation
                    world_t = rotation_s @ translation + translation_s
                    best = (count, child, world_r, world_t, parent)
        if best is None:
            break
        count, child, world_r, world_t, _parent = best
        linked[child] = (world_r, world_t, count)
    poses = {tag_id: (rotation, translation) for tag_id, (rotation, translation, _) in linked.items()}
    samples = {tag_id: count for tag_id, (_, _, count) in linked.items()}
    return poses, samples


def calibrate(video, output, stride):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )
    raw = defaultdict(list)
    seen = defaultdict(int)
    index = 0
    used = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            poses = detect_frame(detector, gray)
            for tag_id in poses:
                seen[tag_id] += 1
            ids = list(poses)
            for i, tag_a in enumerate(ids):
                rotation_a, translation_a, margin_a = poses[tag_a]
                for tag_b in ids[i + 1:]:
                    rotation_b, translation_b, margin_b = poses[tag_b]
                    weight = min(margin_a, margin_b)
                    if weight <= 0:
                        continue
                    rotation, translation = _tag_in_other(
                        rotation_a, translation_a, rotation_b, translation_b
                    )
                    raw[(tag_a, tag_b)].append((rotation, translation))
                    raw[(tag_b, tag_a)].append(_tag_in_other(
                        rotation_b, translation_b, rotation_a, translation_a
                    ))
            used += 1
        index += 1
        if index % 200 == 0:
            print(f"frame {index}", flush=True)
    capture.release()
    if ORIGIN_ID not in seen:
        raise SystemExit("tag 10 never appeared, no origin")
    edges = average_edges(raw)
    poses, samples = link_from_origin(edges)
    print(f"frames {index}, sampled {used}")
    for tag_id in sorted(TAG_SIZE_M):
        hits = seen.get(tag_id, 0)
        if tag_id not in poses:
            print(f"tag {tag_id}  seen {hits}  UNLINKED")
            continue
        translation = poses[tag_id][1]
        print(
            f"tag {tag_id}  seen {hits}  samples {samples[tag_id]}"
            f"  x {translation[0]:.3f}  y {translation[1]:.3f}  z {translation[2]:.3f} m"
        )
    path = save_map(poses, samples, output)
    missing = [tag_id for tag_id in TAG_SIZE_M if tag_id not in poses]
    if missing:
        print("not in the map:", " ".join(str(tag_id) for tag_id in missing))
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Measure tags 5, 6, 7 in the tag-10 frame.")
    parser.add_argument("video")
    parser.add_argument("-o", "--output", default=None, help="defaults to tag_map.json beside this script")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("stride must be >= 1")
    calibrate(args.video, args.output, args.stride)


if __name__ == "__main__":
    main()

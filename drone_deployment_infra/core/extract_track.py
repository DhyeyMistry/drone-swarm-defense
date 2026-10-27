#!/usr/bin/env python3
"""Reconstruct where a drone thought it was from a flight recording.

Re-runs the same tag36h11 + tag-10 world-frame pipeline the live code uses
(tello_tags) over every frame of an .avi, and writes one JSONL row per sampled
frame: the drone's position in the tag-10 frame and which tags were visible.
This is the "where the drone thought it was / when it saw the tag" track that
feed the flight visualization.

  .venv/bin/python scripts/extract_track.py                 # last 3 recordings
  .venv/bin/python scripts/extract_track.py recordings/013_TELLO-1_*.avi

WiFi state is not in the recording (frames only exist while connected, and the
.avi drops no timestamps), so it is not reconstructed here -- future flights log
it live. Output goes next to the recordings: recordings/<name>.track.jsonl.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

ROOT = Path(__file__).resolve().parent.parent          # drone_deployment_infra/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # core/, for tello_tags

from tello_tags import TAG_SIZE_M as WORLD_TAG_SIZE_M
from tello_tags import camera_in_tag, load_map, origin_in_camera

# Same focal calibration as tello_dual_video (960 px wide stream).
CAL_F_AT_960 = 871.0
# Everything a flight produces lives in one place: recordings/. The video, the live
# track, the training data and this offline reconstruction all land here, so there is
# no hunting between recordings/ and outputs/.
RECORD_DIR = ROOT / "recordings"
OUT_DIR = RECORD_DIR
FPS = 30.0


def session_of(name):
    """Leading session number of a recording/track filename, e.g. '013'."""
    return name.split("_", 1)[0]


def already_tracked(avi):
    """True if this recording has a track already (live track wins over offline)."""
    if (OUT_DIR / (avi.stem + ".track.jsonl")).is_file():
        return True
    session = session_of(avi.stem)
    return any(session_of(p.name) == session for p in OUT_DIR.glob("*.live.jsonl"))


def pending_recordings():
    avis = sorted(RECORD_DIR.glob("*.avi"), key=lambda p: p.stat().st_mtime)
    return [a for a in avis if not already_tracked(a)]


def extract(path, stride=2):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {path}")
    detector = Detector(families="tag36h11", nthreads=4, quad_decimate=2.0)
    tag_map = load_map()
    samples = []
    index = 0
    hits = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            focal = CAL_F_AT_960 * (width / 960.0)
            tags = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(focal, focal, width / 2.0, height / 2.0),
                tag_size=1.0,
            )
            ids = []
            best = None  # (margin, drone_xyz_in_tag10)
            for tag in tags:
                if tag.pose_t is None:
                    continue
                tag_id = int(tag.tag_id)
                ids.append(tag_id)
                if tag_id not in WORLD_TAG_SIZE_M:
                    continue
                rotation = np.asarray(tag.pose_R, dtype=float).reshape(3, 3)
                raw_t = np.asarray(tag.pose_t, dtype=float).reshape(3) * WORLD_TAG_SIZE_M[tag_id]
                origin = origin_in_camera(tag_id, rotation, raw_t, tag_map)
                if origin is None:
                    continue
                margin = float(getattr(tag, "decision_margin", 0.0))
                if best is None or margin > best[0]:
                    best = (margin, camera_in_tag(*origin))
            row = {"frame": index, "t": round(index / FPS, 3), "ids": sorted(set(ids))}
            if best is not None:
                drone = best[1]
                row["x"] = round(float(drone[0]), 3)
                row["y"] = round(float(drone[1]), 3)
                row["z"] = round(float(drone[2]), 3)
                row["range_m"] = round(float(np.linalg.norm(drone)), 3)
                hits += 1
            samples.append(row)
        index += 1
    capture.release()
    return samples, index, hits


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args:
        paths = [Path(a) for a in args]
    else:
        paths = pending_recordings()
    if not paths:
        print("every recording already has a track; nothing to do")
        return
    for path in paths:
        samples, frames, hits = extract(path)
        out = OUT_DIR / (path.stem + ".track.jsonl")
        with open(out, "w", encoding="utf-8") as handle:
            for row in samples:
                handle.write(json.dumps(row) + "\n")
        print(f"{path.name}: {frames} frames, {hits} with a tag-10 pose -> {out}", flush=True)


if __name__ == "__main__":
    main()

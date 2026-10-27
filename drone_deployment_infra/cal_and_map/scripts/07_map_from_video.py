"""Build the tag map from recorded video by bundle adjustment (tello_calibration.tags.bundle).

Every frame with two or more tags adds their corners; all camera poses and tag poses are
solved together (origin tag fixed), and each tag gets a quality check: views, 1-sigma
uncertainty, view spread, and REDUNDANCY (the video split into alternating time blocks,
each half solved alone, the two positions compared). The map is written only if every
tag passes, unless --force. No drone needed.

    python scripts/07_map_from_video.py --video data/map_tello_4_20261024_201500.avi --drone tello_4
    python scripts/07_map_from_video.py --video a.avi b.avi --camera ../config/camera/tello_4.yaml --out /tmp/x.yaml

Several videos are solved as one map (same room, same tag placement). --camera applies to all
of them; with --drone, config/camera/<drone>.yaml.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from tello_calibration.calib.io import camera_from_dict, load_yaml, save_yaml
from tello_calibration.tags.bundle import Acceptance, BundleMapper, report, result_to_dict
from tello_calibration.tags.detect import TagDetector, load_tag_config
from tello_calibration.utils.config import CONFIG_DIR

log = logging.getLogger("map_from_video")


def usable(gate):
    """Corners count when decoded well and sharp enough; single-tag ambiguity and view angle
    don't matter here (the solve uses corners, not single-tag poses)."""
    def ok(o):
        return (o.tvec is not None and o.decision_margin >= gate.min_decision_margin
                and o.side_px >= gate.min_side_px and o.edge_width_px <= gate.max_edge_width_px)
    return ok


def add_video(mapper: BundleMapper, path: Path, detector: TagDetector, K, dist, every: int,
              index0: int, use) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n = used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % every == 0:
            obs = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            used += mapper.add_frame(obs, K, dist, index=index0 + n, usable=use) > 0
        n += 1
    cap.release()
    return n, used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, nargs="+", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--drone", help="camera = config/camera/<drone>.yaml, e.g. tello_4")
    src.add_argument("--camera", type=Path)
    ap.add_argument("--tags", type=Path, default=CONFIG_DIR / "tags.yaml")
    ap.add_argument("--out", type=Path, default=CONFIG_DIR / "map.yaml")
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame (default all)")
    ap.add_argument("--fast-detect", action="store_true",
                    help="detector decimation from tags.yaml (default: full resolution, sharper corners)")
    ap.add_argument("--min-views", type=int, default=Acceptance.min_views)
    ap.add_argument("--max-sigma-cm", type=float, default=Acceptance.max_sigma_cm)
    ap.add_argument("--max-split-cm", type=float, default=Acceptance.max_split_cm)
    ap.add_argument("--min-spread-deg", type=float, default=Acceptance.min_spread_deg)
    ap.add_argument("--force", action="store_true", help="write the map even if a tag fails its checks")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cam_path = args.camera or CONFIG_DIR / "camera" / f"{args.drone}.yaml"
    K, dist = camera_from_dict(load_yaml(cam_path))
    tags = load_tag_config(args.tags)
    if not args.fast_detect:
        tags.detector["quad_decimate"] = 1.0
    detector = TagDetector(tags, K, dist)
    sizes = {t: tags.size_m(t) for t in tags.sizes_mm if tags.size_m(t)}
    acc = Acceptance(args.min_views, args.max_sigma_cm, args.max_split_cm, args.min_spread_deg)
    mapper = BundleMapper(tags.origin_tag, sizes, acc)

    index0 = 0
    for v in args.video:
        n, used = add_video(mapper, v, detector, K, dist, args.every, index0, usable(tags.gate))
        log.info("%s: %d frames, %d with 2+ tags", v, n, used)
        index0 += n + 1000                       # keeps videos in separate redundancy blocks
    res = mapper.solve()
    if res is None:
        log.error("not enough frames with the origin tag and others to solve a map")
        return 1
    log.info("\n%s", report(res))
    if tags.origin_tag not in res.tag_map.poses:
        log.error("origin tag %d not in the solution; no map written", tags.origin_tag)
        return 1
    bad = [t for t, q in res.quality.items() if not q.ok]
    if bad and not args.force:
        log.error("\ntags %s failed their checks: NOT writing %s. Record more views of them "
                  "(from several angles, with tag %d or a good tag in frame), or --force.",
                  bad, args.out, tags.origin_tag)
        return 1
    if args.out.exists():
        bak = args.out.with_name(args.out.name + ".bak")
        bak.write_text(args.out.read_text())
        log.info("previous map backed up -> %s", bak)
    save_yaml(args.out, result_to_dict(res))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

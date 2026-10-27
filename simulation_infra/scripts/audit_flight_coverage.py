"""Audit real-flight data before fitting dynamics. Does not modify the fit or fly a drone."""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(paths):
    flights, all_rc = [], []
    channels = ("right", "forward", "up", "yaw")
    for path in paths:
        content = Path(path).read_bytes()
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if not rows:
            continue
        rc = np.array([r["rc"] for r in rows], float)
        t = np.array([r["t"] for r in rows], float)
        dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.3)
        height_valid = np.array([isinstance(r.get("h"), (int, float)) and r["h"] > 0.25 for r in rows])
        measured = rc[height_valid]
        coverage = {}
        for j, channel in enumerate(channels):
            coverage[channel] = {
                "airborne_rc_range": [float(measured[:, j].min()), float(measured[:, j].max())]
                if len(measured)
                else None,
                "positive_seconds": float(dt[height_valid & (rc[:, j] >= 5)].sum()),
                "negative_seconds": float(dt[height_valid & (rc[:, j] <= -5)].sum()),
                "command_changes": int((np.abs(np.diff(rc[:, j])) >= 5).sum()),
            }
        flights.append(
            {
                "file": str(Path(path).resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "rows": len(rows),
                "airborne_rows": int(height_valid.sum()),
                "missing_or_ground_height_rows": int((~height_valid).sum()),
                "modes": sorted({str(r.get("mode")) for r in rows}),
                "coverage": coverage,
                "pose_rotation_rows": sum("R" in r for r in rows),
                "filter_covariance_rows": sum("position_variance" in r and "velocity_variance" in r for r in rows),
            }
        )
        all_rc.append(measured)
    ready = {}
    for channel in channels:
        both_directions = [
            f
            for f in flights
            if min(f["coverage"][channel]["positive_seconds"], f["coverage"][channel]["negative_seconds"]) >= 2.0
        ]
        ready[channel] = {
            "flights_with_2s_each_direction": len(both_directions),
            "enough_for_train_and_holdout": len(both_directions) >= 3,
        }
    return {
        "flights": flights,
        "channels": ready,
        "coverage_ready": bool(flights) and all(r["enough_for_train_and_holdout"] for r in ready.values()),
        "criteria": (
            "At least 3 independent flights with >=2 seconds of each command sign per axis; "
            "this is a coverage heuristic, not model validation."
        ),
        "required_validation": [
            "Split by complete flight (and drone), never random adjacent samples.",
            "Verify camera/map/body axes and state-packet velocity units before fitting.",
            "Record capture/receive timestamps, pose rotation, accepted fixes, covariance, RC, and drone identity.",
            "Fit gains/response delay on training flights; report speed residuals on untouched flights.",
            "A one-sided logged envelope is insufficient: missing positive up/negative forward is missing evidence.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    paths = sorted({p for pattern in args.logs for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("No flight logs matched")
    result = audit(paths)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"AUDIT COMPLETE: {len(result['flights'])} flights; coverage_ready={result['coverage_ready']}; {report}")
    for key, row in result["channels"].items():
        print(f"  {key}: {row['flights_with_2s_each_direction']} flights with >=2s each direction")


if __name__ == "__main__":
    main()

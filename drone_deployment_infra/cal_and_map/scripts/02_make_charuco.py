"""Generate a ChArUco board PNG sized to this laptop's screen, plus its definition yaml.

The board is shown fullscreen on the screen and filmed by the drone for calibration.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

import cv2

from tello_calibration.calib.board import render_board, spec_for_screen
from tello_calibration.calib.io import save_board_spec

log = logging.getLogger("make_charuco")

BOARD_DIR = Path(__file__).resolve().parents[1] / "data" / "boards"


def screen_resolution() -> tuple[int, int] | None:
    """Resolution of the primary display from xrandr, or None if unavailable."""
    try:
        out = subprocess.run(["xrandr", "--current"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = (re.search(r" connected primary (\d+)x(\d+)\+", out)
             or re.search(r" connected (\d+)x(\d+)\+", out))
    return (int(match.group(1)), int(match.group(2))) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, help="screen width in px (default: detect)")
    parser.add_argument("--height", type=int, help="screen height in px (default: detect)")
    parser.add_argument("--out-dir", type=Path, default=BOARD_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if args.width and args.height:
        width, height = args.width, args.height
    else:
        detected = screen_resolution()
        if detected is None:
            log.error("Could not detect the screen resolution; pass --width and --height")
            return 1
        width, height = detected
        log.info("Detected primary screen: %dx%d", width, height)

    spec = spec_for_screen(width, height)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_{spec.squares_x}x{spec.squares_y}_{width}x{height}"
    png, yml = args.out_dir / f"{stem}.png", args.out_dir / f"{stem}.yaml"
    cv2.imwrite(str(png), render_board(spec))
    save_board_spec(yml, spec)

    log.info("Board: %s, %dx%d squares, square %d px, marker %d px (ratio %.3f)",
             spec.dictionary, spec.squares_x, spec.squares_y, spec.square_px,
             spec.marker_px, spec.marker_px / spec.square_px)
    log.info("Wrote %s", png)
    log.info("Wrote %s", yml)
    log.info("Next:")
    log.info("  1. Show it fullscreen at 100%%:  eog -f %s", png)
    log.info("     Do not zoom or resize afterwards; set screen brightness to max.")
    log.info("  2. Measure one square edge with a ruler in mm. For accuracy, measure across")
    log.info("     all %d squares of a row and divide by %d.", spec.squares_x, spec.squares_x)
    log.info("  3. Run: .venv/bin/python scripts/03_calibrate.py --drone tello_4 "
             "--square-mm <measured>")
    return 0


if __name__ == "__main__":
    sys.exit(main())

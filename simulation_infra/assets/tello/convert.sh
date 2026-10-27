#!/usr/bin/env bash
# Convert the Tello URDF to USD. Run from a machine with Isaac Lab installed.
#   ISAACLAB=/path/to/IsaacLab ./convert.sh [output.usd]
set -e
ISAACLAB=${ISAACLAB:-/mnt/data/isaac/IsaacLab}
OUT=${1:-/mnt/data/isaac/assets/tello.usd}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$ISAACLAB"
./isaaclab.sh -p scripts/tools/convert_urdf.py "$HERE/tello.urdf" "$OUT" --merge-joints --headless
echo "wrote $OUT"

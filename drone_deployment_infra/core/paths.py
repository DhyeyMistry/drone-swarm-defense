"""Where the deployment config lives: one place for every script.

All of it is in drone_deployment_infra/config/ (the simulator reads the same map.yaml and
tags.yaml through AS_DEPLOY_CONFIG's default, and cal_and_map's CONFIG_DIR points here too):

  dongles.json   USB dongle <-> drone (core/tello_dongle_setup.py writes it)
  fleet.yaml     which drones fly, roles, quick-connect list (core/fleet.py reads it)
  map.yaml       every tag's pose relative to tag 10 (swarm_ops/map_tags.py or cal_and_map write it)
  tags.yaml      tag sizes and detector settings
  camera/        per-drone calibration tello_<n>.yaml, default.yaml for uncalibrated drones
"""

from __future__ import annotations

from pathlib import Path

DD_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = DD_ROOT / "config"
DONGLES_PATH = CONFIG_DIR / "dongles.json"
FLEET_PATH = CONFIG_DIR / "fleet.yaml"
MAP_PATH = CONFIG_DIR / "map.yaml"
TAGS_PATH = CONFIG_DIR / "tags.yaml"
CAMERA_DIR = CONFIG_DIR / "camera"
DEFAULT_CAMERA = CAMERA_DIR / "default.yaml"


def camera_file(num: int | str, camera_dir: Path | str | None = None) -> Path:
    """Calibration for drone <num> (3 or "tello_3"); default.yaml if that drone has none."""
    name = str(num) if str(num).startswith("tello_") else f"tello_{num}"
    root = Path(camera_dir) if camera_dir else CAMERA_DIR
    path = root / f"{name}.yaml"
    if path.is_file():
        return path
    fallback = root / "default.yaml"
    if fallback.is_file():
        print(f"[paths] no calibration {path.name}; using {fallback} (less accurate)", flush=True)
        return fallback
    raise FileNotFoundError(f"no {path} and no {fallback}")

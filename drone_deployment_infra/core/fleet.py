"""Read the fleet config (roles, quick-connect list, count) from config/fleet.yaml.

The dongle -> drone mapping is separate (config/dongles.json, via tello_dongle_setup).
This is the higher-level question of which drones fly, their roles, and which ones are
quick-connected first for calibration / mapping / small tests.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DD_ROOT = Path(__file__).resolve().parent.parent
FLEET_PATH = DD_ROOT / "config" / "fleet.yaml"


def load_fleet_config(path: Path | None = None) -> dict:
    data = yaml.safe_load((Path(path) if path else FLEET_PATH).read_text()) or {}
    roles = data.get("roles") or {}
    return {
        "fly_count": int(data.get("fly_count", 0)),
        "attack": list(roles.get("attack") or []),
        "defense": list(roles.get("defense") or []),
        "quick_connect": list(data.get("quick_connect") or []),
    }


def quick_connect_drones(path: Path | None = None) -> list[str]:
    return load_fleet_config(path)["quick_connect"]


if __name__ == "__main__":
    import json

    print(json.dumps(load_fleet_config(), indent=2))

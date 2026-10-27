"""Load project config files from config/."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Central config lives at drone_deployment_infra/config (one level above cal_and_map).
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

# fleet.yaml `drones:` value meaning "this firmware has no sn? command; rely on BSSID".
SERIAL_UNSUPPORTED = "unsupported"
_BSSID_RE = re.compile(r"^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$")
_REQUIRED_KEYS = ("iface", "ssid", "bssid", "serial")


class DroneConfigError(ValueError):
    """fleet.yaml `drones:` entry missing or malformed."""


@dataclass(frozen=True)
class DroneConfig:
    name: str
    iface: str
    ssid: str
    bssid: str
    serial: str | None  # None = not recorded yet; SERIAL_UNSUPPORTED = no sn? on firmware
    local_ip: str | None
    notes: str


def load_drone_config(name: str, path: Path = CONFIG_DIR / "fleet.yaml") -> DroneConfig:
    with path.open() as f:
        drones = (yaml.safe_load(f) or {}).get("drones") or {}
    if name not in drones:
        raise DroneConfigError(f"Drone {name!r} not in {path} (known: {', '.join(drones)})")
    entry = drones[name] or {}

    missing = [key for key in _REQUIRED_KEYS if key not in entry]
    if missing:
        raise DroneConfigError(f"{path}: {name} is missing {', '.join(missing)}")
    bssid = str(entry["bssid"])
    if not _BSSID_RE.match(bssid):
        raise DroneConfigError(f"{path}: {name}.bssid {bssid!r} is not like 60:60:1F:C9:4B:CC")
    serial = entry["serial"]

    return DroneConfig(
        name=name,
        iface=str(entry["iface"]),
        ssid=str(entry["ssid"]),
        bssid=bssid,
        serial=None if serial is None else str(serial),
        local_ip=entry.get("local_ip"),
        notes=entry.get("notes", ""),
    )

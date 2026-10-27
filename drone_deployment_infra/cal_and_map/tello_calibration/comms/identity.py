"""Check that the drone we are talking to is the one named in fleet.yaml.

Pure functions returning (ok, detail) so they can be tested without hardware.
"""

from __future__ import annotations

from tello_calibration.utils.config import SERIAL_UNSUPPORTED, DroneConfig
from tello_calibration.utils.net import WifiLink


def is_unsupported_reply(reply: str) -> bool:
    return reply.startswith("unknown command") or reply == "error"


def check_wifi(drone: DroneConfig, wifi: WifiLink | None) -> tuple[bool, str]:
    if wifi is None:
        return False, f"{drone.iface} is not associated with any Wi-Fi network"
    if wifi.ssid != drone.ssid:
        return False, f"SSID is {wifi.ssid!r}, config expects {drone.ssid!r}"
    if wifi.bssid.upper() != drone.bssid.upper():
        return False, f"BSSID is {wifi.bssid}, config expects {drone.bssid}"
    return True, f"{wifi.ssid} / {wifi.bssid}"


def check_serial(drone: DroneConfig, reply: str | None) -> tuple[bool, str]:
    """Compare the sn? reply with fleet.yaml. A null serial in config never passes."""
    if reply is None:
        return False, "no reply to sn?"
    unsupported = is_unsupported_reply(reply)

    if drone.serial is None:
        if unsupported:
            return False, (f"firmware answered {reply!r}. If that is expected, set "
                           f"'serial: {SERIAL_UNSUPPORTED}' for {drone.name} in fleet.yaml")
        return False, f"sn? = {reply!r}. Add 'serial: \"{reply}\"' for {drone.name} in fleet.yaml"

    if drone.serial == SERIAL_UNSUPPORTED:
        if unsupported:
            return True, f"firmware has no sn? ({reply!r}), as configured; identity by BSSID"
        return False, f"config says sn? unsupported, but drone reports serial {reply!r}"

    if reply != drone.serial:
        return False, f"sn? = {reply!r}, config expects {drone.serial!r}"
    return True, f"sn? = {reply}"

"""Short audible cues for hands-busy operation. Never blocks, never raises."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SOUND_DIR = Path("/usr/share/sounds/freedesktop/stereo")
SOUNDS = {"capture": SOUND_DIR / "camera-shutter.oga", "done": SOUND_DIR / "complete.oga"}


def _player_command(sound: Path) -> list[str] | None:
    for player in ("paplay", "pw-play"):
        if shutil.which(player):
            return [player, str(sound)]
    if shutil.which("canberra-gtk-play"):
        return ["canberra-gtk-play", "-f", str(sound)]
    return None


def beep(kind: str) -> None:
    """Play the 'capture' or 'done' sound; fall back to the terminal bell."""
    sound = SOUNDS[kind]
    command = _player_command(sound) if sound.exists() else None
    if command is not None:
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    sys.stderr.write("\a" if kind == "capture" else "\a\a\a")
    sys.stderr.flush()

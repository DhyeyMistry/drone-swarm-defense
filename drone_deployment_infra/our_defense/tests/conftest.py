"""Make our_defense/, core/ and swarm_ops/ importable the way defend.py sees them."""

import sys
from pathlib import Path

OUR = Path(__file__).resolve().parents[1]
DD_ROOT = OUR.parent
for p in (OUR, DD_ROOT / "core"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(DD_ROOT / "swarm_ops") not in sys.path:
    sys.path.append(str(DD_ROOT / "swarm_ops"))       # last, as feeds.py does

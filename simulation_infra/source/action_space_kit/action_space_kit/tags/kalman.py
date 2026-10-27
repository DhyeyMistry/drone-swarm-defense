"""The laptop's Kalman filter, run in the sim on every drone of every env at once.

Loads ``drone_deployment_infra/core/positional_kalman_filter.py`` (the file the flight engine
uses) by path and runs its ``KalmanCV`` on torch tensors, so the policy is trained on exactly
the estimate it gets in flight. Nothing is copied: change the filter there and both sides change.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]
FILTER_PATH = Path(os.environ.get("AS_KALMAN_PATH", _REPO / "drone_deployment_infra" / "core" / "positional_kalman_filter.py"))


def _load():
    name = "as_positional_kalman_filter"
    if name in sys.modules:
        return sys.modules[name]
    if not FILTER_PATH.exists():
        raise FileNotFoundError(f"no Kalman filter at {FILTER_PATH} (set AS_KALMAN_PATH)")
    spec = importlib.util.spec_from_file_location(name, FILTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up while the class is built
    spec.loader.exec_module(mod)
    return mod


_mod = _load()
KalmanCfg = _mod.KalmanCfg
KalmanCV = _mod.KalmanCV

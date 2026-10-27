"""Load the same landing gate used by the hardware inference adapter."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[5] / "drone_deployment_infra/core/landing_gate.py"
_spec = importlib.util.spec_from_file_location("action_space_landing_gate", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
update_landing_gate = _module.update_landing_gate

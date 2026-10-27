"""Load the same estimated peer features as the offline inference adapter."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[5] / "drone_deployment_infra/core/team_features.py"
_spec = importlib.util.spec_from_file_location("action_space_team_features", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
peer_features = _module.peer_features
predicted_separation_cost = _module.predicted_separation_cost

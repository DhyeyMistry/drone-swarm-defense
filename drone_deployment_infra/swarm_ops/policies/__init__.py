"""Scripted swarm policies, one per file. Add a policy: write <name>.py with a Policy
subclass (see base.py) and register it in POLICIES below; swarm_policies.py --run uses it.
"""
from .base import Policy
from .formation import FormationPolicy
from .hover import HoverPolicy
from .levels import LevelsPolicy
from .spin import SpinPolicy
from .telemetry import TelemetryBus, TelemetryClient, TelemetryHub
from .umbrella import Umbrella

POLICIES = {"spin": SpinPolicy, "levels": LevelsPolicy, "hover": HoverPolicy,
            "formation": FormationPolicy}

__all__ = ["POLICIES", "Policy", "FormationPolicy", "HoverPolicy", "LevelsPolicy", "SpinPolicy", "Umbrella",
           "TelemetryBus", "TelemetryClient", "TelemetryHub"]

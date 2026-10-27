"""Detection and tracking helpers for the detect-track-defeat loop."""

from .tracking import ConstantVelocityTracker, TrackerCfg, gate_detections

__all__ = ["ConstantVelocityTracker", "TrackerCfg", "gate_detections"]

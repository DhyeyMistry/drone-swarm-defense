"""hover: take off and hold a height: target_cm, or (None) the height the takeoff left it at."""
from .base import Policy, hold_height


class HoverPolicy(Policy):
    name = "hover"

    def __init__(self, target_cm=None):
        self.target_cm = target_cm

    def setup(self, drones):
        super().setup(drones)
        self.target = {}

    def step(self, d, t):
        if d.num not in self.target:
            self.target[d.num] = self.target_cm or d.height_cm()   # else the takeoff height
        return (0, 0, hold_height(d, self.target[d.num]), 0)

    def describe(self, d):
        return f"{self.name} {self.target_cm}cm" if self.target_cm else self.name

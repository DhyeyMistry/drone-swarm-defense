"""levels: the lower-numbered half of its drones hover at 3 ft, the rest at 6 ft.

Odd counts put the extra drone low.
"""
from .base import HIGH_CM, LOW_CM, Policy, hold_height


class LevelsPolicy(Policy):
    name = "levels"

    def __init__(self, low_cm=LOW_CM, high_cm=HIGH_CM):
        self.low_cm, self.high_cm = low_cm, high_cm

    def setup(self, drones):
        super().setup(drones)
        nums = sorted(d.num for d in self.drones)
        n_low = (len(nums) + 1) // 2
        self.target = {n: (self.low_cm if i < n_low else self.high_cm) for i, n in enumerate(nums)}

    def step(self, d, t):
        return (0, 0, hold_height(d, self.target[d.num]), 0)

    def describe(self, d):
        return f"{self.name} {self.target[d.num]}cm"

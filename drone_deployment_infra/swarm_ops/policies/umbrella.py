"""umbrella: run several policies at once, each on its own drones.

  Umbrella({SpinPolicy(): [1, 2], LevelsPolicy(): [3, 4, 5]})

Each drone may be in one group only. Each policy's setup sees only its own drones. All
groups share one telemetry bus (self.bus, see telemetry.py).
"""
from .base import Policy
from .telemetry import TelemetryBus


class Umbrella(Policy):
    name = "umbrella"

    def __init__(self, groups, bus=None):
        self.bus = bus if bus is not None else TelemetryBus()
        self.groups = {p: list(nums) for p, nums in groups.items()}
        seen = {}
        for p, nums in self.groups.items():
            for n in nums:
                if n in seen:
                    raise ValueError(f"drone {n} is in both {seen[n].name} and {p.name}")
                seen[n] = p
        self.owner = seen

    def setup(self, drones):
        super().setup(drones)
        by_num = {d.num: d for d in self.drones}
        for p, nums in self.groups.items():
            p.bus = self.bus
            p.setup([by_num[n] for n in nums])

    def step(self, d, t):
        return self.owner[d.num].step(d, t)

    def describe(self, d):
        return self.owner[d.num].describe(d)

"""spin: defensive dummy. Take off, hold the takeoff height, spin in place."""
from .hover import HoverPolicy

SPIN_YAW_RC = 40             # rc yaw while spinning (sign = direction)


class SpinPolicy(HoverPolicy):
    name = "spin"

    def __init__(self, yaw_rc=SPIN_YAW_RC):
        super().__init__()
        self.yaw_rc = yaw_rc

    def step(self, d, t):
        a, b, c, _ = super().step(d, t)
        return (a, b, c, self.yaw_rc)

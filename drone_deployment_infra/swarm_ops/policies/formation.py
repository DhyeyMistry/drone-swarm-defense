"""formation: swarm_formation.py on its drones, as its own process.

swarm_formation needs video + tag localization, so it isn't stepped per tick like the
other policies: swarm_policies.py launches it with --only <its drones>, relays its output,
answers its FLY prompt once you have typed FLY, and sends it one SIGINT (= land all) when
the run ends. Its camera windows and keys (q = land, SPACE = emergency) still work.
It only avoids its own drones: keep the other groups out of the space in front of tag 10.
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "Type FLY:"


class FormationPolicy:
    name = "formation"
    external = True

    def __init__(self, extra=""):
        self.extra = shlex.split(extra)

    def command(self, nums, fly, config):
        return ([sys.executable, "-u", os.path.join(HERE, "swarm_formation.py"),
                 "--only", ",".join(str(n) for n in nums), "--config", str(config)]
                + (["--fly"] if fly else []) + self.extra)

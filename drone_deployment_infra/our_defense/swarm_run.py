#!/usr/bin/env python3
"""ONE command for attack + defence: their swarm_policies.py with our `defend` policy added.

  python swarm_run.py --run formation:8,9 --run defend:2,3,5 --fly

Takes exactly swarm_policies.py's arguments (--run <policy>:<drones>, --fly, --duration,
--formation-args, ...) plus --defend-args "<extra defend.py flags>". It runs THEIR
swarm_policies.py main() unchanged; the only addition is a `defend` policy, registered at
runtime (no repo file is edited). Like their `formation`, `defend` is an external policy:
the umbrella starts defend.py for those drones with its telemetry hub in $SWARM_TELEM (so
we read the attackers' formation/drone<N> poses), counts our PASS lines, sends FLY when
you type FLY -- every defender takes off at that moment -- and one SIGINT at the end
(Ctrl+C / --duration) lands them. Drones are dongle numbers, as in swarm_policies.py.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DD_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DD_ROOT / "core"))
sys.path.insert(0, str(DD_ROOT / "swarm_ops"))

import swarm_policies as sp  # noqa: E402  (their runner, unmodified)
from feeds import display_name  # noqa: E402

DEFEND_PY = HERE / "defend.py"


class DefendPolicy:
    """External policy: our defend.py flies these drones from its own process."""
    name = "defend"
    external = True

    def __init__(self, extra: str = "") -> None:
        self.extra = shlex.split(extra)

    def command(self, nums, fly, config):
        names = [display_name(int(n)) for n in nums]
        return ([sys.executable, "-u", str(DEFEND_PY), "--defenders", ",".join(names), "--swarm_external",
                 "--fly" if fly else "--dry_run"] + self.extra)

    # what the umbrella may call on any policy
    def setup(self, drones):
        self.drones = list(drones)

    def describe(self, d):
        return self.name


def main() -> int:
    argv = sys.argv[1:]
    defend_args = ""
    if "--defend-args" in argv:                 # ours, not swarm_policies.py's
        i = argv.index("--defend-args")
        defend_args = argv[i + 1] if i + 1 < len(argv) else ""
        del argv[i:i + 2]

    sp.POLICIES["defend"] = DefendPolicy       # so `--run defend:...` passes their name check
    their_parse = sp.parse_runs

    def parse_runs(runs, yaw_rc, formation_args, hover_cm=None):
        ours = [r for r in runs if r.partition(":")[0] == "defend"]
        rest = [r for r in runs if r.partition(":")[0] != "defend"]
        umbrella, externals = their_parse(rest, yaw_rc, formation_args, hover_cm)
        for r in ours:
            nums = [int(x) for x in r.partition(":")[2].split(",") if x.strip()]
            if not nums:
                sys.exit(f"bad --run {r!r}: use defend:<drones>, e.g. defend:2,3,5")
            externals.append((DefendPolicy(defend_args), nums))
        try:                                    # their Umbrella checks no drone is in two groups
            return sp.Umbrella(dict(umbrella.groups) | dict(externals)), externals
        except ValueError as ex:
            sys.exit(str(ex))

    sp.parse_runs = parse_runs
    sys.argv = [str(DD_ROOT / "swarm_ops" / "swarm_policies.py")] + argv
    sp.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())

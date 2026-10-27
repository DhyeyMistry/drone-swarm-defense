"""End to end: defend.py --sim with attackers published on a real swarm_policies telemetry hub."""

import os
import subprocess
import sys
import time
from pathlib import Path

OUR = Path(__file__).resolve().parents[1]
DD_ROOT = OUR.parent


def test_three_defenders_knock_two_attackers_in_simulation():
    sys.path.append(str(DD_ROOT / "swarm_ops"))
    from policies.telemetry import TelemetryBus, TelemetryHub

    bus = TelemetryBus()
    hub = TelemetryHub(bus)
    env = {**os.environ, "SWARM_TELEM": f"{hub.addr[0]}:{hub.addr[1]}"}
    proc = subprocess.Popen([sys.executable, "-u", str(OUR / "defend.py"), "--sim",
                             "--defenders", "TELLO-2,TELLO-3,TELLO-5", "--attackers", "auto",
                             "--feed", "swarm", "--status", "", "--seconds", "14", "--log_every_s", "0"],
                            cwd=OUR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    try:
        while proc.poll() is None and time.time() - t0 < 20:
            t = time.time() - t0
            # two attackers flying from tag 10's side toward the defenders' V
            for name, x0, vx in (("drone8", 1.1, 0.02), ("drone9", 2.0, -0.03)):
                bus.publish(f"formation/{name}", {"state": "APPROACH", "height_cm": 120,
                                                  "p": [x0 + vx * t, 0.0, 0.8 + 0.25 * t], "fix_age": 0.05})
            time.sleep(0.07)
        out = proc.communicate(timeout=10)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
        hub.close()
    assert proc.returncode == 0, out
    assert "feed: tracking attacker TELLO-8" in out and "feed: tracking attacker TELLO-9" in out
    assert "CONTACT" in out, out
    assert "Traceback" not in out, out

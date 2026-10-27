#!/usr/bin/env python3
"""One command for a practice duel: dummy attacker(s) + our defender(s), roles on the CLI.

Starts dummy_attacker.py for each --attackers drone (shuttles toward/away from tag 10 and
publishes its position as fleet telemetry), waits --head_start_s so the attackers are up
and localized, then starts defend.py with the --defenders, which reads that telemetry and
rams them. Every line is prefixed with its process. Ctrl-C lands everyone.

Each drone needs its own USB stick plugged in (config/dongles.json) and is connected by
exactly ONE process -- never list a drone on both teams.

  python duel.py --defenders TELLO-2 --attackers TELLO-4 --map home/map.yaml --tags home/tags.yaml          # dry run
  python duel.py --defenders TELLO-2 --attackers TELLO-4 --map home/map.yaml --tags home/tags.yaml \\
      --tag10_height_m 1.0 --room=-1.5,1.5,3.0 --site 0,1.5 --fly

Anything after `--` goes to defend.py only (e.g. -- --speed_cap 0.4 --guard_radius 0.8).
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def pump(proc: subprocess.Popen, tag: str) -> None:
    for line in proc.stdout:
        print(f"[{tag}] {line.rstrip()}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defenders", required=True, help="comma list, e.g. TELLO-2")
    ap.add_argument("--attackers", required=True, help="comma list, e.g. TELLO-4")
    ap.add_argument("--fly", action="store_true", help="really fly (default: dry run for everyone)")
    ap.add_argument("--map", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--tag10_height_m", type=float, default=None)
    ap.add_argument("--room", default=None, help="x_min,x_max,z_max; write --room=-1.5,1.5,3.0")
    ap.add_argument("--site", default=None)
    ap.add_argument("--feed_port", type=int, default=9100)
    ap.add_argument("--amp_m", type=float, default=0.5, help="attacker shuttle distance")
    ap.add_argument("--period_s", type=float, default=6.0, help="attacker shuttle period")
    ap.add_argument("--head_start_s", type=float, default=4.0,
                    help="attackers fly this long before the defenders start (0-4 s)")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--climb_cm", type=int, default=30, help="climb after takeoff (0 = off)")
    args, extra = ap.parse_known_args()
    if not 0 <= args.head_start_s <= 4:
        ap.error("--head_start_s must be between 0 and 4 s")
    if extra and extra[0] == "--":
        extra = extra[1:]
    defenders = [s.strip() for s in args.defenders.split(",") if s.strip()]
    attackers = [s.strip() for s in args.attackers.split(",") if s.strip()]
    if set(defenders) & set(attackers):
        ap.error(f"a drone can't be on both teams: {sorted(set(defenders) & set(attackers))}")

    common = []
    if args.map:
        common += ["--map", args.map]
    if args.tags:
        common += ["--tags", args.tags]
    procs: list[tuple[str, subprocess.Popen]] = []

    def start(tag: str, cmd: list[str]) -> None:
        print(f"[duel] start {tag}: {' '.join(cmd[1:])}", flush=True)
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1, start_new_session=True)     # our Ctrl-C reaches it via us, once
        threading.Thread(target=pump, args=(p, tag), daemon=True).start()
        procs.append((tag, p))

    fly = ["--fly"] if args.fly else []
    dfly = ["--fly"] if args.fly else ["--dry_run"]      # defend.py's config may say fly: true
    common += ["--climb_cm", str(args.climb_cm)]
    try:
        for a in attackers:
            start(a, [PY, "-u", str(HERE / "dummy_attacker.py"), a, *common, *fly,
                      "--feed", f"127.0.0.1:{args.feed_port}", "--amp_m", str(args.amp_m),
                      "--period_s", str(args.period_s), "--seconds", str(args.seconds + args.head_start_s)])
        print(f"[duel] attackers get a {args.head_start_s:.0f} s head start...", flush=True)
        time.sleep(args.head_start_s)
        dcmd = [PY, "-u", str(HERE / "defend.py"), "--defenders", ",".join(defenders),
                "--attackers", ",".join(attackers), *common, *dfly, "--feed_port", str(args.feed_port),
                "--seconds", str(args.seconds)]
        if args.tag10_height_m is not None:
            dcmd += ["--tag10_height_m", str(args.tag10_height_m)]
        if args.room:
            dcmd += [f"--room={args.room}"]
        if args.site:
            dcmd += ["--site", args.site]
        start("defend", dcmd + extra)
        while any(p.poll() is None for _, p in procs):
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("[duel] Ctrl-C -> landing everyone", flush=True)
    finally:
        for tag, p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)             # each script lands its own drones on SIGINT
        deadline = time.time() + 20
        for tag, p in procs:
            try:
                p.wait(timeout=max(0.5, deadline - time.time()))
            except subprocess.TimeoutExpired:
                print(f"[duel] {tag} did not exit; killing it (drone auto-lands in ~15 s)", flush=True)
                p.kill()
        for tag, p in procs:
            print(f"[duel] {tag} exit {p.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

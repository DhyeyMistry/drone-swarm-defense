#!/home/jaimeromero/action-space/.venv/bin/python
"""Walk-range test using AprilTag 10 as the distance ruler.

Put the laptop at the same X,Y as tag 10. This connects the drone and streams
(it never takes off), so the drone's measured distance to tag 10 is your
distance from the base station. Carry the drone away keeping tag 10 in the
camera and watch the range climb; when the WiFi link drops, the last range is
how far you got while staying connected.

  .venv/bin/python tello_range_test.py            # first plugged-in drone
  .venv/bin/python tello_range_test.py TELLO-1

Relies on the live track tello_dual_video.py writes (time, wifi, range_m), so
the result is also saved and can be replayed in the flight visualization.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DD_ROOT = Path(__file__).resolve().parent.parent      # drone_deployment_infra/
REPO = DD_ROOT.parent                                 # repo root holds the shared .venv
sys.path.insert(0, str(DD_ROOT / "core"))             # for tello_dongle_setup

from tello_dongle_setup import load_fleet, present_fleet

PYTHON = REPO / ".venv" / "bin" / "python"
ENGINE = DD_ROOT / "core" / "tello_dual_video.py"
TRACKS = DD_ROOT / "outputs" / "tracks"


def newest_live_after(ts):
    cands = [p for p in TRACKS.glob("*.live.jsonl") if p.stat().st_mtime >= ts - 1]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    fleet = load_fleet(args) if args else present_fleet()
    if not fleet:
        raise SystemExit("no drone found: plug a numbered stick in or pass a drone name")
    name, iface = fleet[0]["drone"], fleet[0]["iface"]

    print(f"range test on {name}. Put the laptop at tag 10; carry the drone away keeping tag 10 in view.")
    print("The drone's range to tag 10 is your distance from the base. The drone does NOT take off. Ctrl-C to stop.\n")
    subprocess.run(["nmcli", "connection", "up", name, "ifname", iface], capture_output=True, text=True, timeout=30)

    (ROOT / "outputs").mkdir(exist_ok=True)
    launch = time.time()
    engine_log = open(ROOT / "outputs" / "range_engine.log", "w")
    proc = subprocess.Popen(
        [str(PYTHON), "-u", str(ENGINE), "--no_fly", name],
        cwd=ROOT, stdout=engine_log, stderr=subprocess.STDOUT, start_new_session=True,
    )

    track = None
    for _ in range(100):
        track = newest_live_after(launch)
        if track:
            break
        time.sleep(0.3)
    if track is None:
        proc.terminate()
        raise SystemExit("engine never started a track -- see outputs/range_engine.log")
    print(f"reading {track.name}  (also open http://127.0.0.1:8770/ to watch the camera)\n")

    best = None
    last_range = None
    connected = None
    t = 0.0
    try:
        with open(track, encoding="utf-8") as handle:
            while True:
                line = handle.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                up = bool(row.get("wifi"))
                rng = row.get("range_m")
                t = row.get("t", t)
                if up and rng is not None:
                    last_range = rng
                    best = rng if best is None else max(best, rng)
                shown = f"{rng:.2f} m" if rng is not None else "no tag 10"
                if connected is None:
                    connected = up
                if up != connected:
                    if up:
                        print(f"[{t:6.1f}s] RECONNECTED at {shown}", flush=True)
                    else:
                        tail = f" last range {last_range:.2f} m" if last_range is not None else ""
                        print(f"[{t:6.1f}s] *** LINK DROPPED ***{tail}", flush=True)
                    connected = up
                print(f"[{t:6.1f}s] {'UP  ' if up else 'DOWN'} range {shown}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
        engine_log.close()
        print("\n--- range summary ---")
        if best is not None:
            print(f"max range while connected: {best:.2f} m (~{best * 3.281:.1f} ft) from the base (tag 10)")
        else:
            print("no tag-10 range captured -- keep tag 10 in the drone's camera while walking")
        print(f"track: {track}  (rebuild the viz to see range vs wifi over the walk)")


if __name__ == "__main__":
    main()

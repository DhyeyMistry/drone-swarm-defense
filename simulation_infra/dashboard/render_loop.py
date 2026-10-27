"""Keep a web page of simulation videos up to date while training runs.

Every --poll seconds (at most one recording per --min_gap, default 30 min): find the newest training run under logs/sb3/<task>/, take its newest
checkpoint, and if it hasn't been recorded yet, record one episode of it with
scripts/record_room.py (same task, same Kalman setting as the run was trained with). Videos
land in dashboard/render_site/videos/ next to index.html and videos.json, which
render_serve.sh serves on 127.0.0.1 behind a Cloudflare tunnel. The oldest videos past --keep
are deleted.

Run by render_serve.sh; by hand:
    python -u dashboard/render_loop.py --poll 60 --keep 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
SITE = HERE / "render_site"
VIDEOS = SITE / "videos"
TASKS = ("AS-Tello-Room-Single-v0", "AS-Tello-Room-v0", "AS-Tello-Room-Peer-v0")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks for a new checkpoint")
ap.add_argument("--keep", type=int, default=30, help="videos kept on the page")
ap.add_argument("--min_gap", type=float, default=1800.0,
                help="seconds between recordings (each takes GPU time from training; the newest checkpoint is recorded)")
ap.add_argument("--max_seconds", type=float, default=None, help="cap each recorded episode (default: full episode)")
ap.add_argument("--view", default="overview", choices=["overview", "side", "top"])
ap.add_argument("--once", action="store_true", help="record the newest checkpoint (if new) and exit")
args = ap.parse_args()


def log(msg: str) -> None:
    print(f"[render {datetime.now():%H:%M:%S}] {msg}", flush=True)


def newest_checkpoint() -> tuple[str, Path, Path] | None:
    """(task, run dir, checkpoint) of the most recently written checkpoint of any room task."""
    best = None
    for task in TASKS:
        root = SIM / "logs" / "sb3" / task
        if not root.is_dir():
            continue
        for run in root.iterdir():
            for ck in list(run.glob("model_*_steps.zip")) + list(run.glob("model.zip")):
                t = ck.stat().st_mtime
                if best is None or t > best[0]:
                    best = (t, task, run, ck)
    return best[1:] if best else None


def kalman_setting(run: Path) -> str | None:
    """'True' / 'False' from the run's params/env.yaml (kalman: enabled: ...), if recorded."""
    p = run / "params" / "env.yaml"
    if not p.exists():
        return None
    m = re.search(r"^kalman:\s*\n\s+enabled:\s*(true|false)", p.read_text(), re.M)
    return None if m is None else ("True" if m.group(1) == "true" else "False")


def steps_of(ck: Path) -> int | None:
    m = re.match(r"model_(\d+)_steps$", ck.stem)
    return int(m.group(1)) if m else None


def write_index(entries: list[dict]) -> None:
    (SITE / "videos.json").write_text(json.dumps(entries, indent=1))


def record(task: str, run: Path, ck: Path) -> dict | None:
    steps = steps_of(ck)
    name = f"{task}__{run.name}__{ck.stem}.mp4"
    out = VIDEOS / name
    rec_log = HERE / "run" / "render_record.log"
    cmd = [sys.executable, "-u", str(SIM / "scripts" / "record_room.py"), "--task", task, "--checkpoint", str(ck),
           "--episodes", "1", "--view", args.view, "--seed", str(int(time.time()) % 10000),
           "--label", f"{run.name}  {ck.stem}", "--out", str(out), "--frame_dir", str(HERE / "run" / "frames")]
    k = kalman_setting(run)
    if k is not None:
        cmd += ["--set", f"kalman.enabled={k}"]
    if args.max_seconds:
        cmd += ["--max_seconds", str(args.max_seconds)]
    log(f"recording {task} {run.name} {ck.stem} (kalman {k})")
    t0 = time.time()
    with open(rec_log, "w") as f:
        proc = subprocess.run(cmd, cwd=SIM, stdout=f, stderr=subprocess.STDOUT)
    text = rec_log.read_text(errors="ignore")
    results = re.findall(r"\[rec\] episode \d+: (.*)", text)
    if proc.returncode != 0 or not out.exists():
        log(f"recording failed (exit {proc.returncode}); see {rec_log}")
        return None
    log(f"wrote {name} in {time.time() - t0:.0f} s: {results[-1] if results else ''}")
    return {"file": f"videos/{name}", "task": task, "run": run.name, "checkpoint": ck.stem, "steps": steps,
            "kalman": k, "result": results[-1] if results else "", "recorded": datetime.now().isoformat(timespec="seconds")}


def main() -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    (HERE / "run").mkdir(exist_ok=True)
    idx = SITE / "videos.json"
    entries = json.loads(idx.read_text()) if idx.exists() else []
    entries = [e for e in entries if (SITE / e["file"]).exists()]
    write_index(entries)
    done = {(e["run"], e["checkpoint"]) for e in entries}
    log(f"watching {SIM / 'logs' / 'sb3'} every {args.poll:.0f} s, recording at most every {args.min_gap / 60:.0f} min; "
        f"{len(entries)} videos on the page")
    last = 0.0
    while True:
        found = newest_checkpoint()
        if found:
            task, run, ck = found
            fresh = time.time() - ck.stat().st_mtime > 5  # let the save finish
            if (run.name, ck.stem) not in done and fresh and time.time() - last >= args.min_gap:
                last = time.time()
                e = record(task, run, ck)
                done.add((run.name, ck.stem))
                if e:
                    entries.insert(0, e)
                    for old in entries[args.keep:]:
                        (SITE / old["file"]).unlink(missing_ok=True)
                    entries = entries[: args.keep]
                    write_index(entries)
        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Move flight data from the old outputs/ layout into recordings/, as the unified code writes it.

Since commit 4d2c4f1 everything a flight produces lives in recordings/:

    old (outputs/)                              new (recordings/)
    train/<session>_<drone>.jsonl          ->   <session>_<drone>.train.jsonl
    tracks/<session>_<drone>.live.jsonl    ->   <session>_<drone>.live.jsonl
    tracks/<name>.jsonl (offline rebuild)  ->   <name>.track.jsonl
    flight_viz.html                        ->   flight_viz.html

Training rows keep every field. Rows with a ``pos`` but no ``range`` (the 027-era
to-origin logs) get ``range`` = |pos|, the distance to tag 10 that tello_to_origin.py
logs; ``bearing`` needs the heading, which those logs never recorded, so it is not invented.

Empty files (a flight that never logged a row) are reported and left where they are.
An existing destination is never overwritten unless --force. Moves by default; --copy
keeps the originals.

  ./migrate_outputs.py --dry_run     # show what would happen
  ./migrate_outputs.py               # do it
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

DD_ROOT = Path(__file__).resolve().parent.parent
OLD = DD_ROOT / "outputs"
NEW = DD_ROOT / "recordings"


def plan() -> list[tuple[Path, Path, str]]:
    """(source, destination, kind) for every old-layout file."""
    moves = []
    for src in sorted((OLD / "train").glob("*.jsonl")):
        moves.append((src, NEW / f"{src.stem}.train.jsonl", "train"))
    for src in sorted((OLD / "tracks").glob("*.jsonl")):
        if src.name.endswith(".live.jsonl"):
            moves.append((src, NEW / src.name, "live"))
        else:
            moves.append((src, NEW / f"{src.stem}.track.jsonl", "track"))
    viz = OLD / "flight_viz.html"
    if viz.is_file():
        moves.append((viz, NEW / viz.name, "viz"))
    return moves


def translate_train(src: Path) -> tuple[str, int, int]:
    """The rows in the unified train schema; returns (text, rows, rows given a range)."""
    out, added = [], 0
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get("pos") and "range" not in row:
            row["range"] = round(math.sqrt(sum(float(v) ** 2 for v in row["pos"])), 2)
            added += 1
        out.append(json.dumps(row))
    return "\n".join(out) + ("\n" if out else ""), len(rows), added


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--copy", action="store_true", help="keep the originals in outputs/")
    p.add_argument("--force", action="store_true", help="overwrite existing files in recordings/")
    a = p.parse_args()

    moves = plan()
    if not moves:
        print(f"nothing to migrate in {OLD}")
        return 0
    NEW.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for src, dst, kind in moves:
        rel = f"{src.relative_to(DD_ROOT)} -> {dst.relative_to(DD_ROOT)}"
        if src.stat().st_size == 0:
            print(f"  EMPTY  {src.relative_to(DD_ROOT)}: 0 bytes, no rows were logged; left in place")
            skipped += 1
            continue
        if dst.exists() and not a.force:
            print(f"  EXISTS {rel}: destination already there (use --force); left in place")
            skipped += 1
            continue
        if kind == "train":
            text, n, added = translate_train(src)
            print(f"  {'would ' if a.dry_run else ''}{'copy' if a.copy else 'move'} {rel}  ({n} rows, range added to {added})")
            if not a.dry_run:
                dst.write_text(text)
                if not a.copy:
                    src.unlink()
        else:
            print(f"  {'would ' if a.dry_run else ''}{'copy' if a.copy else 'move'} {rel}")
            if not a.dry_run:
                (shutil.copy2 if a.copy else shutil.move)(src, dst)
        done += 1
    if not a.dry_run and not a.copy:
        for d in (OLD / "train", OLD / "tracks"):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    print(f"{'would migrate' if a.dry_run else 'migrated'} {done}, left {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

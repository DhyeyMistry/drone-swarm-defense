#!/usr/bin/env python3
"""Benchmark the defence in simulation: many attack rounds, knockdowns / breaches / timing.

Starts hil_gui.py (the attacker simulator: attackers fly from tag 10's side at the table,
optionally weaving) and defend.py --sim (our real brain + controller driving kinematic
defenders) on spare ports, then plays rounds through hil_gui's HTTP API:

    reset -> defenders settle into their hold formation -> "Attack!" -> watch until every
    attacker is down (knocked within 0.35 m), one reaches the table (breach) or timeout.

Writes one CSV row per round and, if matplotlib is available, a summary chart.

  python sim_benchmark.py                                         # 3 speeds x 2 weaves x 5 rounds
  python sim_benchmark.py --speeds 0.3,0.8 --weaves 0 --rounds 3 --defend-args "--max_lead_s 1.0"
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def venv_python() -> str:
    p = REPO / ".venv" / "bin" / "python"
    return str(p) if p.exists() else sys.executable


class Gui:
    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"

    def state(self) -> dict:
        with urllib.request.urlopen(self.base + "/state", timeout=2) as r:
            return json.load(r)

    def cmd(self, **c) -> None:
        req = urllib.request.Request(self.base + "/cmd", data=json.dumps(c).encode(), method="POST")
        urllib.request.urlopen(req, timeout=2).read()


def settle(gui: Gui, timeout: float) -> bool:
    """Wait until defend.py is live and every defender holds its formation slot."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = gui.state().get("status")
        if st and st["defenders"] and all(d["mode"] == "hold" and d["pos"] and d["target"] and
                                          sum((a - b) ** 2 for a, b in zip(d["pos"], d["target"])) ** 0.5 < 0.15
                                          for d in st["defenders"]):
            return True
        time.sleep(0.2)
    return False


def play_round(gui: Gui, speed: float, weave: float, timeout: float) -> dict:
    gui.cmd(op="attack", on=False)
    gui.cmd(op="reset")
    for a in gui.state()["attackers"]:          # attackers wait "down" while defenders settle
        gui.cmd(op="toggle", name=a["name"])
    settled = settle(gui, 25.0)
    for a in gui.state()["attackers"]:
        gui.cmd(op="toggle", name=a["name"])
    gui.cmd(op="speed", v=speed)
    gui.cmd(op="weave", v=weave)
    n_att = len(gui.state()["attackers"])
    gui.cmd(op="attack", on=True)
    t0 = time.time()
    first_down = all_down = None
    while time.time() - t0 < timeout:
        s = gui.state()
        down = sum(1 for a in s["attackers"] if not a["alive"])
        t = time.time() - t0
        if down and first_down is None:
            first_down = t
        if down == n_att:
            all_down = t
            break
        if s["counts"]["breaches"]:
            break
        time.sleep(0.05)
    s = gui.state()
    gui.cmd(op="attack", on=False)
    return {"speed": speed, "weave": weave, "attackers": n_att, "settled": settled,
            "knocked_down": sum(1 for a in s["attackers"] if not a["alive"]),
            "breaches": s["counts"]["breaches"],
            "t_first_down_s": None if first_down is None else round(first_down, 2),
            "t_all_down_s": None if all_down is None else round(all_down, 2)}


def plot(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: CSV only", flush=True)
        return
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": surf, "axes.facecolor": surf})
    combos = sorted({(r["weave"], r["speed"]) for r in rows})
    weaves = sorted({w for w, _ in combos})
    speeds = sorted({s for _, s in combos})
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.8 / max(1, len(weaves))
    for i, w in enumerate(weaves):
        rate, tdown = [], []
        for sp in speeds:
            rr = [r for r in rows if r["speed"] == sp and r["weave"] == w]
            att = sum(r["attackers"] for r in rr) or 1
            rate.append(100.0 * sum(r["knocked_down"] for r in rr) / att)
            ts = [r["t_all_down_s"] for r in rr if r["t_all_down_s"] is not None]
            tdown.append(sum(ts) / len(ts) if ts else 0.0)
        xs = [j + (i - (len(weaves) - 1) / 2) * width for j in range(len(speeds))]
        a1.bar(xs, rate, width * 0.92, color=colors[i % 3], label=f"weave {w}")
        a2.bar(xs, tdown, width * 0.92, color=colors[i % 3], label=f"weave {w}")
        for x, v in zip(xs, rate):
            a1.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=ink)
        for x, v in zip(xs, tdown):
            a2.text(x, v + 0.15, f"{v:.1f}", ha="center", fontsize=9, color=ink)
    for ax in (a1, a2):
        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([f"{s} m/s" for s in speeds])
        ax.set_xlabel("attacker speed")
        ax.grid(axis="y", color=grid)
        ax.set_axisbelow(True)
    a1.set_ylim(0, 112)
    a1.set_ylabel("attackers knocked down (%)")
    a1.set_title("Knockdown rate", color=ink, loc="left")
    a2.set_ylabel("seconds")
    a2.set_title("Time until every attacker is down", color=ink, loc="left")
    a2.legend(loc="upper right", frameon=False)
    n = len(rows)
    br = sum(r["breaches"] for r in rows)
    fig.suptitle(f"Simulation: 3 defenders vs 2 attackers, {n} rounds, {br} breaches", x=0.01, ha="left",
                 color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeds", default="0.3,0.5,0.8", help="attacker speeds, m/s")
    ap.add_argument("--weaves", default="0,0.6", help="attacker weave amounts (0 = straight)")
    ap.add_argument("--rounds", type=int, default=5, help="rounds per speed/weave combination")
    ap.add_argument("--defenders", default="TELLO-2,TELLO-3,TELLO-5")
    ap.add_argument("--attackers", default="TELLO-8,TELLO-9")
    ap.add_argument("--timeout_s", type=float, default=30.0, help="a round ends after this")
    ap.add_argument("--defend-args", default="", help="extra defend.py flags, e.g. \"--formation home\"")
    ap.add_argument("--csv", default=str(REPO / "docs" / "results" / "sim_benchmark.csv"))
    ap.add_argument("--plot", default=str(REPO / "docs" / "media" / "sim_benchmark.png"))
    ap.add_argument("--gui_port", type=int, default=8796)
    ap.add_argument("--python", default=None,
                    help="interpreter with the repo's packages for hil_gui.py / defend.py (default: <repo>/.venv)")
    args = ap.parse_args()
    py = args.python or venv_python()
    feed, status = 9186, 9187
    gui_p = subprocess.Popen([py, "-u", str(HERE / "hil_gui.py"), "--attackers", args.attackers,
                              "--port", str(args.gui_port), "--feed", f"127.0.0.1:{feed}", "--status_port", str(status)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    gui = Gui(args.gui_port)
    t0 = time.time()
    while True:                                  # wait for the GUI's web server
        try:
            gui.state()
            break
        except OSError:
            if gui_p.poll() is not None or time.time() - t0 > 20:
                raise SystemExit(f"hil_gui.py did not start (python: {py}); try --python <venv python>")
            time.sleep(0.3)
    def_p = subprocess.Popen([py, "-u", str(HERE / "defend.py"), "--sim", "--defenders", args.defenders,
                              "--attackers", "auto", "--feed", "gui", "--feed_port", str(feed),
                              "--status", f"127.0.0.1:{status}", "--seconds", "100000", "--log_every_s", "0"]
                             + shlex.split(args.defend_args), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    rows = []
    try:
        for sp in [float(v) for v in args.speeds.split(",")]:
            for w in [float(v) for v in args.weaves.split(",")]:
                for k in range(args.rounds):
                    r = play_round(gui, sp, w, args.timeout_s)
                    r["round"] = k + 1
                    rows.append(r)
                    print(f"speed {sp} weave {w} round {k + 1}: {r['knocked_down']}/{r['attackers']} down, "
                          f"{r['breaches']} breach(es), all down at {r['t_all_down_s']} s", flush=True)
    finally:
        for p in (def_p, gui_p):
            p.terminate()
        for p in (def_p, gui_p):
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv}", flush=True)
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plot(rows, Path(args.plot))
    tot = sum(r["attackers"] for r in rows)
    print(f"TOTAL: {sum(r['knocked_down'] for r in rows)}/{tot} attackers knocked down, "
          f"{sum(r['breaches'] for r in rows)} breaches in {len(rows)} rounds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

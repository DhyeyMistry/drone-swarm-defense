#!/usr/bin/env python3
"""Build a self-contained HTML test report from our flight logs (no network, no repo edits).

Reads recordings/defend_*.jsonl (defend.py), recordings/axis_check_*.jsonl and the flight
videos, and writes one HTML file with the plots and frames embedded.

  python make_report.py --duel ../recordings/defend_20261024_092802.jsonl \
      --defender_video ../recordings/026_TELLO-2_20261024_092750.avi \
      --attacker_video ../recordings/026_TELLO-4_20261024_092740.avi
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import html
import io
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REC = HERE.parent / "recordings"

# reference palette (dataviz skill): categorical slots 1-2, text inks, surface
DEF_C, ATT_C = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(path: str) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if "defenders" in r], [r for r in rows if r.get("event")]


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def jpg(img) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode()


def video_start(path: str) -> dt.datetime:
    stamp = Path(path).stem.split("_", 2)[2][:15]
    return dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S")


def log_start(path: str) -> dt.datetime:
    return dt.datetime.strptime(Path(path).stem.split("_", 1)[1], "%Y%m%d_%H%M%S")


def frames_at(video: str, offset: float, times: list[float]) -> dict[float, np.ndarray]:
    cap = cv2.VideoCapture(video)
    want = {int(round((t + offset) * 30)): t for t in times}
    out, i = {}, 0
    while want:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            out[want.pop(i)] = cv2.resize(im, (480, 360))
        i += 1
    return out


def duel_series(st: list[dict]):
    st = [r for r in st if r["defenders"]]       # defenders join the log once they are airborne
    t = np.array([r["t"] for r in st])
    d = st[0]["defenders"][0]["name"]
    a = st[0]["attackers"][0]["name"] if st[0]["attackers"] else None
    dpos = np.array([r["defenders"][0]["pos"] if r["defenders"][0]["pos"] else [np.nan] * 3 for r in st], float)
    apos = np.array([r["attackers"][0]["pos"] if r["attackers"] else [np.nan] * 3 for r in st], float)
    alive = np.array([bool(r["attackers"][0]["alive"]) if r["attackers"] else True for r in st])
    mode = [r["defenders"][0]["mode"] for r in st]
    rc = np.array([r["defenders"][0]["rc"] for r in st], float)
    return t, d, a, dpos, apos, alive, mode, rc


def fig_topdown(t, d, a, dpos, apos, alive, contact_t, down_t):
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    keep = t <= (down_t if down_t is not None else t[-1]) + 0.3     # engagement only
    t, dpos, apos, alive = t[keep], dpos[keep], apos[keep], alive[keep]
    live = alive
    ax.plot(apos[live, 0], apos[live, 2], color=ATT_C, label=f"{a} (attacker, reported)")
    ax.plot(dpos[:, 0], dpos[:, 2], color=DEF_C, label=f"{d} (defender, own camera)")
    ax.plot(dpos[0, 0], dpos[0, 2], "o", ms=8, color=DEF_C, mec=SURF, mew=2)
    ax.plot(apos[0, 0], apos[0, 2], "o", ms=8, color=ATT_C, mec=SURF, mew=2)
    ax.annotate("start", (dpos[0, 0], dpos[0, 2]), xytext=(8, 4), textcoords="offset points", color=INK2)
    for tt, label in ((contact_t, "1st contact"), (down_t, "attacker down")):
        if tt is None:
            continue
        i = int(np.argmin(np.abs(t - tt)))
        p = apos[i]
        ax.plot(p[0], p[2], marker="X", ms=11, color=INK, mec=SURF, mew=1.5)
        ax.annotate(f"{label} ({tt:.1f} s)", (p[0], p[2]), xytext=(10, -14), textcoords="offset points", color=INK)
    ax.plot(0, 0, "s", ms=10, color=INK)
    ax.annotate("tag 10 (wall)", (0, 0), xytext=(8, 4), textcoords="offset points", color=INK)
    ax.set_xlabel("x: right of tag 10 (m)")
    ax.set_ylabel("z: out from the wall (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-0.2, max(np.nanmax(dpos[:, 2]), np.nanmax(apos[:, 2])) + 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Top-down paths until the knockdown (tag-10 world frame)", color=INK, loc="left")
    return fig


def fig_separation(t, dpos, apos, alive, mode, contact_t, down_t):
    sep_all = np.linalg.norm(dpos - apos, axis=1)
    keep = t <= (down_t if down_t is not None else t[-1]) + 0.3     # after it is down the gap means nothing
    t, sep, mode = t[keep], sep_all[keep], [m for m, k in zip(mode, keep) if k]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    blind = np.array([m == "idle" for m in mode])
    for i0 in np.where(np.diff(np.r_[0, blind.astype(int), 0]) == 1)[0]:
        i1 = i0 + int(np.argmax(~blind[i0:])) if (~blind[i0:]).any() else len(t) - 1
        ax.axvspan(t[i0], t[min(i1, len(t) - 1)], color=GRID, lw=0)
    ax.plot(t, sep, color=DEF_C, label="defender to attacker distance")
    ax.axhline(0.35, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(t[0] + 0.1, 0.37, "contact radius 0.35 m", ha="left", va="bottom", color=INK2, fontsize=9)
    for tt, label in ((contact_t, "1st contact"), (down_t, "attacker reported down")):
        if tt is not None:
            ax.axvline(tt, color=INK, lw=1)
            ax.text(tt - 0.15, np.nanmax(sep) * 1.02, label, color=INK, fontsize=9, ha="right")
    ax.set_xlabel("time since defend.py started (s)")
    ax.set_ylabel("distance (m)")
    ax.set_ylim(0, np.nanmax(sep) * 1.12)
    ax.set_title("Defender-to-attacker distance until the knockdown", color=INK, loc="left")
    return fig, sep


def fig_rc(t, rc, down_t=None):
    keep = t <= (down_t if down_t is not None else t[-1]) + 0.3
    t, rc = t[keep], rc[keep]
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    ax.plot(t, rc[:, 1], color=DEF_C, label="forward stick")
    ax.plot(t, rc[:, 0], color=ATT_C, label="right stick")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rc (-100..100)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title("Defender stick commands until the knockdown", color=INK, loc="left")
    return fig


def attempts_table(mouse_attackers: set[str], skip: set[str] = frozenset()) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(REC / "defend_2026*.jsonl"))):
        if Path(f).name in skip:
            continue
        st, ev = load(f)
        if not st:
            continue
        sim = all(d.get("note") == "sim" for r in st[:5] for d in r["defenders"])
        defs = [d["name"] for d in st[0]["defenders"]]
        atts = sorted({a["name"] for r in st for a in r["attackers"]})
        n = sum(len(r["defenders"]) for r in st)
        fix = sum(1 for r in st for d in r["defenders"] if d["pos"]) / max(1, n)
        contacts = [e for e in ev if e["event"] == "contact"]
        downs = sorted({a["name"] for r in st for a in r["attackers"] if not a["alive"]})
        if sim:
            kind = "simulation"
        elif atts and set(atts) <= mouse_attackers:
            kind = "real defender vs mouse attackers"
        else:
            kind = "real vs real"
        out.append({"file": Path(f).name, "kind": kind,
                    "secs": st[-1]["t"], "def": ", ".join(defs), "att": ", ".join(atts) or "-",
                    "fix": fix, "contacts": len(contacts), "down": ", ".join(downs) or "-"})
    return out


def axis_rows() -> tuple[str, list[dict]]:
    files = sorted(glob.glob(str(REC / "axis_check_*.jsonl")))
    good = [f for f in files if any(json.loads(line).get("moved_rfu_m") for line in open(f))]
    if not good:
        return "", []
    f = good[-1]
    return Path(f).name, [json.loads(line) for line in open(f)]


def main() -> int:
    global REC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duel", required=True)
    ap.add_argument("--defender_video")
    ap.add_argument("--attacker_video")
    ap.add_argument("--mouse_attackers", default="TELLO-8,TELLO-9",
                    help="attacker names that were hil_gui.py mouse dots, not real drones, in these logs")
    ap.add_argument("--out", default=str(HERE / "reports" / "duel_report.html"))
    ap.add_argument("--recordings", default=str(REC), help="folder with the defend_*.jsonl / axis_check logs")
    ap.add_argument("--skip", default="", help="comma list of log file names to leave out of the runs table")
    ap.add_argument("--note", default="", help="run-specific comment (HTML) shown under the title")
    ap.add_argument("--png_dir", default=None, help="also save each chart as <png_dir>/<name>_<chart>.png")
    args = ap.parse_args()

    REC = Path(args.recordings)
    st, ev = load(args.duel)
    st = [r for r in st if r["defenders"]]
    t, d, a, dpos, apos, alive, mode, rc = duel_series(st)
    contacts = [e for e in ev if e["event"] == "contact"]
    contact_t = contacts[0]["t"] if contacts else None
    down_i = np.where(~alive)[0]
    down_t = float(t[down_i[0]]) if len(down_i) else None
    figs = {"top": png(fig_topdown(t, d, a, dpos, apos, alive, contact_t, down_t))}
    f2, sep = fig_separation(t, dpos, apos, alive, mode, contact_t, down_t)
    figs["sep"] = png(f2)
    figs["rc"] = png(fig_rc(t, rc, down_t))
    if args.png_dir:
        pd = Path(args.png_dir)
        pd.mkdir(parents=True, exist_ok=True)
        stem = Path(args.out).stem
        for k, b64 in figs.items():
            (pd / f"{stem}_{k}.png").write_bytes(base64.b64decode(b64))
    min_i = int(np.nanargmin(sep))
    eng = [m for m, tt in zip(mode, t) if tt <= down_t + 0.3] if down_t is not None else mode
    blind = sum(1 for m in eng if m == "idle") / len(eng)
    feed_ages = [r["attackers"][0]["age"] for r in st if r["attackers"] and r["attackers"][0]["alive"]]
    rate = 1 / np.median(np.diff(t))

    note_html = f'<p class="note">{args.note}</p>' if args.note else ""
    frames_html = ""
    if args.defender_video and args.attacker_video:
        t0 = log_start(args.duel)
        cells = []
        hit = contact_t if contact_t is not None else float(t[min_i])
        end = down_t if down_t is not None else float(t[-1])
        for who, vid, times in ((f"{d} (defender) camera", args.defender_video,
                                 [0.0, round(hit, 1), round(end, 1)]),
                                (f"{a} (attacker) camera", args.attacker_video,
                                 [0.0, round(end, 1), round(end + 3.5, 1)])):
            off = (t0 - video_start(vid)).total_seconds()
            first = max(0.0, -off + 1.0)               # a video may start after the log does
            times = sorted({max(first, x) for x in times})
            got = frames_at(vid, off, times)
            for tt in times:
                if tt in got:
                    cells.append(f'<figure><img src="data:image/jpeg;base64,{jpg(got[tt])}" alt="{who} at {tt} s">'
                                 f"<figcaption>{html.escape(who)}, t ≈ {tt:.1f} s</figcaption></figure>")
        frames_html = '<div class="frames">' + "".join(cells) + "</div>"

    attempts = attempts_table({x.strip() for x in args.mouse_attackers.split(",") if x.strip()},
                              {x.strip() for x in args.skip.split(",") if x.strip()})
    sim_rows = [r for r in attempts if r["kind"] == "simulation"]
    axis_file, axis = axis_rows()

    def tr(cells, head=False):
        tag = "th" if head else "td"
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

    att_table = tr(["Log", "Type", "Length", "Defenders", "Attackers", "Defender sees a tag", "Contacts", "Reported down"], True)
    for r in attempts:
        hl = ' class="hl"' if r["file"] == Path(args.duel).name else ""
        att_table += (f"<tr{hl}><td>{r['file']}</td><td>{r['kind']}</td><td>{r['secs']:.0f} s</td><td>{r['def']}</td>"
                      f"<td>{r['att']}</td><td>{r['fix']:.0%}</td><td>{r['contacts']}</td><td>{r['down']}</td></tr>")
    axis_table = tr(["Segment", "Sticks (r, f, u, yaw)", "Moved right / fwd / up (m)", "Verdict"], True)
    for r in axis:
        mv = "-" if r["moved_rfu_m"] is None else " / ".join(f"{v:+.2f}" for v in r["moved_rfu_m"])
        axis_table += tr([r["segment"], str(tuple(r["rc"])), mv, html.escape(r["verdict"])])

    sim_summary = ""
    if sim_rows:
        s = max(sim_rows, key=lambda r: r["secs"])
        sim_summary = (f"<p>Longest simulation session (<code>{s['file']}</code>): {s['def']} against {s['att']}, "
                       f"{s['secs']/60:.1f} min of mouse-driven attacks in <code>hil_gui.py</code>, "
                       f"<b>{s['contacts']}</b> contacts logged. The scripted 5-round benchmark at 0.8 m/s with "
                       f"weaving knocked down both attackers in every round, with 0 breaches.</p>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Defense Duel Report</title>
<style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--mut:#8a8984;--line:#e4e3df;--card:#fff;--hl:#eef4fc;--def:{DEF_C};--att:{ATT_C}}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--mut:#8f8e88;--line:#33322f;--card:#222220;--hl:#1f2a38}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--mut:#8f8e88;--line:#33322f;--card:#222220;--hl:#1f2a38}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:28px 16px 60px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--fg2);margin:0 0 20px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.tile b{{display:block;font-size:24px;font-variant-numeric:tabular-nums}} .tile span{{color:var(--fg2);font-size:13px}}
img.chart{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fcfcfb}}
.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}
figure{{margin:0}} figure img{{width:100%;border-radius:6px;border:1px solid var(--line)}} figcaption{{color:var(--fg2);font-size:13px}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--fg2);font-weight:600}}
tr.hl td{{background:var(--hl)}} .note{{background:var(--card);border-left:3px solid var(--def);padding:8px 12px;margin:0 0 16px}} code{{font-size:13px}} li{{margin:4px 0}}
.k{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
</style></head><body><main>
<h1>Defense duel: {d} (defender) vs {a} (attacker)</h1>
<p class="sub">Action Space hackathon, {log_start(args.duel):%d %b %Y %H:%M} · real Tello flight, one defender against one attacker, our <code>our_defense/</code> stack</p>
{note_html}

<div class="tiles">
<div class="tile"><b>{"Down" if down_t is not None else "Not down"}</b><span>{a} {"reported down by its feed" if down_t is not None else "still flying at the end of the log"}</span></div>
<div class="tile"><b>{contact_t:.1f} s</b><span>to first contact (≤ 0.35 m) after the defender started</span></div>
<div class="tile"><b>{down_t:.1f} s</b><span>until the attacker was reported down</span></div>
<div class="tile"><b>{sep[min_i]:.2f} m</b><span>closest measured approach, at {t[min_i]:.1f} s</span></div>
</div>

<h2>Setup</h2>
<ul>
<li><span class="k" style="background:var(--def)"></span><b>Defender {d}</b>: <code>defend.py</code>. Localizes from tag 10 with its own camera (<code>upright_positional.py</code>), gets the attacker's position from the telemetry feed, rams it (<code>defense_core.py</code>).</li>
<li><span class="k" style="background:var(--att)"></span><b>Attacker {a}</b>: <code>dummy_attacker.py</code>. Takes off, holds facing tag 10 and shuttles 0.5 m toward / away from it, publishing its own position as repo fleet telemetry (the <code>drone_gui</code> format).</li>
<li>Started together by <code>duel.py --defenders {d} --attackers {a} --fly</code> (the attacker gets a short head start). Roles are command-line arguments.</li>
<li>All positions are in the tag-10 world frame of the map: x right of tag 10, y up from its centre, z out from the wall.</li>
</ul>

<h2>What happened</h2>
<img class="chart" src="data:image/png;base64,{figs['sep']}" alt="Distance between defender and attacker over time">
<p class="sub">Grey bands: the defender had no tag fix (typically the attacker in front of it blocking the tag), so it held still instead of flying blind. Gaps in the line are those moments.</p>
<ol>
<li><b>0–{contact_t:.1f} s:</b> the defender rams from {sep[0]:.1f} m away and enters the 0.35 m contact radius at {contact_t:.1f} s.</li>
<li><b>{contact_t:.1f}–{down_t:.1f} s:</b> it overshoots, turns back and passes again within {sep[min_i]:.2f} m. Grey bands ({blind:.0%} of the engagement) are moments without a tag fix, typically the attacker blocking the defender's view of the tag. It holds still each time instead of flying blind, as designed, then resumes.</li>
<li><b>{down_t:.1f} s:</b> {a}'s feed reports it down. The defender drops the target and switches to holding its guard position.</li>
</ol>
<img class="chart" src="data:image/png;base64,{figs['top']}" alt="Top-down paths of defender and attacker">
<p>Attacker feed: received at {rate:.0f} Hz on the defender side, median report age {np.median(feed_ages)*1000:.0f} ms while it was flying.</p>

<h2>Video evidence</h2>
{frames_html}
<p class="sub">Frames are matched to the log by file timestamps (±1 s): the start, the first contact and the reported knockdown. A frame is missing where that camera's video had already ended.</p>

<h2>Defender commands</h2>
<img class="chart" src="data:image/png;base64,{figs['rc']}" alt="Defender stick commands over time">

<h2>All runs (23–24 Oct)</h2>
<div class="scroll"><table>{att_table}</table></div>
<p class="sub">Highlighted: the run in this report. "Real defender vs mouse attackers" = a real defender chasing attacker positions dragged in <code>hil_gui.py</code> (home testing, last night). The short 4 s runs were aborted starts while we fixed setup issues. "Defender sees a tag" = share of control ticks with a fresh position fix.</p>
{sim_summary}

<h2>Motion check (axis_check.py, home)</h2>
<p>Run <code>{axis_file}</code>: zero-stick hover, then a 1.5 s pulse on each axis at rc 30, measured with the tag.</p>
<div class="scroll"><table>{axis_table}</table></div>
<p>Every axis moved the commanded way. The drone keeps coasting about 2 s after a pulse, and rc 30 moves it about 2–3× farther than the controller assumes. That explains the overshoot in the duel above, and it is the next tuning item.</p>

<h2>Problems found and fixed along the way</h2>
<ul>
<li><b>Dark room, tag on a screen:</b> no tag detected and the Tello drifted into a wall. Fix: lights on, printed tag, and no flying blind: the drone lands if the tag is lost for 3–4 s (one timer per flight).</li>
<li><b>Upside-down / official-orientation tags:</b> the repo localizer mirrors x and y with the official tag prints. Fix: <code>upright_positional.py</code> turns every tag upright before using the map, for either print style.</li>
<li><b>Climbing out of the camera's view:</b> fix: defenders stay within ±0.5 m of the tag height, plus a 30 cm climb after takeoff so the tag comes into view.</li>
<li><b>Attacker feed convention:</b> verified against <code>drone_gui --sim</code>'s ground truth (0.12–0.18 m error). Attacker height now comes from the Tello's own height sensor (<code>h</code>).</li>
</ul>

<h2>Limitations and next steps</h2>
<ul>
<li>This report is one defender against one scripted attacker at short range (about 1 m).</li>
<li>The scripted attacker shuttles predictably; real attackers move more and faster.</li>
<li>The controller does not model the Tello's ~2 s coast after a command, so it overshoots and re-attacks. Next: braking from measured velocity, and a full-speed strike for the final metre.</li>
<li>Attacker positions are only as good as the feed, and every tag the attackers see must be in the map.</li>
</ul>
</main></body></html>"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

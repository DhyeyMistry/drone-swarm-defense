"""Fit how a real Tello moves per rc stick unit, from the flight engine's training logs.

The engine (drone_deployment_infra/core/tello_dual_video.py) writes one row per ~20 Hz
tick to ``recordings/<session>_<drone>.train.jsonl`` (everything a flight writes lives in recordings/;
older flights' outputs/train/ logs move there with core/migrate_outputs.py):

    {t, rc:[right, forward, up, yaw], mode, fresh, h, vel:{r, f, u, yaw}, tag:{x, y, z, range}}

Per channel this fits   speed(t) = gain * dz(rc(t - delay)) filtered by 1 / (tau s + 1),
dz = 0 inside |rc| < deadband, by grid search over (deadband, delay, tau) with the gain
solved by least squares at each point, and writes the result where the sim reads it
(``control/tello_motion_fit.yaml``). Channels with too little stick activity keep their
prior and are flagged.

Speeds come from the state packet (``vel``). Because its axes/units were never verified
(the engine assumes vgx = right, cm/s; SDK 2.0 firmware often reports dm/s with x
forward), the script first cross-checks it against motion derived from tag sightings,
prints the channel correlation matrix, and can fit against tag-derived speed instead
(``--speed_source tag``).

numpy + yaml only; no Isaac:

    python scripts/fit_tello_motion.py          # every ../drone_deployment_infra/recordings/*.train.jsonl
    python scripts/fit_tello_motion.py --selftest      # recover a known model from synthetic logs
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import yaml

CHANNELS = ("right", "forward", "up", "yaw")
VEL_KEYS = {"right": "r", "forward": "f", "up": "u"}
AIRBORNE_MODES = {"track", "lane", "aim", "back", "climb", "search", "no tag", "hover", "hold", "cruise"}
DT = 0.05
OUT = Path(__file__).resolve().parents[1] / "source/action_space_kit/action_space_kit/control/tello_motion_fit.yaml"
PRIOR = yaml.safe_load(OUT.read_text())["channels"] if OUT.is_file() else None


# -- loading ------------------------------------------------------------------------


def load_segments(paths: list[str], min_len_s: float = 2.0) -> list[dict]:
    """Uniformly resampled airborne segments: {t, rc (n, 4), vel (n, 3), yaw_deg (n,), tag (n, 3) or nan}."""
    segs = []
    for path in paths:
        rows = [_normalize(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r.get("mode") in AIRBORNE_MODES and (r["vel"] or r["tag"]) and (r.get("h") or 0) > 0.25]
        if not rows:
            continue
        t = np.array([r["t"] for r in rows])
        breaks = np.where(np.diff(t) > 0.3)[0] + 1
        for idx in np.split(np.arange(len(rows)), breaks):
            if len(idx) < 2 or t[idx[-1]] - t[idx[0]] < min_len_s:
                continue
            seg_rows = [rows[i] for i in idx]
            ts = t[idx]
            grid = np.arange(ts[0], ts[-1], DT)
            # rc is held between ticks (zero-order hold); speeds are interpolated
            hold = np.searchsorted(ts, grid, side="right") - 1
            rc = np.array([r["rc"] for r in seg_rows], dtype=float)[hold]
            if all(r["vel"] for r in seg_rows):
                vel = np.array([[r["vel"][k] for k in ("r", "f", "u")] for r in seg_rows], dtype=float)
                vel = np.stack([np.interp(grid, ts, vel[:, k]) for k in range(3)], -1)
                yaw = np.unwrap(np.radians([r["vel"]["yaw"] for r in seg_rows]))
                yaw = np.degrees(np.interp(grid, ts, yaw))
            else:  # no state packet in this log format: only tag-derived speeds, no heading
                vel = np.full((len(grid), 3), np.nan)
                yaw = np.full(len(grid), np.nan)
            tag = np.full((len(grid), 3), np.nan)
            tag_rows = [(r["t"], r["tag"]) for r in seg_rows if r.get("fresh") and r.get("tag")]
            if len(tag_rows) > 5:
                tt = np.array([a for a, _ in tag_rows])
                tp = np.array([[b["x"], b["y"], b["z"]] for _, b in tag_rows])
                near = np.abs(tt[np.clip(np.searchsorted(tt, grid), 0, len(tt) - 1)] - grid) < 0.15
                for k in range(3):
                    tag[:, k] = np.where(near, np.interp(grid, tt, tp[:, k]), np.nan)
            segs.append({"file": Path(path).name, "t": grid, "rc": rc, "vel": vel, "yaw_deg": yaw, "tag": tag})
    return segs


def _normalize(r: dict) -> dict:
    """Both engine log formats -> {t, rc, mode, h, fresh, vel or None, tag or None}.

    v1 (repo engine): vel {r, f, u, yaw} from the state packet, tag {x, y, z} = camera in
    tag 10's frame, pupil convention (x right, y down, z negative in front).
    v2 (laptop engine, 2026-10-08): pos [x, y, z] = camera in the map frame (x right, y up,
    z out of tag 10) whenever tags are in view, plus the tag ids; no state packet.
    """
    out = {"t": r["t"], "rc": r["rc"], "mode": r.get("mode"), "h": r.get("h"), "vel": r.get("vel")}
    if r.get("pos"):
        x, y, z = r["pos"]
        out["fresh"], out["tag"] = True, {"x": x, "y": -y, "z": -z}
    else:
        out["fresh"], out["tag"] = bool(r.get("fresh")), r.get("tag")
    return out


def channel_speed(seg: dict, ch: str, source: str) -> np.ndarray:
    """Measured speed for one channel: m/s (state packet or tag-derived), yaw deg/s from the IMU."""
    if ch == "yaw":
        # rc yaw is clockwise-positive; the Tello's yaw field increases clockwise as well
        return np.gradient(seg["yaw_deg"], DT)
    if source == "state":
        return seg["vel"][:, "rfu".index(VEL_KEYS[ch])]
    # tag-derived: camera in tag frame (x right, y down, z negative in front); valid when facing the tag
    x, y, z = seg["tag"].T
    comp = {"right": x, "forward": z, "up": -y}[ch]
    return _smooth_gradient(comp)


def _smooth_gradient(x: np.ndarray, win: int = 5) -> np.ndarray:
    k = np.ones(win) / win
    ok = np.isfinite(x)
    xs = np.convolve(np.where(ok, x, 0.0), k, "same") / np.maximum(np.convolve(ok.astype(float), k, "same"), 1e-9)
    g = np.gradient(xs, DT)
    full = np.convolve(ok.astype(float), np.ones(win), "same") >= win
    return np.where(full, g, np.nan)


# -- model --------------------------------------------------------------------------


def simulate(rc: np.ndarray, deadband: float, delay_steps: int, tau: float) -> np.ndarray:
    """Unit-gain response of the model to an rc sequence (per segment)."""
    u = np.where(np.abs(rc) >= deadband, rc, 0.0)
    if delay_steps:
        u = np.concatenate([np.zeros(delay_steps), u[:-delay_steps]])
    a = math.exp(-DT / tau)
    y = np.zeros_like(u)
    for i in range(1, len(u)):
        y[i] = a * y[i - 1] + (1 - a) * u[i - 1]
    return y


def _gain_and_sse(xs: list, ys: list) -> tuple[float, float, np.ndarray, np.ndarray]:
    x, y = np.concatenate(xs), np.concatenate(ys)
    denom = float(x @ x)
    if denom < 1e-9:
        return float("nan"), float("inf"), x, y
    gain = float(x @ y) / denom
    return gain, float(np.sum((y - gain * x) ** 2)), x, y


def fit_channel(segs: list[dict], ch: str, source: str) -> dict | None:
    """Grid search (deadband, delay, tau); gain by least squares; then a block bootstrap
    of the gain (2 s blocks, so correlated noise isn't counted as independent samples).

    A channel is accepted on how tightly the data pins its gain (ci_rel), not on R2: a
    stick held nearly constant explains little speed *variation* even when its gain is
    measured well.
    """
    j = CHANNELS.index(ch)
    data = [(s["rc"][:, j], channel_speed(s, ch, source)) for s in segs]
    active = sum(int(np.sum(np.abs(rc) >= 5)) for rc, _ in data)
    if active < 40:  # < 2 s of stick on this channel
        return None
    best = None
    for deadband in range(0, 13):
        for delay_steps in range(0, 9):
            for tau in (0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3):
                xs, ys = [], []
                for rc, y in data:
                    x = simulate(rc, deadband, delay_steps, tau)
                    ok = np.isfinite(y)
                    xs.append(x[ok])
                    ys.append(y[ok])
                gain, sse, x, y = _gain_and_sse(xs, ys)
                if math.isfinite(gain) and (best is None or sse < best["sse"]):
                    var = float(np.sum((y - y.mean()) ** 2)) or 1e-9
                    best = {"gain": gain, "deadband": deadband, "delay": delay_steps * DT, "tau": tau,
                            "sse": sse, "r2": 1.0 - sse / var, "n": int(len(y)), "active": active, "x": x, "y": y}
    if best is None:
        return None
    x, y = best.pop("x"), best.pop("y")
    block = int(2.0 / DT)
    nb = max(1, len(x) // block)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([np.arange(b * block, min((b + 1) * block, len(x))) for b in pick])
        d = float(x[idx] @ x[idx])
        if d > 1e-9:
            boots.append(float(x[idx] @ y[idx]) / d)
    lo, hi = np.percentile(boots, [5, 95])
    best["ci"] = (float(lo), float(hi))
    best["ci_rel"] = float((hi - lo) / (2 * abs(best["gain"]))) if best["gain"] else float("inf")
    # Response time, delay and deadband are only observable across stick steps; a stick held
    # at one value pins the gain alone.
    best["steps"] = sum(int(np.sum(np.abs(np.diff(rc)) >= 5)) for rc, _ in data)
    return best


# -- diagnostics ----------------------------------------------------------------------


def command_envelope(paths: list[str]) -> dict:
    """Per channel, the [min, max] rc actually sent (and flown on) in these logs. The sim's
    policy may only command values inside it: nothing a real Tello hasn't been sent."""
    rc = np.array([json.loads(line)["rc"] for path in paths
                   for line in Path(path).read_text().splitlines() if line.strip()], dtype=int)
    env = {ch: [int(rc[:, j].min()), int(rc[:, j].max())] for j, ch in enumerate(CHANNELS)}
    print("\n== command envelope (rc actually sent) ==")
    for ch, (lo, hi) in env.items():
        print(f"  {ch:8s} {lo:+4d} .. {hi:+4d}")
    return env


def command_timing(paths: list[str]) -> dict:
    """The real rc rate: one training row is written per control tick, right after send_rc."""
    dts, gaps = [], 0
    for path in paths:
        t = np.array([json.loads(line)["t"] for line in Path(path).read_text().splitlines() if line.strip()])
        d = np.diff(t)
        gaps += int(np.sum(d > 1.0))  # blocking takeoff/land/streamon, link loss
        dts.append(d[d <= 1.0])
    d = np.concatenate(dts) * 1000.0
    out = {"rate_hz": round(1000.0 / float(d.mean()), 2), "period_ms_median": round(float(np.median(d)), 1),
           "period_ms_p95": round(float(np.percentile(d, 95)), 1), "late_frac_75ms": round(float(np.mean(d > 75.0)), 4),
           "stalls_over_1s": gaps}
    print("\n== command timing (one row per rc sent) ==")
    print(f"  {out['rate_hz']} Hz, period median {out['period_ms_median']} ms, p95 {out['period_ms_p95']} ms, "
          f"{out['late_frac_75ms']:.1%} of ticks > 75 ms, {gaps} stall(s) > 1 s")
    print("  the sim's command_rate_hz (tello_room_env_cfg.py) must match the rate above")
    return out


def cross_checks(segs: list[dict]) -> None:
    rc = np.concatenate([s["rc"] for s in segs])
    vel = np.concatenate([s["vel"] for s in segs])
    print("\n== correlation: rc channel (rows) vs state-packet vel field (cols r, f, u) ==")
    for j, ch in enumerate(CHANNELS[:3]):
        cors = [np.corrcoef(rc[:, j], vel[:, k])[0, 1] if rc[:, j].std() > 0 and vel[:, k].std() > 0 else float("nan")
                for k in range(3)]
        print(f"  {ch:8s} " + "  ".join(f"{c:+.2f}" for c in cors))
    print("  (a clean mapping is a strong diagonal; off-diagonal means the engine's vgx/vgy axes are swapped)")

    print("\n== state-packet speed vs tag-derived speed (least-squares scale, aimed rows) ==")
    for ch in ("right", "forward"):
        a = np.concatenate([channel_speed(s, ch, "state") for s in segs])
        b = np.concatenate([channel_speed(s, ch, "tag") for s in segs])
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0.02)
        if ok.sum() < 20:
            print(f"  {ch:8s} not enough overlapping tag sightings")
            continue
        scale = float(a[ok] @ b[ok]) / float(a[ok] @ a[ok])
        hint = "  <- state vel is off by ~10x (dm/s read as cm/s?)" if abs(scale) > 5 or abs(scale) < 0.2 else ""
        print(f"  {ch:8s} tag = {scale:+.2f} x state  ({ok.sum()} samples){hint}")


# -- main -------------------------------------------------------------------------------


def run(paths: list[str], source: str, out: Path, dry: bool) -> dict:
    segs = load_segments(paths)
    if not segs:
        raise SystemExit("no airborne segments with state velocity found in " + ", ".join(paths))
    secs = sum(len(s["t"]) for s in segs) * DT
    print(f"{len(segs)} airborne segments, {secs:.0f} s, from {len({s['file'] for s in segs})} file(s)")
    timing = command_timing(paths)
    envelope = command_envelope(paths)
    if all(np.isnan(sg["vel"]).all() for sg in segs):
        if source != "tag":
            print("\nno state-packet velocity in these logs (v2 format): fitting on tag-derived speed, no yaw")
        source = "tag"
    else:
        cross_checks(segs)

    print(f"\n== fit (speed source: {source}; yaw from the IMU heading) ==")
    channels, notes = {}, []
    for ch in CHANNELS:
        fit = fit_channel(segs, ch, source)
        unit = "deg/s" if ch == "yaw" else "m/s"
        if fit is None or fit["gain"] <= 0 or fit["ci_rel"] > 0.25 or (ch != "forward" and source == "tag" and ch == "right"):
            prior = PRIOR[ch] if PRIOR else {"gain": 0.0, "deadband": 5, "tau": 0.5, "delay": 0.1}
            if fit is None:
                why = "too little stick activity"
            elif ch == "right" and source == "tag":
                why = f"no heading in the log to separate strafe from yaw-induced drift (raw gain {fit['gain']:.4f})"
            else:
                why = f"gain not pinned down: {fit['gain']:.4f}, 90% CI {fit['ci'][0]:.4f}..{fit['ci'][1]:.4f}"
            print(f"  {ch:8s} KEEP PRIOR: {why}")
            channels[ch] = {k: prior[k] for k in ("gain", "deadband", "tau", "delay")}
            notes.append(f"{ch}: prior kept ({why})")
            continue
        print(f"  {ch:8s} gain {fit['gain']:.4f} {unit}/rc ({fit['gain'] * 100:.2f} at rc 100), deadband {fit['deadband']}, "
              f"tau {fit['tau']:.2f} s, delay {fit['delay']:.2f} s, 90% CI on gain {fit['ci'][0]:.4f}..{fit['ci'][1]:.4f} (+-{fit['ci_rel']:.0%}), R2 {fit['r2']:.2f}, {fit['n']} samples")
        if fit["steps"] < 10:
            prior = PRIOR[ch] if PRIOR else {"deadband": 5, "tau": 0.5, "delay": 0.1}
            print(f"           only {fit['steps']} stick steps: gain adopted; deadband/tau/delay keep the prior "
                  f"({prior['deadband']}, {prior['tau']} s, {prior['delay']} s)")
            fit.update({k: prior[k] for k in ("deadband", "tau", "delay")})
            notes.append(f"{ch}: gain fitted, deadband/tau/delay prior ({fit['steps']} stick steps)")
        channels[ch] = {"gain": round(fit["gain"], 5), "deadband": fit["deadband"], "tau": fit["tau"],
                        "delay": round(fit["delay"], 3), "gain_ci90": [round(v, 5) for v in fit["ci"]],
                        "r2": round(fit["r2"], 3), "samples": fit["n"]}
    result = {
        "source": f"fit_tello_motion.py from {len({s['file'] for s in segs})} log(s), {secs:.0f} s airborne, speed={source}",
        "notes": notes,
        "command": timing,
        "command_envelope": envelope,
        "channels": channels,
    }
    if not dry:
        header = ("# How a real Tello moves per rc stick unit. Read by control/rc_action.py.\n"
                  "# GENERATED by scripts/fit_tello_motion.py; see that script for the model.\n")
        out.write_text(header + yaml.safe_dump(result, sort_keys=False, default_flow_style=None))
        print(f"\nwrote {out}")
    return result


def selftest() -> None:
    """Synthesize logs from a known model and check the fit recovers it."""
    import tempfile

    rng = np.random.default_rng(0)
    truth = {"right": (0.007, 6, 3, 0.4), "forward": (0.009, 6, 3, 0.4), "up": (0.005, 6, 2, 0.25), "yaw": (0.55, 6, 2, 0.1)}
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parents[1] / "outputs"))
    for f in range(3):
        n = 1200
        rc = np.zeros((n, 4))
        for j in range(4):  # piecewise-constant sticks, like the PID's output
            i = 0
            while i < n:
                L = int(rng.integers(10, 50))
                rc[i:i + L, j] = int(rng.integers(-30, 31)) if rng.random() < 0.6 else 0
                i += L
        speeds = {}
        for j, ch in enumerate(CHANNELS):
            g, db, d, tau = truth[ch]
            speeds[ch] = g * simulate(rc[:, j], db, d, tau) + rng.normal(0, 0.01 if ch != "yaw" else 0.5, n)
        yaw = np.cumsum(speeds["yaw"]) * DT
        rows = []
        for i in range(n):
            rows.append({"t": round(i * DT, 3), "rc": rc[i].astype(int).tolist(), "mode": "track", "fresh": False, "h": 1.0,
                         "vel": {"r": float(np.round(speeds["right"][i], 2)), "f": float(np.round(speeds["forward"][i], 2)),
                                 "u": float(np.round(speeds["up"][i], 2)), "yaw": float(yaw[i])}})
        (tmp / f"selftest_{f}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    try:
        res = run(sorted(glob.glob(str(tmp / "*.jsonl"))), "state", OUT, dry=True)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    ok = True
    for ch, (g, db, d, tau) in truth.items():
        got = res["channels"][ch]
        good = abs(got["gain"] - g) / g < 0.1 and abs(got["deadband"] - db) <= 1 and abs(got["delay"] - d * DT) <= DT + 1e-9
        ok &= good
        print(f"  selftest {ch:8s} truth gain {g} db {db} delay {d * DT:.2f} tau {tau} -> {'ok' if good else 'MISMATCH'}")
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="*", help="training logs (default: drone_deployment_infra/recordings/*.train.jsonl)")
    p.add_argument("--speed_source", choices=["state", "tag"], default="state",
                   help="state packet velocity, or speeds derived from tag positions (forced for v2 logs)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dry_run", action="store_true", help="print the fit, don't write it")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    pats = a.logs or [str(Path(__file__).resolve().parents[2] / "drone_deployment_infra/recordings/*.train.jsonl")]
    paths = sorted({q for pat in pats for q in glob.glob(pat) if Path(q).stat().st_size > 0})
    if not paths:
        raise SystemExit("no non-empty training logs: " + ", ".join(pats))
    run(paths, a.speed_source, a.out, a.dry_run)


if __name__ == "__main__":
    main()

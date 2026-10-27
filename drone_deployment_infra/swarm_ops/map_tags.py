#!/usr/bin/env python3
"""Build map.yaml: every tag's pose relative to tag 10. NO FLYING (props off, hand-held).

Show tag 10 together with another tag (or a tag together with an already-GREEN tag:
chained). Each tag collects many views; outliers are dropped, the rest averaged.
A tag turns GREEN when it has enough views that agree.

DEFAULT = relaxed: small/far/noisy views are used; straight-on "ambiguous" views are
resolved (tag upright on a wall + consistency with the tag's running estimate) instead
of rejected.  --strict = the old, picky rules.

  Window: green = mapped, yellow = collecting (views / spread), red = rejected (reason)
  Keys: s = save (keep going), q = save + quit, r = reset tag nearest image center,
        Esc = quit WITHOUT saving.  Ctrl+C = save + quit.

  python map_tags.py --drone 4 --tags-list 10,4,5,6,7
  python map_tags.py --drone 4 --tags-list 10,6,7 --strict
  python map_tags.py --drone 4 --origin 6 --tags-list 6,9,11 --out ../config/map_home.yaml

Writes ../config/map.yaml (repo) or ./map.yaml, backs up the old one first.
NOTE: tags added with add_tag_measured.py are NOT kept: re-add them after mapping.
"""
import argparse
import json
import math
import os
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import yaml

import swarm_formation as sf          # same connection / camera / tag-pose code

TAG_ID = 10
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "config")
if not os.path.isdir(CFG):
    CFG = HERE
C_FLIP = np.diag([1.0, -1.0, -1.0])   # internal (y down, z into) <-> map (y up, z out)

STRICT = dict(MIN_MARGIN=30, MIN_SIDE_PX=25, MAX_ERR_PX=1.5, MIN_SAMPLES=15, MAX_SPREAD_CM=3.0)
RELAXED = dict(MIN_MARGIN=15, MIN_SIDE_PX=15, MAX_ERR_PX=3.0, MIN_SAMPLES=8, MAX_SPREAD_CM=8.0)
P = dict(RELAXED)
AMBIG_DEG = 3.0
OUTLIER_CM = 6.0


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class TagMapBuilder:
    def __init__(self, ids):
        self.ids = ids
        self.obs = {k: [] for k in ids if k != TAG_ID}     # (R_10k, t_10k, direct)
        self.locked = {}                                   # confirmed tags stay green

    def add(self, k, R, t, direct):
        self.obs[k].append((R, t, direct))
        del self.obs[k][:-400]

    def estimate(self, k):
        """-> (R, t, n_inliers, spread_cm, n_direct) or None."""
        ob = self.obs.get(k) or []
        direct = [o for o in ob if o[2]]
        use = direct if len(direct) >= P["MIN_SAMPLES"] else ob
        if len(use) < 3:
            return None
        T = np.array([o[1] for o in use])
        med = np.median(T, axis=0)
        keep = np.linalg.norm(T - med, axis=1) < OUTLIER_CM / 100
        if keep.sum() < 3:
            return None
        Rs = [use[i][0] for i in range(len(use)) if keep[i]]
        Rm = rot_mean(Rs)
        ok = [i for i, R in zip(np.nonzero(keep)[0], Rs) if rot_deg(R, Rm) < 5]
        if len(ok) < 3:
            return None
        Tk = T[ok]
        t = Tk.mean(axis=0)
        spread = float(np.sqrt(np.mean(np.sum((Tk - t) ** 2, axis=1)))) * 100
        return rot_mean([use[i][0] for i in ok]), t, len(ok), spread, len(direct)

    def _ok(self, e):
        return e is not None and e[2] >= P["MIN_SAMPLES"] and e[3] <= P["MAX_SPREAD_CM"]

    def mapped(self, k):
        """Green = confirmed once (sticky). Later views refine it only if they still agree."""
        if k == TAG_ID:
            return True
        e = self.estimate(k)
        if self._ok(e):
            if k not in self.locked:
                print(f"  + tag {k} CONFIRMED ({e[2]} views, spread {e[3]:.1f} cm)", flush=True)
            self.locked[k] = e
        return k in self.locked

    def best(self, k):
        """Estimate to use/save: current if it still agrees, else the confirmed one."""
        e = self.estimate(k)
        return e if self._ok(e) else self.locked.get(k, e)

    def reset(self, k):
        if k in self.obs:
            self.obs[k] = []
        self.locked.pop(k, None)


def tag_pose(c, K, D, size, strict, to_ref=None, guess=None):
    """camera<-tag pose (R, t) or (None, reason).
    strict: reject ambiguous views. relaxed: resolve them -> candidate whose rotation (in the
    reference frame given by to_ref) is closest to `guess`; else the one facing horizontally."""
    cands = sf.tag_candidates(c, K, D, size)
    if not cands:
        return None, "no pose"
    e0 = cands[0][0]
    if e0 > P["MAX_ERR_PX"]:
        return None, f"fit {e0:.1f}px"
    near = [cc for cc in cands if cc[0] < 2.0 * e0 + 0.5]
    distinct = [cc for cc in near[1:] if rot_deg(cc[1], near[0][1]) > AMBIG_DEG]
    if not distinct:
        return (near[0][1], near[0][2]), None
    if strict:
        return None, "ambiguous (view at an angle)"
    opts = [near[0]] + distinct
    if to_ref is not None and guess is not None:          # consistency with running estimate
        best = min(opts, key=lambda cc: rot_deg(to_ref(cc[1]), guess))
    else:                                                 # wall tags hang upright
        best = min(opts, key=lambda cc: abs(cc[1][1, 2]))
    return (best[1], best[2]), None


def connect(d):
    if not sf.iface_ip(d.iface):
        sys.exit(f"{d.iface} has no 192.168.10.x address: run  python connect.py {d.num}")
    ssid = sf.active_ssid(d.iface)
    if ssid != d.ssid:
        sys.exit(f"{d.iface} is on {ssid!r}, expected {d.ssid!r}")
    threading.Thread(target=d.state_loop, daemon=True).start()
    for _ in range(5):
        if d.send("command", 3) == "ok":
            break
        time.sleep(1)
    else:
        sys.exit("drone did not answer 'command'")
    threading.Thread(target=d.video_loop, daemon=True).start()
    for _ in range(3):
        if d.send("streamon", 5) == "ok":
            break
        time.sleep(1)
    t0 = time.time()
    while d.frames < 5 and time.time() - t0 < 15:
        time.sleep(0.1)
    if d.frames < 5:
        sys.exit("no video frames")
    print(f"connected to {d.ssid}, battery {d.send('battery?', 5)}%")


def save(builder, ids, path):
    if os.path.exists(path):
        bak = path + time.strftime(".bak_%Y%m%d_%H%M%S")
        shutil.copy(path, bak)
        print(f"backed up old map -> {bak}")
    tags = {TAG_ID: {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "seen": 0, "samples": 0}}
    print(f"\n{'tag':>4} {'x right':>8} {'y up':>8} {'z out':>8}  {'dist':>6}  views  spread  link")
    for k in sorted(i for i in ids if i != TAG_ID):
        e = builder.best(k)
        if e is None or not builder.mapped(k):
            print(f"{k:>4}  NOT MAPPED (needs {P['MIN_SAMPLES']} consistent views) -> left out")
            continue
        R, t, n, spread, ndir = e
        Rm, tm = C_FLIP @ R @ C_FLIP, C_FLIP @ t
        tags[k] = {"R": Rm.tolist(), "t": tm.tolist(), "seen": len(builder.obs[k]), "samples": int(n)}
        print(f"{k:>4} {tm[0]:+8.3f} {tm[1]:+8.3f} {tm[2]:+8.3f}  {np.linalg.norm(tm):5.2f}m  "
              f"{n:5d}  {spread:4.1f}cm  {'direct' if ndir >= P['MIN_SAMPLES'] else 'chained'}")
    with open(path, "w") as f:
        yaml.safe_dump({"origin_tag": TAG_ID,
                        "frame": "origin tag: x right, y up, z out of the tag face; meters",
                        "tags": tags}, f, sort_keys=False)
    print(f"wrote {path}  ({len(tags)} tags incl. tag 10)\n"
          f"Tape-check a couple (center to center) before flying. Re-add tape-measured tags now "
          f"(add_tag_measured.py) if you use any.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone", type=int, required=True)
    ap.add_argument("--tags-list", required=True, help="e.g. 10,4,5,6,7")
    ap.add_argument("--config", default=os.path.join(CFG, "dongles.json"))
    ap.add_argument("--tags", default=os.path.join(CFG, "tags.yaml"))
    ap.add_argument("--camera-dir", default=os.path.join(CFG, "camera"))
    ap.add_argument("--out", default=os.path.join(CFG, "map.yaml"))
    ap.add_argument("--strict", action="store_true", help="old picky rules (reject ambiguous views)")
    ap.add_argument("--origin", type=int, default=10, help="origin tag of this map (default 10)")
    args = ap.parse_args()
    global TAG_ID
    TAG_ID = args.origin
    if args.strict:
        P.update(STRICT)
    print(f"mode: {'STRICT' if args.strict else 'RELAXED'}  {P}")

    ids = sorted({int(x) for x in args.tags_list.split(",") if x.strip()} | {TAG_ID})
    sz = (yaml.safe_load(open(args.tags)) or {}).get("sizes_mm") or {}
    sizes = {int(k): float(v) / 1000 for k, v in sz.items() if v is not None}
    missing = [k for k in ids if k not in sizes]
    if missing:
        sys.exit(f"no size in {args.tags} for tag(s) {missing}: add them under sizes_mm")
    data = json.load(open(args.config))
    data = data.get("dongles", []) if isinstance(data, dict) else data
    ent = next((e for e in data if int(e["number"]) == args.drone and e.get("ssid")), None)
    if ent is None:
        sys.exit(f"drone {args.drone} not in {args.config}")
    cam = os.path.join(args.camera_dir, f"tello_{args.drone}.yaml")
    if not os.path.exists(cam):
        cam = os.path.join(args.camera_dir, "default.yaml")
    K, D = sf.load_camera(cam)
    d = sf.Drone(args.drone, ent["iface"], ent["ssid"], K, D, False, {}, {}, None)
    print(f"MAPPING {ids} with drone {args.drone} (no flying), camera {os.path.basename(cam)}")
    connect(d)

    from pupil_apriltags import Detector
    det = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0, refine_edges=True,
                   decode_sharpening=0.25)
    builder = TagMapBuilder(ids)
    last_id, last_print = 0, 0.0
    cv2.namedWindow("map", cv2.WINDOW_NORMAL)
    try:
        while True:
            if d.frame is None or d.frame_id == last_id:
                if (cv2.waitKey(5) & 0xFF) == 27:
                    print("Esc -> quit without saving")
                    return
                continue
            last_id, img = d.frame_id, d.frame.copy()
            raw = {}
            for t_ in det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)):
                k, c = t_.tag_id, np.array(t_.corners, np.float64)
                if k not in ids:
                    cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (128, 128, 128), 1)
                    continue
                side = min(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4))
                if t_.decision_margin < P["MIN_MARGIN"]:
                    raw[k] = (c, f"weak {t_.decision_margin:.0f}")
                elif side < P["MIN_SIDE_PX"]:
                    raw[k] = (c, f"small {side:.0f}px")
                else:
                    raw[k] = (c, None)
            seen, notes = {}, {}
            # tag 10 first (anchor for the others' disambiguation)
            if TAG_ID in raw and raw[TAG_ID][1] is None:
                pose, why = tag_pose(raw[TAG_ID][0], K, D, sizes[TAG_ID], args.strict)
                if pose:
                    seen[TAG_ID] = pose
                notes[TAG_ID] = (raw[TAG_ID][0], why)
            anchor = None
            if TAG_ID in seen:
                anchor = (seen[TAG_ID][0], np.eye(3))            # (R_c,anchor ; R_10,anchor)
            else:
                cands = [m for m in raw if m != TAG_ID and raw[m][1] is None and builder.mapped(m)]
                if cands:
                    m = cands[0]
                    pose, why = tag_pose(raw[m][0], K, D, sizes[m], args.strict)
                    if pose:
                        seen[m] = pose
                        anchor = (pose[0], builder.best(m)[0])
                    notes[m] = (raw[m][0], why)
            for k, (c, why) in raw.items():
                if k in notes:
                    continue
                if why is not None:
                    notes[k] = (c, why)
                    continue
                est = builder.estimate(k) if k != TAG_ID else None
                to_ref = None
                if anchor is not None:
                    Rca, R10a = anchor
                    to_ref = lambda Rck, Rca=Rca, R10a=R10a: R10a @ Rca.T @ Rck   # -> tag-10 frame
                pose, why = tag_pose(c, K, D, sizes[k], args.strict, to_ref, est[0] if est else None)
                if pose:
                    seen[k] = pose
                notes[k] = (c, why)
            # ---- link: directly with tag 10, else through a mapped tag ----
            if TAG_ID in seen:
                R10, t10 = seen[TAG_ID]
                for k, (Rk, tk) in seen.items():
                    if k != TAG_ID:
                        builder.add(k, R10.T @ Rk, R10.T @ (tk - t10), True)
            else:
                anchors = [m for m in seen if builder.mapped(m)]
                if anchors:
                    m = max(anchors, key=lambda a: builder.best(a)[2])
                    Rm0, tm0, *_ = builder.best(m)
                    Rcm, tcm = seen[m]
                    for k, (Rk, tk) in seen.items():
                        if k in (m, TAG_ID):
                            continue
                        builder.add(k, Rm0 @ (Rcm.T @ Rk), Rm0 @ (Rcm.T @ (tk - tcm)) + tm0, False)
            # ---- draw ----
            for k, (c, why) in notes.items():
                if why is not None:
                    col, label = (0, 0, 255), f"{k}: {why}"
                elif builder.mapped(k):
                    col, label = (0, 220, 0), f"{k} mapped"
                else:
                    e = builder.estimate(k) if k != TAG_ID else None
                    col = (0, 220, 220)
                    label = f"{k} " + (f"{e[2]}/{P['MIN_SAMPLES']} {e[3]:.1f}cm" if e else "need tag 10 / green tag")
                cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, col, 3)
                x, y = c.min(axis=0)
                for cc, th in (((0, 0, 0), 4), (col, 2)):
                    cv2.putText(img, label, (int(x), max(int(y) - 8, 18)), 0, 0.6, cc, th)
            done = [k for k in ids if builder.mapped(k)]
            todo = [k for k in ids if not builder.mapped(k)]
            hdr = f"mapped {done}  still need {todo}   s=save q=save+quit Esc=quit"
            for cc, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.putText(img, hdr, (10, 25), 0, 0.6, cc, th)
            cv2.imshow("map", img)
            now = time.time()
            if now - last_print > 2.0:
                last_print = now
                parts = []
                for k in ids:
                    if k == TAG_ID:
                        continue
                    e = builder.estimate(k)
                    parts.append(f"{k}:" + ("OK" if builder.mapped(k) else
                                            (f"{e[2]}v/{e[3]:.1f}cm" if e else "-")))
                print("  " + "  ".join(parts), flush=True)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save(builder, ids, args.out)
            elif key == ord("q"):
                save(builder, ids, args.out)
                return
            elif key == 27:
                print("Esc -> quit without saving")
                return
            elif key == ord("r") and notes:
                h, w = img.shape[:2]
                k = min(notes, key=lambda i: np.linalg.norm(notes[i][0].mean(0) - [w / 2, h / 2]))
                builder.reset(k)
                print(f"reset tag {k}")
    except KeyboardInterrupt:
        print("\nCtrl+C -> saving")
        save(builder, ids, args.out)
    finally:
        d.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
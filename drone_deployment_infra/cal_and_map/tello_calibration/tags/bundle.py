"""Tag map by bundle adjustment: every camera pose and every tag pose solved together.

The pose-graph map (mapping.MapBuilder) averages tag-to-tag poses taken from single
detections. A small tag seen from a few metres has two near-equal single-tag poses (the
IPPE flip, often 10+ deg apart), and a flipped pose averaged into the map moves the tag
by centimetres to decimetres. Here the unknowns are the camera pose of every frame and the
pose of every tag (origin tag fixed), and the measurements are the raw corner pixels of
every detection. Corners are good measurements even when the single-tag pose is ambiguous,
so ambiguous views count too.

    X_cam = R_co (R_ot P + t_ot) + t_co        P = corner in its tag's frame (detect.py)

Steps: (1) flip-proof start: for each tag pair, the relative pose most frames agree on
(all four IPPE combinations per frame vote; wrong flips scatter, the right one repeats),
chained from the origin; (2) each frame's camera by joint PnP on that start; (3) robust
(Huber) least squares over everything; observations still off by > OUTLIER_PX dropped,
solved again; (4) per tag: covariance (Schur complement of the normal equations), and a
REDUNDANCY check: the frames split into alternating time blocks, each half solved alone,
the two tag positions compared. A tag is OK only if every check passes (TagQuality.ok).

Frames are the same convention as mapping.TagMap: M_origin_tag, origin frame x right,
y up, z out of the origin tag's face, metres.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from tello_calibration.tags.detect import object_corners, solve_ippe
from tello_calibration.tags.mapping import TagMap, invert

OUTLIER_PX = 3.0             # an observation's corner rms above this after the solve -> dropped
START_MAX_RMS_PX = 5.0       # a frame whose joint PnP on the start map is worse is skipped
VOTE_POS_M = 0.03            # pair votes within this distance and angle agree
VOTE_ROT_DEG = 3.0
BLOCK_FRAMES = 45            # redundancy halves alternate in blocks of this many frames (~1.5 s)
MIN_SIGMA_PX = 0.3           # corner noise floor for the covariance


@dataclass(frozen=True)
class Acceptance:
    """A tag is OK when all hold."""
    min_views: int = 30            # frames it was solved from
    max_sigma_cm: float = 0.5      # 1-sigma position uncertainty from the solve
    max_split_cm: float = 1.5      # the two independent halves agree to this
    min_spread_deg: float = 15.0   # range of viewing directions (depth / rotation need it)


@dataclass
class TagQuality:
    tag_id: int
    views: int
    links: dict[int, int]          # tag -> frames seen together
    spread_deg: float              # max angle between viewing directions
    rms_px: float                  # corner rms of its observations
    sigma_cm: float                # 1-sigma position
    sigma_deg: float               # 1-sigma rotation
    split_cm: float | None         # |half A - half B| position, None = not in both halves
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


@dataclass
class BundleResult:
    tag_map: TagMap
    quality: dict[int, TagQuality]
    rms_px: float
    frames_used: int
    observations: int
    dropped: int


# ------------------------------------------------------------------ geometry helpers
def _rodrigues(r: np.ndarray) -> np.ndarray:
    """(n, 3) rotation vectors -> (n, 3, 3)."""
    th = np.linalg.norm(r, axis=1)[:, None, None]
    k = r / np.maximum(th[:, :, 0], 1e-12)
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    small = th < 1e-9
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return np.where(small, np.eye(3) + _skew(r), R)


def _skew(r: np.ndarray) -> np.ndarray:
    S = np.zeros((len(r), 3, 3))
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 0] = -r[:, 2], r[:, 1], r[:, 2]
    S[:, 1, 2], S[:, 2, 0], S[:, 2, 1] = -r[:, 0], -r[:, 1], r[:, 0]
    return S


def _project(Xc: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(n, 4, 3) camera points, per-observation K (n, 3, 3) and dist (n, 5) -> (n, 4, 2) px."""
    z = np.maximum(Xc[..., 2], 1e-6)
    x, y = Xc[..., 0] / z, Xc[..., 1] / z
    k1, k2, p1, p2, k3 = (dist[:, i:i + 1] for i in range(5))
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = K[:, 0:1, 0] * xd + K[:, 0:1, 2]
    v = K[:, 1:2, 1] * yd + K[:, 1:2, 2]
    return np.stack([u, v], axis=-1)


def _mat(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def _rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def _rot_mean(Rs: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


# ------------------------------------------------------------------ the mapper
@dataclass
class _Obs:
    frame: int          # index into BundleMapper.frames
    tag: int
    corners: np.ndarray  # (4, 2) IPPE order


class BundleMapper:
    """Collect detections frame by frame, then solve() the map with its quality."""

    def __init__(self, origin_tag: int, sizes_m: dict[int, float],
                 acceptance: Acceptance = Acceptance()) -> None:
        self.origin = origin_tag
        self.sizes = dict(sizes_m)
        self.acc = acceptance
        self.frames: list[tuple[int, np.ndarray, np.ndarray]] = []   # (time index, K, dist)
        self.obs: list[_Obs] = []

    def add_frame(self, observations, K: np.ndarray, dist: np.ndarray, index: int | None = None,
                  usable=None) -> int:
        """Keep the frame's corner observations of sized tags (>= 2 per frame link tags).
        usable(obs) decides which detections count (default: decoded well, sized)."""
        usable = usable or (lambda o: o.tag_id in self.sizes)
        keep = [o for o in observations if usable(o)]
        if len(keep) < 2:
            return 0
        f = len(self.frames)
        d = np.zeros(5)
        d[:min(5, len(np.ravel(dist)))] = np.ravel(dist)[:5]
        self.frames.append((f if index is None else index, np.asarray(K, float), d))
        for o in keep:
            self.obs.append(_Obs(f, int(o.tag_id), np.asarray(o.corners, float).reshape(4, 2)))
        return len(keep)

    # -------------------------------------------------------------- start (flip-proof)
    def _candidates(self):
        """frame -> {tag: [M_cam_tag candidates]} from both IPPE solutions."""
        cands = defaultdict(dict)
        for o in self.obs:
            _, K, d = self.frames[o.frame]
            rv, tv, _ = solve_ippe(o.corners, K, d, self.sizes[o.tag])
            cands[o.frame][o.tag] = [_mat(cv2.Rodrigues(r)[0], t) for r, t in zip(rv, tv)]
        return cands

    @staticmethod
    def _vote(Ms: list[np.ndarray]) -> tuple[np.ndarray, int]:
        """The relative pose the most votes agree on (within VOTE_POS_M / VOTE_ROT_DEG)."""
        T = np.array([M[:3, 3] for M in Ms])
        seeds = range(0, len(Ms), max(1, len(Ms) // 300))
        best, best_n = None, -1
        for s in seeds:
            near = np.nonzero(np.linalg.norm(T - T[s], axis=1) < VOTE_POS_M)[0]
            near = [i for i in near if _rot_deg(Ms[i][:3, :3], Ms[s][:3, :3]) < VOTE_ROT_DEG]
            if len(near) > best_n:
                best, best_n = near, len(near)
        M = _mat(_rot_mean([Ms[i][:3, :3] for i in best]), T[best].mean(axis=0))
        return M, best_n

    def _start_map(self, cands) -> dict[int, np.ndarray]:
        votes = defaultdict(list)                       # (i, j) -> [M_i_j]
        for per in cands.values():
            ids = sorted(per)
            for a in ids:
                for b in ids:
                    if a != b:
                        for Ma in per[a]:
                            for Mb in per[b]:
                                votes[(a, b)].append(invert(Ma) @ Mb)
        edges = {key: self._vote(Ms) for key, Ms in votes.items()}
        poses = {self.origin: np.eye(4)}
        while True:                                      # attach by the best-supported edge
            best = None
            for (i, j), (M, n) in edges.items():
                if i in poses and j not in poses and (best is None or n > best[0]):
                    best = (n, j, poses[i] @ M)
            if best is None:
                return poses
            poses[best[1]] = best[2]

    def _camera(self, f: int, tags: dict[int, np.ndarray], poses, cands):
        """Joint PnP of frame f on the map, started from each candidate; best rms."""
        _, K, d = self.frames[f]
        obj = np.vstack([(poses[t][:3, :3] @ object_corners(self.sizes[t]).T).T + poses[t][:3, 3]
                         for t in tags])
        img = np.vstack([tags[t] for t in tags])
        best = None
        for t in tags:
            for M_ct in cands[f][t]:
                M_co = M_ct @ invert(poses[t])
                ok, rv, tv = cv2.solvePnP(obj, img, K, d, cv2.Rodrigues(M_co[:3, :3])[0],
                                          M_co[:3, 3].reshape(3, 1).copy(), useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
                pr, _ = cv2.projectPoints(obj, rv, tv, K, d)
                rms = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - img) ** 2, axis=1))))
                if best is None or rms < best[0]:
                    best = (rms, rv.ravel(), tv.ravel())
        return best

    # -------------------------------------------------------------- the solve
    def _solve(self, obs: list[_Obs], poses: dict[int, np.ndarray], cams: dict[int, tuple]):
        """Least squares over cams (frame -> (rvec, t)) and non-origin tag poses."""
        frames = sorted({o.frame for o in obs if o.frame in cams})
        tags = sorted({o.tag for o in obs} - {self.origin})
        fi = {f: i for i, f in enumerate(frames)}
        ti = {t: i for i, t in enumerate(tags)}
        obs = [o for o in obs if o.frame in fi and (o.tag == self.origin or o.tag in ti)]
        nc = len(frames)
        x0 = np.concatenate([np.r_[cams[f][0], cams[f][1]] for f in frames] +
                            [np.r_[cv2.Rodrigues(poses[t][:3, :3])[0].ravel(), poses[t][:3, 3]]
                             for t in tags])
        cam_i = np.array([fi[o.frame] for o in obs])
        tag_i = np.array([ti.get(o.tag, -1) for o in obs])
        P = np.array([object_corners(self.sizes[o.tag]) for o in obs])        # (n, 4, 3)
        uv = np.array([o.corners for o in obs])
        Ks = np.array([self.frames[o.frame][1] for o in obs])
        Ds = np.array([self.frames[o.frame][2] for o in obs])
        is_tag = tag_i >= 0

        def residuals(x):
            c = x[:6 * nc].reshape(nc, 6)
            g = x[6 * nc:].reshape(-1, 6)
            R_co, t_co = _rodrigues(c[cam_i, :3]), c[cam_i, 3:]
            Xo = P.copy()
            if is_tag.any():
                R_ot = _rodrigues(g[tag_i[is_tag], :3])
                Xo[is_tag] = np.einsum("nij,nkj->nki", R_ot, P[is_tag]) + g[tag_i[is_tag], None, 3:]
            Xc = np.einsum("nij,nkj->nki", R_co, Xo) + t_co[:, None, :]
            return (_project(Xc, Ks, Ds) - uv).reshape(-1)

        S = lil_matrix((8 * len(obs), len(x0)), dtype=int)
        for n in range(len(obs)):
            S[8 * n:8 * n + 8, 6 * cam_i[n]:6 * cam_i[n] + 6] = 1
            if tag_i[n] >= 0:
                S[8 * n:8 * n + 8, 6 * nc + 6 * tag_i[n]:6 * nc + 6 * tag_i[n] + 6] = 1
        sol = least_squares(residuals, x0, jac_sparsity=S, loss="huber", f_scale=1.0,
                            x_scale="jac", max_nfev=100)
        c = sol.x[:6 * nc].reshape(nc, 6)
        g = sol.x[6 * nc:].reshape(-1, 6)
        new_poses = {self.origin: np.eye(4)}
        for t in tags:
            new_poses[t] = _mat(cv2.Rodrigues(g[ti[t], :3])[0], g[ti[t], 3:])
        new_cams = {f: (c[fi[f], :3], c[fi[f], 3:]) for f in frames}
        per_obs = np.sqrt(np.mean(np.sum(sol.fun.reshape(-1, 4, 2) ** 2, axis=2), axis=1))
        return new_poses, new_cams, obs, per_obs, sol, (nc, tags, ti)

    @staticmethod
    def _tag_covariance(sol, nc: int, ntags: int) -> np.ndarray:
        """(ntags, 6, 6) covariance of the tag parameters: Schur complement of the cameras."""
        J = sol.jac.tocsc() if hasattr(sol.jac, "tocsc") else sol.jac
        s2 = max(float(np.median(np.abs(sol.fun)) / 0.6745) ** 2, MIN_SIGMA_PX ** 2)   # robust corner noise
        Jc, Jt = J[:, :6 * nc], J[:, 6 * nc:]
        C = (Jt.T @ Jt).toarray() if hasattr(Jt, "toarray") else Jt.T @ Jt
        for i in range(nc):
            Ji = Jc[:, 6 * i:6 * i + 6]
            rows = np.unique(Ji.nonzero()[0])
            if not len(rows):
                continue
            A = (Ji[rows].T @ Ji[rows])
            A = A.toarray() if hasattr(A, "toarray") else A
            B = Ji[rows].T @ Jt[rows]
            B = B.toarray() if hasattr(B, "toarray") else B
            C -= B.T @ np.linalg.solve(A + 1e-9 * np.eye(6), B)
        cov = np.linalg.pinv(C) * s2
        return np.array([cov[6 * k:6 * k + 6, 6 * k:6 * k + 6] for k in range(ntags)])

    def _run(self, obs: list[_Obs], poses, cams):
        poses, cams, kept, per_obs, sol, meta = self._solve(obs, poses, cams)
        bad = per_obs > OUTLIER_PX
        if bad.any():
            kept = [o for o, b in zip(kept, bad) if not b]
            poses, cams, kept, per_obs, sol, meta = self._solve(kept, poses, cams)
        return poses, cams, kept, per_obs, sol, meta

    def solve(self, redundancy: bool = True) -> BundleResult | None:
        if not self.obs:
            return None
        cands = self._candidates()
        poses = self._start_map(cands)
        by_frame = defaultdict(dict)
        for o in self.obs:
            if o.tag in poses:
                by_frame[o.frame][o.tag] = o.corners
        cams = {}
        for f, tags in by_frame.items():
            if len(tags) >= 2:
                c = self._camera(f, tags, poses, cands)
                if c is not None and c[0] < START_MAX_RMS_PX:
                    cams[f] = (c[1], c[2])
        obs = [o for o in self.obs if o.frame in cams and o.tag in poses]
        if len(cams) < 3:
            return None
        poses, cams, kept, per_obs, sol, (nc, tags, ti) = self._run(obs, poses, cams)
        cov = self._tag_covariance(sol, nc, len(tags)) if tags else []

        split = {}
        if redundancy and tags:
            halves = ([], [])
            for o in kept:
                halves[(self.frames[o.frame][0] // BLOCK_FRAMES) % 2].append(o)
            sols = []
            for h in halves:
                try:
                    sols.append(self._run(h, poses, {f: cams[f] for f in {o.frame for o in h}})[0])
                except ValueError:
                    sols.append({})
            for t in tags:
                if t in sols[0] and t in sols[1] and \
                        all(any(o.tag == t for o in h) for h in halves):
                    split[t] = float(np.linalg.norm(sols[0][t][:3, 3] - sols[1][t][:3, 3]))

        quality = {}
        views, links = defaultdict(set), defaultdict(lambda: defaultdict(int))
        dirs = defaultdict(list)
        res_by_tag = defaultdict(list)
        by_frame_kept = defaultdict(list)
        for o, e in zip(kept, per_obs):
            views[o.tag].add(o.frame)
            res_by_tag[o.tag].append(e)
            by_frame_kept[o.frame].append(o.tag)
        for f, ts in by_frame_kept.items():
            R_co = cv2.Rodrigues(np.asarray(cams[f][0]))[0]
            cam_in_origin = -R_co.T @ np.asarray(cams[f][1])
            for t in ts:
                for u in ts:
                    if u != t:
                        links[t][u] += 1
                M = poses[t]
                v = M[:3, :3].T @ (cam_in_origin - M[:3, 3])      # camera in the tag's frame
                dirs[t].append(v / np.linalg.norm(v))
        for t in sorted(poses):
            D = np.array(dirs[t]) if dirs[t] else np.zeros((0, 3))
            spread = 0.0
            if len(D) > 1:
                m = D.mean(axis=0)
                m /= np.linalg.norm(m)
                ang = np.degrees(np.arccos(np.clip(D @ m, -1, 1)))
                spread = float(2 * np.percentile(ang, 95))
            if t == self.origin:
                sig_cm = sig_deg = 0.0
            else:
                C = cov[ti[t]]
                sig_cm = float(np.sqrt(np.trace(C[3:, 3:]))) * 100
                sig_deg = math.degrees(float(np.sqrt(np.trace(C[:3, :3]))))
            q = TagQuality(t, len(views[t]), dict(links[t]), spread,
                           float(np.sqrt(np.mean(np.square(res_by_tag[t])))) if res_by_tag[t] else 0.0,
                           sig_cm, sig_deg, None if t not in split else split[t] * 100)
            if t != self.origin:
                a = self.acc
                if q.views < a.min_views:
                    q.reasons.append(f"views {q.views}<{a.min_views}")
                if q.sigma_cm > a.max_sigma_cm:
                    q.reasons.append(f"sigma {q.sigma_cm:.1f}cm>{a.max_sigma_cm}")
                if redundancy and q.split_cm is None:
                    q.reasons.append("not in both halves")
                elif q.split_cm is not None and q.split_cm > a.max_split_cm:
                    q.reasons.append(f"halves differ {q.split_cm:.1f}cm>{a.max_split_cm}")
                if q.spread_deg < a.min_spread_deg:
                    q.reasons.append(f"view spread {q.spread_deg:.0f}deg<{a.min_spread_deg:.0f}")
            quality[t] = q

        seen = {t: len(views[t]) for t in poses}
        samples = {t: len(res_by_tag[t]) for t in poses}
        tag_map = TagMap(self.origin, poses, seen, samples)
        rms = float(np.sqrt(np.mean(np.sum(sol.fun.reshape(-1, 4, 2) ** 2, axis=2))))
        return BundleResult(tag_map, quality, rms, len(cams), len(kept), len(obs) - len(kept))


def result_to_dict(res: BundleResult) -> dict:
    """map.yaml contents: TagMap's fields + each tag's quality (extra keys; readers of R/t
    are unaffected)."""
    data = res.tag_map.to_dict()
    data["method"] = "bundle adjustment (tello_calibration.tags.bundle)"
    data["rms_px"] = round(res.rms_px, 3)
    for t, q in res.quality.items():
        data["tags"][int(t)].update({
            "ok": q.ok, "views": q.views, "sigma_cm": round(q.sigma_cm, 2),
            "sigma_deg": round(q.sigma_deg, 2),
            "split_cm": None if q.split_cm is None else round(q.split_cm, 2),
            "spread_deg": round(q.spread_deg, 1), "rms_px": round(q.rms_px, 2),
            "links": {int(k): int(v) for k, v in sorted(q.links.items())},
            "issues": list(q.reasons)})
    return data


def report(res: BundleResult) -> str:
    lines = [f"solved {res.frames_used} frames, {res.observations} tag views "
             f"({res.dropped} dropped as outliers), corner rms {res.rms_px:.2f} px",
             f"{'tag':>4} {'x right':>8} {'y up':>8} {'z out':>8} {'views':>6} {'sigma':>7} "
             f"{'halves':>7} {'spread':>7}  status"]
    for t in sorted(res.quality):
        q, p = res.quality[t], res.tag_map.poses[t][:3, 3]
        split = "-" if q.split_cm is None else f"{q.split_cm:.1f}cm"
        status = "origin" if t == res.tag_map.origin_tag else ("OK" if q.ok else "; ".join(q.reasons))
        lines.append(f"{t:>4} {p[0]:+8.3f} {p[1]:+8.3f} {p[2]:+8.3f} {q.views:>6} "
                     f"{q.sigma_cm:5.2f}cm {split:>7} {q.spread_deg:5.0f}deg  {status}")
    return "\n".join(lines)


def _self_check() -> None:
    """Synthetic room: origin + 3 small tags (one on a side wall, one only seen with the
    others = chained), a camera sweeping in front, 0.3 px corner noise. The solved map must
    match the truth to < 5 mm and every tag must pass the quality checks."""
    rng = np.random.default_rng(1)
    K = np.array([[930.0, 0, 480], [0, 930, 360], [0, 0, 1]])
    dist = np.zeros(5)
    sizes = {10: 0.5715, 4: 0.13208, 5: 0.13208, 7: 0.13208}

    def ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    truth = {10: np.eye(4),
             4: _mat(np.eye(3), [0.7, 0.7, 0.0]),
             5: _mat(ry(math.radians(-60)), [1.4, 0.1, 0.6]),       # side wall, at an angle
             7: _mat(np.eye(3), [-0.8, -0.6, 0.3])}
    mapper = BundleMapper(10, sizes)

    class O:
        def __init__(self, tag_id, corners):
            self.tag_id, self.corners = tag_id, corners
    for i in range(600):
        a = 2 * math.pi * i / 600
        eye = np.array([0.9 * math.sin(a), 0.2 * math.cos(2 * a), 2.6 + 0.6 * math.cos(a)])
        look = np.array([0.4, 0.0, 0.0]) - eye
        z = look / np.linalg.norm(look)                     # camera z (forward), origin frame
        x = np.cross([0.0, -1.0, 0.0], z)                  # camera y = down = -origin y
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_oc = np.column_stack([x, y, z])
        M_co = invert(_mat(R_oc, eye))
        obs = []
        for t, M_ot in truth.items():
            M_ct = M_co @ M_ot
            pts = (M_ct[:3, :3] @ object_corners(sizes[t]).T).T + M_ct[:3, 3]
            if (pts[:, 2] < 0.3).any():
                continue
            uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
            uv = uv.reshape(4, 2)
            facing = (M_ct[:3, :3] @ [0, 0, 1.0]) @ (-M_ct[:3, 3]) > 0
            inside = ((uv >= 0) & (uv < [960, 720])).all()
            if facing and inside:
                obs.append(O(t, uv + rng.normal(0, 0.3, uv.shape)))
        mapper.add_frame(obs, K, dist, index=i)
    res = mapper.solve()
    assert res is not None
    for t, M in truth.items():
        err = np.linalg.norm(res.tag_map.poses[t][:3, 3] - M[:3, 3])
        assert err < 0.005, (t, err, report(res))
    assert all(q.ok for q in res.quality.values()), report(res)
    print(report(res))
    print("bundle self-check ok")


if __name__ == "__main__":
    _self_check()

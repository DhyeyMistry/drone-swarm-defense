"""Defense brain: who goes after which attacker, and where each defender should fly.

Pure logic -- no sockets, no drones -- so it runs the same in the mouse GUI, in the
kinematic sim and on real Tellos. Everything is in the MAP frame (tag-10 frame from
config/map.yaml): x right, y up, z out of tag 10's face into the room, metres.

Rules (any number of defenders vs any number of attackers):
  * An attacker is a threat while it is alive (flying, fresh report). A downed attacker
    (on the ground / dead) is ignored -- knocking them down is how we score.
  * Every threat gets at least one defender, most urgent (closest to the site) first.
    Spare defenders: one keeps goal between the site (the table) and the most urgent
    threat; any more double-team the most urgent threat from a different height.
  * No live attacker: every defender returns to its hold point (formation home / v /
    triangle / line, see hold_points).
  * A defender assigned to an attacker RAMS it at full speed, at the attacker's height.
    Aim: its current position (report age brought up to now), or with max_lead_s > 0
    the true intercept point -- where both arrive at the same time.
  * Assignments are sticky (switch only if clearly better) so defenders don't dither.
  * Targets are clamped to the geofence (room box, min/max height).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """One attacker as we know it (map frame)."""
    name: str
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    age: float = 0.0            # seconds since this position was measured
    alive: bool = True          # False once it is on the ground / not flying


@dataclass
class Plan:
    """What one defender should do this tick."""
    name: str
    mode: str                   # "ram", "guard", "hold", "idle"
    target: np.ndarray | None
    attacker: str | None = None
    lead_s: float = 0.0


@dataclass
class Geofence:
    lo: np.ndarray              # (x, y, z) min, map frame
    hi: np.ndarray              # (x, y, z) max

    def clamp(self, p: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(p, self.lo), self.hi)


@dataclass
class BrainCfg:
    site: np.ndarray                    # protected table centre, map frame
    fence: Geofence
    guard_radius: float = 1.2           # goalkeeper distance out from the site
    defender_speed: float = 0.4         # m/s we assume when computing the lead
    max_lead_s: float = 0.0             # 0 = aim at the attacker's current position
    latency_s: float = 0.0              # extra lag compensation in the aim, s
    stale_s: float = 1.5                # reports older than this are ignored
    switch_margin_m: float = 0.6        # keep the current assignment unless this much better
    double_team_dy: float = 0.35        # 2nd defender on one attacker flies this much higher
    hold_height_y: float = 0.0          # idle hold height (map y; 0 = tag-10 centre height)
    hold_spread: float = 0.8            # idle defenders spread this far apart along x
    formation: str = "home"             # idle formation: "home" (start spot), "v", "triangle", "line"
    tri_rise_m: float = 0.4             # triangle: how much higher the inner defender(s) hold
    v_center: np.ndarray | None = None  # "v": map point of the V's tip (x, z used)
    v_width: float = 1.2                # "v": distance between the two outer defenders
    v_depth: float = 0.6                # "v": how far the arms trail behind the tip (away from tag 10)
    front: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))  # where attacks come from


class DefenseBrain:
    def __init__(self, cfg: BrainCfg) -> None:
        self.cfg = cfg
        self.assigned: dict[str, str | None] = {}     # defender -> attacker (sticky)
        self.anchors: dict[str, np.ndarray] = {}      # "home" formation: where each defender started

    # ---- geometry -----------------------------------------------------------------
    def intercept(self, d_pos: np.ndarray, a: Track) -> tuple[np.ndarray, float]:
        """Where to meet attacker `a`: its position led by our travel time + latency."""
        c = self.cfg
        # Where it is NOW: the report is a.age old (capped: a stale velocity must not fling the aim
        # far away) and our commands land latency_s late.
        lag = c.latency_s + min(a.age, 0.3)
        now_pos = a.pos + a.vel * lag
        # Earliest t with |now_pos + v t - d_pos| = s t (fly straight at full speed and meet it):
        # (v.v - s^2) t^2 + 2 (r.v) t + r.r = 0, r = now_pos - d_pos.
        r, v, s = now_pos - d_pos, a.vel, max(c.defender_speed, 0.05)
        qa, qb, qc = float(v @ v - s * s), 2.0 * float(r @ v), float(r @ r)
        t_hit = None
        if abs(qa) < 1e-9:
            t_hit = -qc / qb if qb < 0 else None
        else:
            disc = qb * qb - 4 * qa * qc
            if disc >= 0:
                roots = sorted(((-qb - disc ** 0.5) / (2 * qa), (-qb + disc ** 0.5) / (2 * qa)))
                t_hit = next((t for t in roots if t > 0), None)
        if t_hit is None:                      # can't catch it (faster, running away): lead as far as allowed
            t_hit = c.max_lead_s
        t_hit = min(t_hit, c.max_lead_s)
        return now_pos + v * t_hit, lag + t_hit

    def guard_point(self, threat: Track | None) -> np.ndarray:
        """Goalkeeper spot: guard_radius out from the site toward the most urgent threat."""
        c = self.cfg
        if threat is None:
            direction = c.front.copy()
            y = c.hold_height_y
        else:
            direction = threat.pos - c.site
            y = threat.pos[1]
        direction[1] = 0.0
        n = float(np.linalg.norm(direction))
        direction = c.front.copy() if n < 1e-6 else direction / n
        p = c.site + c.guard_radius * direction
        p[1] = y
        return p

    def hold_points(self, names: list[str], positions: dict[str, np.ndarray] | None = None
                    ) -> dict[str, np.ndarray]:
        """No threats: where each defender holds.

        line      side by side in front of the site, hold_spread apart along x, one height
        triangle  the same line with the inner defender(s) tri_rise_m higher (3 -> a triangle)
        home      each defender's own anchor (where it first localized); line slot if none
        v         a V: tip at v_center toward tag 10, arms v_width apart trailing v_depth back
        Line slots go by the defenders' current left-to-right order, so nobody crosses.
        """
        c = self.cfg
        base = self.guard_point(None)
        k = len(names)
        if positions and all(n in positions for n in names):
            order = sorted(names, key=lambda n: positions[n][0])
        else:
            order = sorted(names)
        out = {}
        if c.formation == "v" and c.v_center is not None:
            # V with its tip toward tag 10 (the attack side), arms trailing back by v_depth;
            # left-to-right slot order matches the defenders' current left-to-right order.
            half = (k - 1) / 2.0
            out = {}
            for i, n in enumerate(order):
                off = (i - half) / half if half else 0.0          # -1 .. +1 across the V
                p = np.array([c.v_center[0] + off * c.v_width / 2.0, c.hold_height_y,
                              c.v_center[2] + abs(off) * c.v_depth])
                out[n] = p
            return out
        for i, n in enumerate(order):
            if c.formation == "home" and n in self.anchors:
                out[n] = self.anchors[n].copy()
                continue
            p = base.copy()
            p[0] += (i - (k - 1) / 2.0) * c.hold_spread
            if c.formation == "triangle" and k >= 3 and 0 < i < k - 1:
                p[1] += c.tri_rise_m
            out[n] = p
        return out

    # ---- assignment ---------------------------------------------------------------
    def _assign(self, defenders: dict[str, np.ndarray], threats: list[Track]) -> dict[str, str | None]:
        """Brute force over every defender -> (attacker | goalkeeper) assignment.

        Cost = sum of distances to the meeting points, a big penalty for every uncovered
        threat (weighted by how close it is to the site), and a bonus for keeping the
        previous assignment. 3 defenders x (2 attackers + keeper) = 27 options: trivial.
        """
        c = self.cfg
        names = list(defenders)
        options = [t.name for t in threats] + [None]
        by_name = {t.name: t for t in threats}
        urgency = {t.name: 1.0 / max(0.3, float(np.linalg.norm((t.pos - c.site)[[0, 2]])))
                   for t in threats}
        best, best_cost = None, float("inf")
        for combo in itertools.product(options, repeat=len(names)):
            cost = 0.0
            keepers = sum(1 for o in combo if o is None)
            for d, o in zip(names, combo):
                if o is None:
                    cost += float(np.linalg.norm(defenders[d] - self.guard_point(self._most_urgent(threats))))
                else:
                    meet, _ = self.intercept(defenders[d], by_name[o])
                    cost += float(np.linalg.norm(defenders[d] - meet))
                if self.assigned.get(d, "__none__") == o:
                    cost -= c.switch_margin_m
            for t in threats:
                if t.name not in combo:
                    cost += 50.0 * urgency[t.name]          # leaving a threat free is the worst
            if keepers > 1:
                cost += 5.0 * (keepers - 1)                 # one keeper is enough
            if threats and len(names) > len(threats) and keepers == 0:
                cost += 2.0                                 # prefer a keeper when we have spares
            if cost < best_cost:
                best, best_cost = combo, cost
        return dict(zip(names, best)) if best is not None else {}

    def _most_urgent(self, threats: list[Track]) -> Track | None:
        if not threats:
            return None
        return min(threats, key=lambda t: float(np.linalg.norm((t.pos - self.cfg.site)[[0, 2]])))

    # ---- main entry ---------------------------------------------------------------
    def plan(self, defenders: dict[str, np.ndarray | None], attackers: list[Track]) -> dict[str, Plan]:
        """defenders: name -> current map position (None = not localized right now)."""
        c = self.cfg
        threats = [a for a in attackers if a.alive and a.age <= c.stale_s]
        known = {n: p for n, p in defenders.items() if p is not None}
        plans = {n: Plan(n, "idle", None) for n in defenders if defenders[n] is None}
        if not known:
            return plans
        if not threats:
            self.assigned = {n: None for n in known}
            for n, p in self.hold_points(list(known), known).items():
                plans[n] = Plan(n, "hold", c.fence.clamp(p))
            return plans
        self.assigned = self._assign(known, threats)
        by_name = {t.name: t for t in threats}
        urgent = self._most_urgent(threats)
        on_target: dict[str, int] = {}
        for n in sorted(known):
            a = self.assigned.get(n)
            if a is None:
                plans[n] = Plan(n, "guard", c.fence.clamp(self.guard_point(urgent)))
                continue
            meet, lead = self.intercept(known[n], by_name[a])
            k = on_target.get(a, 0)
            on_target[a] = k + 1
            meet = meet.copy()
            meet[1] += k * c.double_team_dy                 # stack double-teamers vertically
            plans[n] = Plan(n, "ram", c.fence.clamp(meet), a, round(lead, 2))
        return plans


def default_fence(tag_map_poses: dict, tag10_height: float, margin: float = 0.4,
                  min_h: float = 0.4, max_h: float = 2.0, base_wall_m: float = 0.4) -> Geofence:
    """Flight box straight from map.yaml: the area spanned by all mapped tags, pulled in by
    `margin` on every side (negative = let it extend past the outermost tags), never closer
    than `base_wall_m` to tag 10's wall; heights above the floor."""
    pts = np.array([np.asarray(M)[:3, 3] for M in tag_map_poses.values()])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    lo = lo + margin
    hi = hi - margin
    lo[2] = max(lo[2], base_wall_m)            # never closer than this to tag 10's wall
    if hi[2] <= lo[2] + 0.5:                   # only one wall mapped (home): allow 4 m out
        hi[2] = lo[2] + 4.0
    if hi[0] - lo[0] < 1.0:                    # narrow map: allow +-2 m sideways
        lo[0], hi[0] = min(lo[0], -2.0), max(hi[0], 2.0)
    lo[1] = min_h - tag10_height
    hi[1] = max_h - tag10_height
    return Geofence(lo, hi)

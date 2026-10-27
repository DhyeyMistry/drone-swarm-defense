"""Kalman filter for the positional feed: smooths core/positional.py's per-frame tag fixes.

``Positioner.locate`` returns a raw position per camera frame (the average over the mapped
tags in view): no memory between frames, so it jitters (cm to tens of cm with range) and
sometimes flips by 0.5 m+ (an ambiguous tag pose). This filter keeps a constant-velocity
estimate per axis and

- predicts every engine tick (``predict(dt)``), so the position moves with the drone
  between fixes and a velocity estimate is available;
- fuses each fix (``update(xyz)``), weighting it against the prediction;
- rejects outliers: a fix whose innovation is improbable (Mahalanobis distance^2 above
  ``gate_chi2``) is dropped; after ``max_rejects`` drops in a row the filter assumes it is
  the estimate that is wrong and restarts from the fix;
- holds still when lost (``freeze()``): no fix for a while means no position, as before;
  the first fix after being lost restarts it (``reset()`` then ``update()``).

The SAME file runs on the laptop (numpy, one drone) and in Isaac Sim (torch, every drone of
every env at once: ``action_space_kit/tags/kalman.py`` loads this file), so a policy is
trained on exactly the estimate it gets in flight. Only array arithmetic and ``where`` are
used, through the ``xp`` module passed in (numpy or torch).

Laptop use, next to the Positioner:

    from positional_kalman_filter import FilteredPositioner   # core/ on sys.path, like positional
    pos = FilteredPositioner("tello_3")
    est = pos.locate(frame_bgr, t=time.monotonic())
    # est["xyz"] filtered position, est["vel"] velocity (m/s), est["raw"] this frame's fix or None,
    # est["accepted"] whether the fix was used, est["lost"] no fix for lost_after seconds

Only numpy is imported at module level (cv2 / tello_calibration only by FilteredPositioner).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanCfg:
    enabled: bool = True
    """Sim: publish the filtered position (False: the raw fix, as before)."""
    accel_sigma: float = 1.5
    """[m/s^2] process noise: how hard the drone may accelerate between ticks."""
    meas_sigma: float = 0.15
    """[m] per-axis noise assumed on one fix."""
    init_vel_sigma: float = 0.5
    """[m/s] velocity uncertainty when (re)starting from a fix."""
    gate_chi2: float = 16.27
    """Reject a fix above this Mahalanobis distance^2 (3 axes, 99.9%)."""
    max_rejects: int = 3
    """This many rejected fixes in a row: restart from the fix."""
    lost_after: float = 0.4
    """[s] without a fix the drone is lost (laptop wrapper; the sim uses its own lost_after)."""


class KalmanCV:
    """Constant-velocity Kalman filter, independent per axis, batched over any leading shape.

    State per axis: position p, velocity v, covariance [[a, b], [b, c]]. Arrays have shape
    ``(*batch, 3)``; masks have shape ``batch``.
    """

    def __init__(self, cfg: KalmanCfg, batch: tuple[int, ...] = (), xp=np, device=None):
        self.cfg, self.xp, self.device = cfg, xp, device
        z = self._zeros((*batch, 3))
        self.p, self.v = z, z * 0
        self.a, self.b, self.c = z * 0, z * 0, z * 0
        self.ready = self._zeros(batch) > 1  # all False: no fix yet
        self.rejects = self._zeros(batch)

    def _zeros(self, shape):
        if self.xp is np:
            return np.zeros(shape)
        return self.xp.zeros(shape, device=self.device)

    def reset(self, mask) -> None:
        """Forget the drones in ``mask`` (new episode / new flight)."""
        w = self.xp.where
        m3 = mask[..., None]
        self.p, self.v = w(m3, 0 * self.p, self.p), w(m3, 0 * self.v, self.v)
        self.ready = self.ready & ~mask
        self.rejects = w(mask, 0 * self.rejects, self.rejects)

    def predict(self, dt: float, active=None) -> None:
        """Advance ``dt`` seconds (drones in ``active``, default all)."""
        q = self.cfg.accel_sigma ** 2
        p = self.p + self.v * dt
        a = self.a + 2 * self.b * dt + self.c * dt ** 2 + q * dt ** 4 / 4
        b = self.b + self.c * dt + q * dt ** 3 / 2
        c = self.c + q * dt ** 2
        if active is None:
            self.p, self.a, self.b, self.c = p, a, b, c
            return
        w, m3 = self.xp.where, (active & self.ready)[..., None]
        self.p, self.a, self.b, self.c = w(m3, p, self.p), w(m3, a, self.a), w(m3, b, self.b), w(m3, c, self.c)

    def freeze(self, mask) -> None:
        """Lost: stop extrapolating (velocity 0), keep the last position."""
        self.v = self.xp.where(mask[..., None], 0 * self.v, self.v)

    def update(self, z, mask=None):
        """Fuse fix ``z`` for the drones in ``mask`` (default all). Returns the accepted mask."""
        cfg, w = self.cfg, self.xp.where
        if mask is None:
            mask = self._zeros(self.ready.shape) < 1
        r = cfg.meas_sigma ** 2
        y = z - self.p
        S = self.a + r
        d2 = (y * y / S).sum(-1)
        ok = mask & self.ready & (d2 <= cfg.gate_chi2)
        rejected = mask & self.ready & ~ok
        self.rejects = w(rejected, self.rejects + 1, w(ok, 0 * self.rejects, self.rejects))
        # fused
        ka, kb = self.a / S, self.b / S
        p, v = self.p + ka * y, self.v + kb * y
        a, b, c = self.a - ka * self.a, self.b - ka * self.b, self.c - kb * self.b
        # (re)start from the fix: first fix, or the estimate kept disagreeing
        start = mask & (~self.ready | (self.rejects >= cfg.max_rejects))
        s3, o3 = start[..., None], ok[..., None]
        self.p = w(s3, z, w(o3, p, self.p))
        self.v = w(s3, 0 * v, w(o3, v, self.v))
        self.a = w(s3, 0 * a + r, w(o3, a, self.a))
        self.b = w(s3, 0 * b, w(o3, b, self.b))
        self.c = w(s3, 0 * c + cfg.init_vel_sigma ** 2, w(o3, c, self.c))
        self.rejects = w(start, 0 * self.rejects, self.rejects)
        self.ready = self.ready | start
        return ok | start


class FilteredPositioner:
    """``core.positional.Positioner`` + KalmanCV, for one drone in the flight loop."""

    def __init__(self, drone: str, cfg: KalmanCfg | None = None, **positioner_kwargs) -> None:
        try:  # flight scripts put core/ on sys.path (``from positional import Positioner``)
            from positional import Positioner
        except ImportError:
            from core.positional import Positioner

        self.positioner = Positioner(drone, **positioner_kwargs)
        self.cfg = cfg or KalmanCfg()
        self.kf = KalmanCV(self.cfg)
        self._t = None
        self._last_fix_t = None
        self._lost = True

    def locate(self, frame_bgr, t: float) -> dict:
        """Filtered position at time ``t`` (seconds, monotonic) from this frame (if it has tags)."""
        if self._t is not None:
            self.kf.predict(max(t - self._t, 0.0), active=np.asarray(not self._lost))
        self._t = t
        raw = self.positioner.locate(frame_bgr)
        accepted = False
        if raw is not None:
            if self._last_fix_t is not None and t - self._last_fix_t > self.cfg.lost_after:
                self.kf.reset(np.asarray(True))  # back from lost: restart from this fix
            accepted = bool(self.kf.update(np.asarray(raw["xyz"], float)))
            if accepted:
                self._last_fix_t = t
        lost = self._last_fix_t is None or t - self._last_fix_t > self.cfg.lost_after
        if lost:
            self.kf.freeze(np.asarray(True))
        self._lost = lost
        ready = bool(self.kf.ready)
        return {
            "xyz": [round(float(v), 3) for v in self.kf.p] if ready else None,
            "vel": [round(float(v), 3) for v in self.kf.v] if ready else None,
            "raw": raw,
            "accepted": accepted,
            "lost": lost,
            "fix_age": t - self._last_fix_t if self._last_fix_t is not None else float("inf"),
            "position_variance": self.kf.a.tolist() if ready else None,
            "velocity_variance": self.kf.c.tolist() if ready else None,
        }


def _selftest() -> None:
    """A drone flying at 0.4 m/s with 15 cm fix noise and one 0.8 m flip."""
    rng = np.random.default_rng(0)
    kf = KalmanCV(KalmanCfg())
    dt, err_raw, err_kf = 0.074, [], []
    for k in range(200):
        truth = np.array([6.0 - 0.4 * k * dt, 0.2, 1.0])
        kf.predict(dt)
        z = truth + rng.normal(0, 0.15, 3)
        if k == 120:
            z = truth + np.array([0.0, 0.8, 0.0])  # pose flip
        ok = kf.update(z)
        if k == 120:
            assert not ok, "the flip should be rejected"
        if k > 20:
            err_raw.append(np.linalg.norm(z - truth))
            err_kf.append(np.linalg.norm(kf.p - truth))
    print(f"raw fix error {np.mean(err_raw) * 100:.1f} cm, filtered {np.mean(err_kf) * 100:.1f} cm, "
          f"velocity {kf.v.round(2)} (true [-0.4, 0, 0])")
    assert np.mean(err_kf) < 0.6 * np.mean(err_raw)
    print("SELFTEST PASSED")


if __name__ == "__main__":
    _selftest()

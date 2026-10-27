"""Offline/inference-only room PPO adapter. No socket, takeoff, or flight API calls.

    python core/room_policy.py --checkpoint RUN/model.zip --replay samples.jsonl --out commands.jsonl

Replay rows contain the simulator-equivalent filtered feed: position, velocity, position_variance,
velocity_variance (world XYZ), yaw (radians), fix_age (since accepted fix), fresh, acquired,
lost, height, taking_off, landed, target. Optional expected_observation / expected_action
arrays turn a recording into a parity check. Use camera_pose_to_world before filtering raw
Positioner xyz/R. The engine owns SDK takeoff and landing acknowledgement/touchdown state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .landing_gate import update_landing_gate
    from .team_features import peer_feature_count, peer_features
except ImportError:
    from landing_gate import update_landing_gate
    from team_features import peer_feature_count, peer_features

P_MAP_TO_WORLD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def camera_pose_to_world(xyz, rotation, tag10_height, camera_offset=(0.05, 0.0, 0.0), camera_pitch=0.0):
    """Positioner camera->tag10 map pose to body centre and yaw in Z-up world.

    OpenCV camera columns are right/down/forward; Tello body columns are forward/left/up.
    The camera lever arm must be removed before feeding the Kalman filter.
    """
    c, s = np.cos(camera_pitch), np.sin(camera_pitch)
    camera_in_body = np.column_stack(([0.0, -1.0, 0.0], [-s, 0.0, -c], [c, 0.0, -s]))
    body_rotation = P_MAP_TO_WORLD @ np.asarray(rotation) @ camera_in_body.T
    position = P_MAP_TO_WORLD @ np.asarray(xyz) + [0.0, 0.0, tag10_height] - body_rotation @ camera_offset
    return position, float(np.arctan2(body_rotation[1, 0], body_rotation[0, 0]))


def build_observation(cfg, feed, actions, at_waypoint, landing):
    """Exact room observation before normalization, optionally extended with peer features."""
    p, yaw = np.asarray(feed["position"]), np.asarray(feed["yaw"])
    target = np.asarray(feed["target"])
    c, s = np.cos(yaw), np.sin(yaw)

    def heading(target):
        r = target - p
        return np.stack([c * r[..., 0] + s * r[..., 1], s * r[..., 0] - c * r[..., 1], r[..., 2]], -1)

    to10 = np.array([0.0, 0.0, cfg.tag10_height]) - p
    bearing = np.arctan2(to10[..., 1], to10[..., 0]) - yaw

    def column(key):
        return np.asarray(feed[key])[..., None]

    blocks = [
        p,
        np.stack([c, s], -1),
        column("fresh"),
        np.minimum(column("fix_age"), cfg.max_fix_age),
        column("lost"),
        heading(target + [0.0, 0.0, cfg.waypoint_height]),
        heading(target),
        at_waypoint[..., None],
        np.stack([np.sin(bearing), np.cos(bearing)], -1),
        column("height"),
        actions,
        (landing | np.asarray(feed["landed"]))[..., None],
    ]
    if cfg.start_on_floor:
        blocks.append(column("taking_off"))
    if getattr(cfg, "peer_observations", False):
        blocks.append(peer_features(p, np.asarray(feed["velocity"]), yaw,
                                    np.asarray(feed["position_variance"]),
                                    np.asarray(feed["velocity_variance"]), np,
                                    sort_neighbors=getattr(cfg, "peer_sort_by_distance", False),
                                    order_version=getattr(cfg, "peer_order_version", 1)))
    return np.nan_to_num(np.concatenate(blocks, -1), nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


class RoomPolicy:
    """Central or shared-peer inference, with the simulator's frame order and landing gate."""

    def __init__(self, checkpoint):
        from stable_baselines3 import PPO

        ckpt = Path(checkpoint).resolve()
        self.manifest = json.loads((ckpt.parent / "params/run.json").read_text())
        self.cfg = cfg = namespace(self.manifest["env"])
        if not cfg.land_gate_enabled or not cfg.kalman.enabled:
            raise ValueError("Deployment adapter requires the estimated-state gate and Kalman filtering")
        self.model = PPO.load(str(ckpt), device="cpu")
        match = re.match(r"model_(\d+)_steps$", ckpt.stem)
        name = f"model_vecnormalize_{match[1]}_steps.pkl" if match else "model_vecnormalize.pkl"
        # The model and normalization pickle must come from the same trusted training run.
        self.normalizer = None
        if self.manifest["agent"].get("normalize_input"):
            with (ckpt.parent / name).open("rb") as f:
                self.normalizer = pickle.load(f)
        self.n_stack = self.manifest["agent"].get("frame_stack", 1)
        self.central = bool(self.manifest["agent"].get("central", False))
        own_dim = 25 if cfg.start_on_floor else 24
        if getattr(cfg, "peer_observations", False):
            own_dim += peer_feature_count(cfg.num_drones)
        dim = own_dim * (cfg.num_drones if self.central else 1)
        if self.model.observation_space.shape != (dim * self.n_stack,):
            raise ValueError("Checkpoint observation layout does not match the room contract")
        if self.model.action_space.shape != (5 * (cfg.num_drones if self.central else 1),):
            raise ValueError("Checkpoint action layout does not match the room contract")
        self.lo, self.hi = -np.asarray(cfg.rc.rc_limit), np.asarray(cfg.rc.rc_limit)
        if cfg.rc.use_envelope:
            import yaml

            asset = self.manifest["assets"]["motion_fit"]
            data = (ckpt.parent / asset["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != asset["sha256"]:
                raise ValueError("Motion-fit snapshot hash mismatch")
            envelope = yaml.safe_load(data)["command_envelope"]
            self.lo = np.array([envelope[c][0] for c in ("right", "forward", "up", "yaw")])
            self.hi = np.array([envelope[c][1] for c in ("right", "forward", "up", "yaw")])
        self.reset()

    def reset(self):
        d = self.cfg.num_drones
        self.actions = np.zeros((d, 5), np.float32)
        self.at_waypoint = np.zeros(d, bool)
        self.landing = np.zeros(d, bool)
        self.dwell = np.zeros(d)
        self.votes = np.zeros(d, int)
        shape = self.model.observation_space.shape if self.central else (d, *self.model.observation_space.shape)
        self.history = np.zeros(shape, np.float32)
        self._steps = 0

    def step(self, feed):
        cfg = self.cfg
        p, v = np.asarray(feed["position"]), np.asarray(feed["velocity"])
        pv, vv = np.asarray(feed["position_variance"]), np.asarray(feed["velocity_variance"])
        if p.shape != (cfg.num_drones, 3) or not np.isfinite(p).all():
            raise ValueError("Feed must provide finite world positions for every configured drone")
        target = np.asarray(feed["target"])
        fresh, lost = np.asarray(feed["fresh"], bool), np.asarray(feed["lost"], bool)
        taking_off, landed = np.asarray(feed["taking_off"], bool), np.asarray(feed["landed"], bool)
        # Simulator initial reset has no waypoint transition; subsequent feeds follow _get_dones.
        if self._steps:
            near = np.linalg.norm(p - target - [0.0, 0.0, cfg.waypoint_height], axis=-1) < cfg.waypoint_radius
            trusted = np.sqrt(vv).max(-1) < cfg.speed_trust_std
            self.at_waypoint |= (
                near & fresh & ~self.landing & trusted & (np.linalg.norm(v, axis=-1) <= cfg.waypoint_max_speed)
            )
        observation = build_observation(cfg, feed, self.actions, self.at_waypoint, self.landing)
        if self.central:
            observation = observation.reshape(-1)
        frame = self.normalizer.normalize_obs(observation) if self.normalizer is not None else observation
        # SB3 zeros the absent NORMALIZED frames on reset. Normalizing raw zero padding is wrong.
        self.history = np.concatenate([self.history[..., frame.shape[-1]:], frame], axis=-1).astype(np.float32)
        action, _ = self.model.predict(self.history, deterministic=True)
        self.actions = np.clip(action.reshape(cfg.num_drones, 5), -1, 1)
        dt = cfg.sim.dt * cfg.decimation
        self.dwell, ready = update_landing_gate(
            cfg,
            p,
            v,
            pv,
            vv,
            target,
            np.asarray(feed["fix_age"]),
            np.asarray(feed["acquired"], bool),
            lost,
            self.at_waypoint,
            taking_off | self.landing | landed,
            self.dwell,
            dt,
            np,
        )
        want = (self.actions[:, 4] > cfg.land_threshold) & ~taking_off
        self.votes = np.where(want, self.votes + 1, 0)
        asked = ~self.landing & (self.votes >= cfg.land_confirm_steps)
        self.votes = np.where(asked & ~ready, 0, self.votes)
        land = ~self.landing & ready & ~taking_off if cfg.auto_land else asked & ready
        self.landing |= land
        a = self.actions[:, :4]
        rc = np.rint(np.where(a >= 0, a * self.hi, -a * self.lo)).astype(int)
        rc[self.landing | taking_off | landed] = 0
        self._steps += 1
        return {
            "rc": rc.tolist(),
            "land_request": land.tolist(),
            "land_ready": ready.tolist(),
            "action": self.actions.tolist(),
            "observation": observation.reshape(-1).tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    policy = RoomPolicy(args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w") as handle:
        for line in Path(args.replay).read_text().splitlines():
            feed = json.loads(line)
            if feed.get("reset"):
                policy.reset()
            result = policy.step(feed)
            for key in ("observation", "action"):
                if "expected_" + key in feed:
                    np.testing.assert_allclose(result[key], feed["expected_" + key], rtol=1e-4, atol=1e-5)
            handle.write(json.dumps(result, allow_nan=False) + "\n")
            count += 1
    print(f"REPLAY PASSED: {count} frames; {out}")


if __name__ == "__main__":
    main()

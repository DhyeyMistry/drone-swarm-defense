"""CPU regression tests; run with Isaac's Python, without launching the simulator."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

REPO = Path(__file__).resolve().parents[2]
RL = REPO / "simulation_infra/source/action_space_kit/action_space_kit/rl"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


terminal = load("terminal_test", RL / "terminal_observation.py")
evaluation = load("evaluation_test", RL / "evaluation.py")
rewards = load("rewards_test", RL / "rewards.py")
starts = load("starts_test", RL / "start_curriculum.py")
finetune = load("finetune_test", RL / "finetune.py")
symmetry = load("symmetry_test", RL / "symmetry.py")
run_config = load("run_config_test", RL / "run_config.py")
gate = load("gate_test", REPO / "drone_deployment_infra/core/landing_gate.py")
peers = load("peers_contract_test", REPO / "drone_deployment_infra/core/team_features.py")
sensor = load("sensor_test", RL.parent / "tags/sensing.py")
room_map = load("room_map_test", RL.parent / "tags/room_map.py")
with patch.dict(sys.modules, {"landing_gate": gate, "team_features": peers}):
    adapter = load("adapter_test", REPO / "drone_deployment_infra/core/room_policy.py")


class FakeIsaac(gym.Env):
    possible_agents = ["a", "b"]
    num_envs = 2
    device = "cpu"
    render_mode = None

    def __init__(self):
        self.observation_spaces = {a: gym.spaces.Box(-100, 100, (1,)) for a in self.possible_agents}
        self.action_spaces = {a: gym.spaces.Box(-1, 1, (1,)) for a in self.possible_agents}
        self.extras = {}
        self.x = torch.zeros(2, 1)
        self.tick = 0

    def _get_observations(self):
        return {"a": self.x.clone(), "b": self.x.clone() + 10}

    def _reset_idx(self, ids):
        self.x[ids] = -1

    def reset(self, **kwargs):
        self._reset_idx(torch.arange(2))
        return self._get_observations(), self.extras

    def seed(self, seed):
        torch.manual_seed(seed)
        return seed

    def step(self, actions):
        self.tick += 1
        self.x += 2
        done = torch.tensor([self.tick == 2, False])
        if done.any():
            self.extras["log"] = {"Episode/count": 1.0}
            self._reset_idx(done.nonzero().flatten())
        return (
            self._get_observations(),
            {a: torch.ones(2) for a in self.possible_agents},
            {a: torch.zeros(2, dtype=torch.bool) for a in self.possible_agents},
            {a: done for a in self.possible_agents},
            self.extras,
        )


class CapturedIsaac(terminal.TerminalObservationMixin, FakeIsaac):
    pass


isaac_stub = types.ModuleType("isaaclab.envs")
isaac_stub.DirectMARLEnv = FakeIsaac
with patch.dict(sys.modules, {"isaaclab.envs": isaac_stub}):
    wrappers = load("wrappers_test", RL / "sb3_shared.py")


class PolicyContractTests(unittest.TestCase):
    def test_room_reflection_matches_physical_coordinates(self):
        def observation(position, yaw, target, action):
            c, s = np.cos(yaw), np.sin(yaw)
            def heading(point):
                x, y, z = point - position
                return [c*x+s*y, s*x-c*y, z]
            bearing = np.arctan2(-position[1], -position[0]) - yaw
            return np.array([*position, c, s, 1, .074, 0,
                             *heading(target), *heading(target * [1, 1, 0]), 0,
                             np.sin(bearing), np.cos(bearing), position[2], *action, 0, 0])
        position, target = np.array([2., .4, 1.1]), np.array([1., -.25, 1.22])
        action = np.array([.2, .5, -.1, .3, 1.])
        original = observation(position, .6, target, action)
        expected = observation(position * [1, -1, 1], -.6, target * [1, -1, 1], action * [-1, 1, 1, -1, 1])
        np.testing.assert_allclose(symmetry.mirror_room_observation(original), expected, atol=1e-12)
        np.testing.assert_array_equal(symmetry.mirror_room_action(action), action * [-1, 1, 1, -1, 1])
        stack = np.concatenate([np.zeros(25), original])
        np.testing.assert_array_equal(symmetry.mirror_room_observation(symmetry.mirror_room_observation(stack)), stack)
        torch.testing.assert_close(symmetry.mirror_room_observation(torch.tensor(original)), torch.tensor(expected))

    def test_precision_shaping_prefers_slow_accurate_arrival(self):
        distance = np.array([0., .2, .5, 0.])
        speed = np.array([0., 0., 0., .3])
        p = rewards.precision_potential(distance, speed, 80., .2, .15, np)
        self.assertEqual(p[0], 80.)
        self.assertTrue(np.all((p >= 0) & (p <= 80)))
        self.assertGreater(p[0], p[1])
        self.assertGreater(p[1], p[2])
        self.assertGreater(p[0], p[3])
        tensor = rewards.precision_potential(torch.tensor(distance), torch.tensor(speed), 80., .2, .15, torch)
        np.testing.assert_allclose(p, tensor.numpy())
        # Adding precision shaping retains the same discounted telescoping property.
        gamma = .995
        total = sum(gamma**i * rewards.potential_delta(p[i], p[i+1], np.array(i == 2), gamma, np)
                    for i in range(3))
        self.assertAlmostEqual(total, -p[0])
        with self.assertRaises(ValueError):
            rewards.precision_potential(distance, speed, 80., 0., .15, np)

    def test_warm_start_preserves_deterministic_actor(self):
        env = gym.make("Pendulum-v1")
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, device="cpu", seed=9)
        model.learn(8)
        obs = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, -0.3]], dtype=np.float32)
        before, _ = model.predict(obs, deterministic=True)
        with torch.no_grad():
            values_before = model.policy.predict_values(torch.from_numpy(obs)).clone()
        finetune.prepare_finetune(model, action_std=.25)
        after, _ = model.predict(obs, deterministic=True)
        np.testing.assert_array_equal(before, after)
        with torch.no_grad():
            self.assertTrue(torch.equal(model.policy.predict_values(torch.from_numpy(obs)), values_before))
        np.testing.assert_allclose(model.policy.log_std.detach().exp().numpy(), .25)
        finetune.prepare_finetune(model, value_scale=.01)
        np.testing.assert_array_equal(before, model.predict(obs, deterministic=True)[0])
        with torch.no_grad():
            torch.testing.assert_close(model.policy.predict_values(torch.from_numpy(obs)), .01 * values_before)
        self.assertNotIn(model.policy.value_net.weight, model.policy.optimizer.state)
        self.assertIn(model.policy.action_net.weight, model.policy.optimizer.state)
        with self.assertRaises(ValueError):
            finetune.prepare_finetune(model, action_std=0)
        with self.assertRaises(ValueError):
            finetune.prepare_finetune(model, value_scale=float("nan"))
        env.close()

    def test_start_mixture_preserves_full_task_and_replays_easier_starts(self):
        cfg = types.SimpleNamespace(num_drones=1, spawn_short_fraction=0.0, spawn_hover_fraction=0.0,
                                    spawn_short_x_range=(1.5, 5.5), spawn_short_y_radius=0.4,
                                    spawn_hover_radius=0.1, waypoint_height=1.22)
        xy = torch.tensor([6.5, 0.0]).expand(4096, 1, 2).clone()
        z = torch.full((4096, 1), 0.025)
        spots = torch.tensor([1.0, 0.25, 0.0]).expand(4096, 1, 3)
        before = torch.random.get_rng_state()
        x0, z0, takeoff0, kind0 = starts.mix_starts(cfg, xy, z, spots, True)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(x0, xy) and torch.equal(z0, z))
        self.assertTrue(takeoff0.all() and (kind0 == 0).all())
        cfg.spawn_short_fraction, cfg.spawn_hover_fraction = 0.30, 0.15
        starts.validate_start_mixture(cfg)
        torch.manual_seed(907)
        x, height, takeoff, kind = starts.mix_starts(cfg, xy, z, spots, True)
        full, short, hover = kind == 0, kind == 1, kind == 2
        self.assertTrue(torch.equal(x[full], xy[full]) and torch.equal(height[full], z[full]))
        self.assertTrue(takeoff[~hover].all() and not takeoff[hover].any())
        self.assertTrue(((x[short, :, 0] >= 1.5) & (x[short, :, 0] <= 5.5)).all())
        self.assertTrue(((x[hover] - spots[hover, :, :2]).abs() <= 0.1 + 1e-6).all())
        self.assertTrue(((height[hover] - 1.22).abs() <= 0.05 + 1e-6).all())
        for mask, fraction in ((full, .55), (short, .30), (hover, .15)):
            self.assertAlmostEqual(mask.float().mean().item(), fraction, delta=.03)
        cfg.spawn_short_fraction = .9
        with self.assertRaises(ValueError):
            starts.validate_start_mixture(cfg)

    def test_tag_visible_at_landing_height(self):
        tags = list(room_map.load_room_tags(1.22).values())
        points = torch.tensor([t.pos.tolist() for t in tags], dtype=torch.float32)
        normals = torch.tensor([t.normal.tolist() for t in tags], dtype=torch.float32)
        sizes = torch.tensor([t.size_m for t in tags])
        for y in (-0.75, -0.25, 0.25, 0.75):
            yaw = torch.atan2(torch.tensor(-y), torch.tensor(-1.0)) + torch.linspace(-15, 15, 61) * torch.pi / 180
            rot = torch.zeros(61, 3, 3)
            rot[:, 0, 0] = rot[:, 1, 1] = yaw.cos()
            rot[:, 1, 0], rot[:, 0, 1], rot[:, 2, 2] = yaw.sin(), -yaw.sin(), 1
            camera = torch.tensor([1.0, y, 1.22]).expand(61, 3) + rot @ torch.tensor([0.05, 0.0, 0.0])
            mask, _ = sensor.detectable(
                sensor.TagSensorCfg(), camera, sensor.camera_axes(rot, 0.0), points, normals, sizes
            )
            self.assertGreaterEqual(mask.any(-1).float().mean().item(), 0.90)

    def test_camera_pose_frames_and_offset(self):
        # Level drone facing tag 10 (-world X); its camera centre is .05 m forward.
        body = np.diag([-1.0, -1.0, 1.0])
        camera_in_body = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        camera_world = body @ camera_in_body
        rotation_map = adapter.P_MAP_TO_WORLD.T @ camera_world
        camera_map = adapter.P_MAP_TO_WORLD.T @ np.array([0.95, 0.25, 0.28])
        p, yaw = adapter.camera_pose_to_world(camera_map, rotation_map, 1.22)
        np.testing.assert_allclose(p, [1.0, 0.25, 1.5], atol=1e-8)
        self.assertAlmostEqual(abs(yaw), np.pi)

    def test_timeout_observation_normalized_and_stacked(self):
        for wrapper in (wrappers.Sb3SharedPolicyVecEnv, wrappers.Sb3CentralVecEnv):
            base = wrapper(CapturedIsaac())
            normal = VecNormalize(base, training=False, norm_reward=False, clip_obs=100)
            normal.obs_rms.mean[:] = 1
            normal.obs_rms.var[:] = 4
            env = VecFrameStack(normal, n_stack=3)
            env.reset()
            actions = np.zeros((env.num_envs, *env.action_space.shape))
            obs1, _, _, _ = env.step(actions)
            obs2, _, done, infos = env.step(actions)
            for idx in np.flatnonzero(done):
                width = base.observation_space.shape[0]
                expected = np.concatenate(
                    [
                        obs1[idx, width:],
                        normal.normalize_obs(np.array([[3, 13]]) if width == 2 else np.array([[3 + 10 * idx]]))[0],
                    ]
                )
                np.testing.assert_allclose(infos[idx]["terminal_observation"], expected)
                self.assertFalse(np.array_equal(obs2[idx], expected))
                self.assertTrue(infos[idx]["TimeLimit.truncated"])
            env.step(actions)  # no completion: old logs must not be counted again
            self.assertEqual(sum(r["Episode/count"] for r in base.log_buffer), 1)
            self.assertEqual(len(base.episode_buffer), 1)  # shared rows don't duplicate episodes
            env.close()

    def test_balanced_cohort_and_duplicates(self):
        c = evaluation.EpisodeCohort(2, 5)
        for n in range(1, 10):
            c.add({"env_id": 0, "episode_id": n, "metrics": {"success": 1}})
        self.assertFalse(c.complete)
        for n in range(1, 3):
            c.add({"env_id": 1, "episode_id": n, "metrics": {"success": 0}})
        self.assertTrue(c.complete)
        self.assertEqual(c.summary()["episodes"], 5)
        self.assertEqual(c.summary()["metrics"]["success"], 3 / 5)
        with self.assertRaises(ValueError):
            c.add({"env_id": 1, "episode_id": 2})

    def test_continuous_play_retains_only_running_statistics(self):
        c = evaluation.EpisodeCohort(1, 10000, retain_records=False)
        for episode_id in range(1, 101):
            c.add({"env_id": 0, "episode_id": episode_id, "metrics": {"score": episode_id}})
        self.assertEqual(c.records, [])
        self.assertEqual(c.summary()["episodes"], 100)
        self.assertEqual(c.summary()["metrics"]["score"], 50.5)

    def test_landing_gate_rejects_drift_loss_restart_and_outside(self):
        cfg = types.SimpleNamespace(
            tag_sensor=types.SimpleNamespace(video_delay_steps=3),
            sim=types.SimpleNamespace(dt=0.002),
            decimation=37,
            land_max_position_std=0.12,
            land_max_velocity_std=0.35,
            land_max_xy_error=0.15,
            land_max_horizontal_speed=0.1,
            waypoint_height=1.5,
            land_height_tolerance=0.2,
            land_max_observation_age=0.35,
            land_dwell_s=0.5,
        )
        args = dict(
            position=np.array([[1.0, 0.0, 1.5]]),
            velocity=np.zeros((1, 3)),
            pos_variance=np.full((1, 3), 0.005),
            vel_variance=np.full((1, 3), 0.04),
            target=np.array([[1.0, 0.0, 0.0]]),
            fix_age=np.zeros(1),
            acquired=np.ones(1, bool),
            lost=np.zeros(1, bool),
            at_waypoint=np.ones(1, bool),
            busy=np.zeros(1, bool),
        )
        dwell = np.zeros(1)
        for _ in range(6):
            dwell, ready = gate.update_landing_gate(cfg, **args, dwell=dwell, dt=0.074, xp=np)
            self.assertFalse(ready[0])
        dwell, ready = gate.update_landing_gate(cfg, **args, dwell=dwell, dt=0.074, xp=np)
        self.assertTrue(ready[0])
        for field, bad in (
            ("position", [[1.31, 0, 1.5]]),
            ("velocity", [[0.2, 0, 0]]),
            ("lost", [True]),
            ("fix_age", [0.2]),
            ("vel_variance", [[0.25] * 3]),
            ("pos_variance", [[float("nan")] * 3]),
            ("acquired", [False]),
        ):
            inputs = {**args, field: np.array(bad)}
            d, r = gate.update_landing_gate(cfg, **inputs, dwell=dwell, dt=0.074, xp=np)
            self.assertFalse(r[0], field)
            self.assertEqual(d[0], 0, field)
        torch_args = {k: torch.from_numpy(v) for k, v in args.items()}
        d, r = gate.update_landing_gate(cfg, **torch_args, dwell=torch.from_numpy(dwell), dt=0.074, xp=torch)
        self.assertTrue(r.item())

    def test_discounted_progress_cannot_farm_loops(self):
        gamma = 0.995
        # Different paths with the same starting potential and absorbing endpoint.
        for potentials in ([2, 8, 2, 8, 2, 8], [2, 2, 2, 2, 2, 2], [2, 4, 6, 10, 20, 40]):
            result = 0
            for i, (p, n) in enumerate(zip(potentials, potentials[1:])):
                r = rewards.potential_delta(np.array(p), np.array(n), np.array(i == 4), gamma, np)
                result += gamma**i * r
            self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(rewards.potential_delta(np.array(2.0), np.array(8.0), np.array(False), gamma, np), 5.96)

    def test_saved_config_preserves_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "params").mkdir()
            (root / "params/run.json").write_text(
                json.dumps({"env": {"x": 1, "nested": {"y": 2}}, "agent": {"central": True, "gamma": 0.995}})
            )
            cfg = types.SimpleNamespace(x=9, nested=types.SimpleNamespace(y=8), collision_substeps=True,
                                        peer_order_version=2)
            agent = {"gamma": 0.9}
            run_config.restore_run_config(root / "model.zip", cfg, agent, ["env.x=9"])
            self.assertEqual(cfg.x, 9)
            self.assertEqual(cfg.nested.y, 2)
            self.assertTrue(agent["central"])
            self.assertFalse(cfg.collision_substeps)  # historical runs retain their original collision sampling
            self.assertEqual(cfg.peer_order_version, 1)  # old checkpoints keep their input ordering
            cfg.collision_substeps = True
            run_config.restore_run_config(root / "model.zip", cfg, agent, ["env.collision_substeps=True"])
            self.assertTrue(cfg.collision_substeps)


if __name__ == "__main__":
    unittest.main(verbosity=2)

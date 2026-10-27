"""CPU checks of independent-to-joint policy transfer, with no simulator or hardware."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "team_transfer_test", REPO / "simulation_infra/source/action_space_kit/action_space_kit/rl/team_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class Spaces(gym.Env):
    def __init__(self, obs, actions):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (actions,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}


def model(obs, actions, hidden):
    return PPO("MlpPolicy", Spaces(obs, actions), n_steps=4, batch_size=4, seed=17, device="cpu",
               policy_kwargs={"activation_fn": torch.nn.ELU, "net_arch": hidden})


class TeamTransferTests(unittest.TestCase):
    def test_appended_features_preserve_trained_outputs_but_can_learn(self):
        source, target = model(24, 8, [16, 12]), model(40, 8, [16, 12])
        transfer.extend_policy_observations(source.policy, target.policy, old_width=3, new_width=5,
                                            frame_stack=2, num_drones=4)
        x = torch.randn(32, 2, 4, 5)
        x[:8, :-1] = 0.
        old, new = x[..., :3].reshape(32, 24), x.reshape(32, 40)
        a = source.policy.get_distribution(old).distribution.mean
        b = target.policy.get_distribution(new).distribution.mean
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(source.policy.predict_values(old), target.policy.predict_values(new))
        b.sum().backward()
        self.assertGreater(float(target.policy.mlp_extractor.policy_net[0].weight.grad[:, 3:5].abs().sum()), 0.)
        norm = VecNormalize(DummyVecEnv([lambda: Spaces(12, 8)]))
        norm.obs_rms.mean = np.arange(12.)
        norm.obs_rms.var = np.arange(1., 13.)
        expanded = transfer.extend_observation_normalizer(norm, DummyVecEnv([lambda: Spaces(20, 8)]),
                                                          old_width=3, new_width=5, num_drones=4)
        raw = x.detach().numpy().reshape(-1, 4, 5)
        np.testing.assert_array_equal(norm.normalize_obs(raw[..., :3].reshape(-1, 12)),
                                      expanded.normalize_obs(raw.reshape(-1, 20)).reshape(-1, 4, 5)[..., :3].reshape(-1, 12))
        np.testing.assert_array_equal(norm.normalize_reward(np.arange(10.)), expanded.normalize_reward(np.arange(10.)))
        norm.close()
        expanded.close()

    def test_actor_and_critic_equal_independent_predictions_including_reset_padding(self):
        frames, width, drones = 8, 25, 4
        source, target = model(200, 5, [16, 12]), model(800, 20, [64, 48])
        transfer.transfer_policy(source.policy, target.policy, observation_dim=width,
                                 frame_stack=frames, num_drones=drones, value_scale=.25)
        x = np.random.default_rng(39).normal(size=(32, frames, drones, width)).astype(np.float32)
        x[:8, :-1] = 0.
        own = torch.from_numpy(x.transpose(0, 2, 1, 3).reshape(-1, 200))
        joint = torch.from_numpy(x.reshape(-1, 800))
        with torch.no_grad():
            expected = source.policy.get_distribution(own).distribution.mean.reshape(32, 20)
            actual = target.policy.get_distribution(joint).distribution.mean
            expected_value = source.policy.predict_values(own).reshape(32, 4).mean(1)
            actual_value = target.policy.predict_values(joint).flatten()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_value, expected_value)
        torch.testing.assert_close(target.policy.log_std, source.policy.log_std.repeat(4))
        self.assertEqual(len(target.policy.optimizer.state), 0)
        # A saved artifact must not depend on a custom policy class or live simulator.
        (REPO / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=REPO / "outputs") as directory:
            target.save(Path(directory) / "model.zip")
            loaded = PPO.load(Path(directory) / "model.zip", device="cpu")
            np.testing.assert_array_equal(target.predict(joint.numpy(), deterministic=True)[0],
                                          loaded.predict(joint.numpy(), deterministic=True)[0])

    def test_cross_drone_connections_can_learn(self):
        source, target = model(6, 2, [8, 8]), model(24, 8, [32, 32])
        transfer.transfer_policy(source.policy, target.policy, observation_dim=3,
                                 frame_stack=2, num_drones=4)
        target.policy.optimizer.zero_grad()
        means = target.policy.get_distribution(torch.randn(10, 24)).distribution.mean
        means[:, :2].sum().backward()
        first = target.policy.mlp_extractor.policy_net[0]
        # Drone 1's first frame occupies columns 3:6; outputs 0:2 belong to drone 0.
        self.assertGreater(float(first.weight.grad[:8, 3:6].abs().sum()), 0.)

    def test_normalization_and_reward_units_survive_roundtrip(self):
        old = VecNormalize(DummyVecEnv([lambda: Spaces(3, 2)]), norm_reward=True, clip_reward=np.inf)
        old.obs_rms.mean = np.array([1., 2., 3.])
        old.obs_rms.var = np.array([4., 9., 16.])
        old.obs_rms.count = 1234.
        old.ret_rms.var = np.array(11.)
        base = DummyVecEnv([lambda: Spaces(12, 8)])
        new, scale = transfer.expand_normalizer(old, base, 4)
        x = np.random.default_rng(3).normal(size=(10, 4, 3))
        np.testing.assert_array_equal(new.normalize_obs(x.reshape(10, 12)),
                                      old.normalize_obs(x.reshape(40, 3)).reshape(10, 12))
        rewards = np.arange(40).reshape(10, 4)
        np.testing.assert_allclose(new.normalize_reward(rewards.sum(1)),
                                   old.normalize_reward(rewards).mean(1), rtol=1e-6)
        self.assertEqual(scale, .25)
        self.assertFalse(new.training)
        self.assertEqual(new.obs_rms.count, old.obs_rms.count)
        with tempfile.TemporaryDirectory(dir=REPO / "outputs") as directory:
            path = Path(directory) / "norm.pkl"
            new.save(path)
            loaded = VecNormalize.load(path, base)
            np.testing.assert_array_equal(loaded.normalize_obs(x.reshape(10, 12)),
                                          new.normalize_obs(x.reshape(10, 12)))
        old.close()
        new.close()

    def test_incompatible_layout_is_rejected(self):
        source, target = model(6, 2, [8, 8]), model(24, 8, [32, 32])
        with self.assertRaises(ValueError):
            transfer.transfer_policy(source.policy, target.policy, observation_dim=3,
                                     frame_stack=3, num_drones=4)

    def test_critic_warmup_preserves_actor_then_restores_learning(self):
        env = gym.make("Pendulum-v1")
        agent = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=2,
                    learning_rate=.001, seed=9, device="cpu")
        actor = (list(agent.policy.mlp_extractor.policy_net.parameters())
                 + list(agent.policy.action_net.parameters()) + [agent.policy.log_std])
        before = [p.detach().clone() for p in actor]
        value_before = agent.policy.value_net.weight.detach().clone()
        agent.learn(8, callback=transfer.CriticWarmupCallback(1))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, actor)))
        self.assertFalse(torch.equal(value_before, agent.policy.value_net.weight))
        self.assertTrue(all(p.requires_grad for p in actor))
        agent.learn(8)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, actor)))
        env.close()


if __name__ == "__main__":
    unittest.main()

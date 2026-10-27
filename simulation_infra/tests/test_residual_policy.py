"""Command preservation, usable braking gradients, frozen pilot, and portable artifacts."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cloudpickle
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "drone_deployment_infra"))
from core import residual_policy

spec = importlib.util.spec_from_file_location(
    "residual_warmup_test", REPO / "simulation_infra/source/action_space_kit/action_space_kit/rl/team_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class FlightSpaces(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (24,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.np_random.normal(size=24).astype(np.float32), {}

    def step(self, action):
        return self.np_random.normal(size=24).astype(np.float32), -float((action ** 2).sum()), False, False, {}


def models(bound_mean=False):
    cfg = dict(n_steps=8, batch_size=8, n_epochs=2, learning_rate=.001, seed=3, device="cpu")
    source = PPO("MlpPolicy", FlightSpaces(), policy_kwargs={"net_arch": [16, 12], "activation_fn": torch.nn.ELU}, **cfg)
    target = PPO(residual_policy.ResidualActorCriticPolicy, FlightSpaces(),
                 policy_kwargs={"net_arch": [16, 12], "activation_fn": torch.nn.ELU,
                                "residual_width": 8, "bound_mean": bound_mean}, **cfg)
    with torch.no_grad():
        source.policy.action_net.weight.zero_()
        source.policy.action_net.bias.copy_(torch.tensor([.2, 2.3, -.1, .4, -.8]))
        source.policy.log_std.fill_(np.log(.1))
    residual_policy.copy_learned_pilot(source.policy, target.policy)
    return source, target


class ResidualPolicyTests(unittest.TestCase):
    def test_commands_match_but_braking_can_be_explored_and_learned(self):
        source, target = models()
        x = torch.randn(256, 24)
        np.testing.assert_array_equal(source.predict(x.numpy(), deterministic=True)[0],
                                      target.predict(x.numpy(), deterministic=True)[0])
        torch.testing.assert_close(source.policy.predict_values(x), target.policy.predict_values(x))
        old = source.policy.get_distribution(x).distribution
        new = target.policy.get_distribution(x).distribution
        self.assertLess(float(old.cdf(torch.ones_like(x[:, :5]))[:, 1].max()), 1e-6)
        self.assertAlmostEqual(float(new.cdf(torch.ones_like(x[:, :5]))[:, 1].mean()), .5)
        base_before = {k: v.clone() for k, v in target.policy.action_net.base.state_dict().items()}
        target.policy.optimizer.zero_grad()
        new.mean[:, 1].mean().backward()
        self.assertGreater(float(target.policy.action_net.correction.bias.grad[1]), 0.)
        target.policy.optimizer.step()
        self.assertLess(float(target.policy.get_distribution(x).distribution.mean[:, 1].mean()), 1.)
        for name, value in target.policy.action_net.base.state_dict().items():
            torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in target.policy.action_net.base.parameters()))

    def test_warmup_freezes_correction_without_unfreezing_pilot(self):
        _, target = models()
        actor = target.policy.actor_parameters()
        before = [p.detach().clone() for p in actor]
        target.learn(8, callback=transfer.CriticWarmupCallback(1))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, actor)))
        self.assertTrue(all(p.requires_grad for p in actor))
        self.assertTrue(all(not p.requires_grad for p in target.policy.mlp_extractor.policy_net.parameters()))
        self.assertTrue(all(not p.requires_grad for p in target.policy.action_net.base.parameters()))
        target.learn(8)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, actor)))

    def test_bounded_migration_retains_weights_and_critic_but_explores_braking(self):
        _, source = models()
        _, target = models(bound_mean=True)
        with torch.no_grad():
            source.policy.action_net.correction.bias[1] = .7
        residual_policy.copy_residual_state(source.policy, target.policy)
        for key, value in source.policy.state_dict().items():
            torch.testing.assert_close(target.policy.state_dict()[key], value, rtol=0, atol=0)
        x = torch.randn(64, 24)
        old = source.policy.get_distribution(x).distribution
        new = target.policy.get_distribution(x).distribution
        torch.testing.assert_close(new.mean, old.mean.tanh(), rtol=0, atol=0)
        torch.testing.assert_close(source.policy.predict_values(x), target.policy.predict_values(x), rtol=0, atol=0)
        self.assertTrue(bool((new.mean.abs() <= 1.).all()))
        self.assertTrue(bool((new.cdf(torch.ones_like(new.mean)) >= .5).all()))
        target.policy.optimizer.zero_grad()
        new.mean[:, 1].mean().backward()
        self.assertGreater(float(target.policy.action_net.correction.bias.grad[1]), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in target.policy.action_net.base.parameters()))

    def test_isolated_reload_resave_and_torchscript_export(self):
        cloudpickle.register_pickle_by_value(residual_policy)
        root = REPO / "outputs"
        root.mkdir(exist_ok=True)
        probe = """
import sys, torch, numpy as np
from stable_baselines3 import PPO
torch.set_num_threads(1)
m=PPO.load(sys.argv[1],device='cpu')
x=torch.zeros(4,24)
a=m.predict(x.numpy(),deterministic=True)[0]
expected=np.array([.2,1.,-.1,.4,-.8])
if sys.argv[3]=='1': expected=np.tanh(expected)
assert np.allclose(a,expected)
assert not any(p.requires_grad for p in m.policy.action_net.base.parameters())
m.save(sys.argv[2])
m=PPO.load(sys.argv[2],device='cpu')
class Export(torch.nn.Module):
 def __init__(self):
  super().__init__();self.policy=m.policy
 def forward(self,x):
  return self.policy._predict(x,deterministic=True).clamp(-1.,1.)
traced=torch.jit.trace(Export(),x)
torch.testing.assert_close(traced(x),torch.tensor(a))
print('ISOLATED RESIDUAL RELOAD AND EXPORT PASSED')
"""
        for bounded in (False, True):
            with self.subTest(bounded=bounded), tempfile.TemporaryDirectory(dir=root) as directory:
                _, target = models(bound_mean=bounded)
                path = Path(directory) / "model.zip"
                target.save(path)
                result = subprocess.run([sys.executable, "-I", "-c", probe, str(path),
                                         str(Path(directory)/"resaved.zip"), str(int(bounded))],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("ISOLATED RESIDUAL RELOAD AND EXPORT PASSED", result.stdout)


if __name__ == "__main__":
    unittest.main()

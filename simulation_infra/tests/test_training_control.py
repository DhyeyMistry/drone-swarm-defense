"""A signal inside a simulator-style callback must still stop PPO and allow saving."""

import importlib.util
import os
import signal
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "training_control_test", REPO / "simulation_infra/source/action_space_kit/action_space_kit/rl/training_control.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class SignalDuringStep(gym.Wrapper):
    def step(self, action):
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt:
            pass  # emulate the physics engine swallowing callback exceptions
        return self.env.step(action)


class TrainingControlTests(unittest.TestCase):
    def test_signal_inside_step_stops_before_updates_and_preserves_reloadable_checkpoint(self):
        stop = control.GracefulStopCallback()
        previous = signal.signal(signal.SIGINT, stop.request_stop)
        env = SignalDuringStep(gym.make("Pendulum-v1"))
        try:
            model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=5, device="cpu")
            model.learn(64, callback=stop)
            self.assertTrue(stop.requested)
            self.assertEqual(model.num_timesteps, 1)
            self.assertEqual(model._n_updates, 0)
            (REPO / "outputs").mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=REPO / "outputs") as directory:
                path = Path(directory) / "stopped.zip"
                model.save(path)
                self.assertEqual(PPO.load(path, device="cpu").num_timesteps, 1)
        finally:
            signal.signal(signal.SIGINT, previous)
            env.close()


if __name__ == "__main__":
    unittest.main()

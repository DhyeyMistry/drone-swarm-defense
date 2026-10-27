"""Estimated peer observations and anticipatory separation reward, without Isaac."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "team_features_test", Path(__file__).resolve().parents[2] / "drone_deployment_infra/core/team_features.py")
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


class PeerFeaturesTests(unittest.TestCase):
    def test_symmetric_starts_keep_neighbor_slots_across_heading_and_backends(self):
        # Drone 2 is equally close to drones 1 and 3. Sorting rotated floats used
        # to swap these neighbors between NumPy and CUDA for some headings.
        p = np.tile(np.array([[6.3, -.825, .025], [6.7, -.275, .025],
                             [6.3, .275, .025], [6.7, .825, .025]], np.float32), (100, 1, 1))
        v = np.zeros_like(p)
        v[..., 2] = np.arange(4)
        yaw = np.linspace(-np.pi, np.pi, 400, dtype=np.float32).reshape(100, 4)
        variance = np.full_like(p, .01)
        expected = features.peer_features(p, v, yaw, variance, variance, np,
                                          sort_neighbors=True, order_version=2)
        np.testing.assert_array_equal(expected[:, 2, 10], -1.)
        np.testing.assert_array_equal(expected[:, 2, 16], 1.)
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            actual = features.peer_features(*(torch.tensor(x, device=device) for x in
                                              (p, v, yaw, variance, variance)), torch,
                                            sort_neighbors=True, order_version=2).cpu().numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)

    def test_nearest_first_features_do_not_depend_on_drone_labels(self):
        rng = np.random.default_rng(88)
        p, v = rng.normal(size=(8, 4, 3)), rng.normal(size=(8, 4, 3))
        yaw = rng.normal(size=(8, 4))
        pv, vv = np.full_like(p, .01), np.full_like(v, .04)
        a = features.peer_features(p, v, yaw, pv, vv, np, sort_neighbors=True)
        perm = [2, 0, 3, 1]
        b = features.peer_features(p[:, perm], v[:, perm], yaw[:, perm], pv[:, perm], vv[:, perm], np,
                                  sort_neighbors=True)
        np.testing.assert_allclose(a[:, perm], b, atol=1e-12)
        c = features.peer_features(*(torch.tensor(x) for x in (p, v, yaw, pv, vv)), torch,
                                  sort_neighbors=True).numpy()
        np.testing.assert_allclose(a, c, atol=1e-12)

    def test_numpy_torch_and_translation_parity(self):
        rng = np.random.default_rng(6)
        p, v = rng.normal(size=(2, 3, 4, 3)), rng.normal(size=(2, 3, 4, 3))
        yaw = rng.normal(size=(2, 3, 4))
        pv, vv = np.full_like(p, .04), np.full_like(v, .09)
        a = features.peer_features(p, v, yaw, pv, vv, np)
        b = features.peer_features(*(torch.tensor(x) for x in (p, v, yaw, pv, vv)), torch).numpy()
        self.assertEqual(a.shape, (2, 3, 4, 23))
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(a, features.peer_features(p + [20., -10., 7.], v, yaw, pv, vv, np), atol=1e-12)
        np.testing.assert_allclose(a[..., 3:5], np.broadcast_to([.2, .3], a[..., 3:5].shape))

    def test_neighbor_order_and_heading_axes(self):
        p = np.array([[0., 0., 0.], [1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        v = np.zeros_like(p)
        yaw = np.array([0., np.pi/2, 0., 0.])
        a = features.peer_features(p, v, yaw, v, v, np)
        np.testing.assert_allclose(a[0, 5:8], [1., -2., 3.])
        np.testing.assert_allclose(a[0, 11:14], [4., -5., 6.])
        np.testing.assert_allclose(a[1, 5:8], [-2., -1., -3.], atol=1e-12)

    def test_predictive_cost_warns_before_contact_and_respects_direction(self):
        p = torch.tensor([[[0., 0., 1.], [2., 0., 1.]]])
        v = torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]])
        landed = torch.zeros((1, 2), dtype=torch.bool)
        cost = features.predicted_separation_cost(p, v, landed, 1.5, .5, torch)
        torch.testing.assert_close(cost, torch.full((1, 2), np.exp(-1/1.5)))
        self.assertEqual(float(features.predicted_separation_cost(p, -v, landed, 1.5, .5, torch).sum()), 0.)
        self.assertEqual(float(features.predicted_separation_cost(p, v, landed, .2, .5, torch).sum()), 0.)

    def test_stationary_pads_and_landed_pairs_have_no_cost(self):
        p = torch.tensor([[[0., 0., 1.], [0., .5, 1.], [0., 1., 1.], [0., 1.5, 1.]]])
        v = torch.zeros_like(p)
        landed = torch.zeros((1, 4), dtype=torch.bool)
        self.assertEqual(float(features.predicted_separation_cost(p, v, landed, 1.5, .5, torch).sum()), 0.)
        self.assertEqual(float(features.predicted_separation_cost(p * 0, v, ~landed, 1.5, .5, torch).sum()), 0.)
        self.assertGreater(float(features.predicted_separation_cost(p * 0, v, landed, 1.5, .5, torch).sum()), 0.)


if __name__ == "__main__":
    unittest.main()

"""Prevent incomplete or weakened evidence from being called 99% joint reliability."""

import copy
import importlib.util
import itertools
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "source/action_space_kit/action_space_kit/rl/qualification.py"
spec = importlib.util.spec_from_file_location("room_qualification_test", path)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def fixtures():
    env = dict(spawn_range=[6., 7.], rest_height=.025, episode_length_s=75., success_radius=.3,
               collision_distance=.25, spot_distance=1., spot_spacing=.5)
    assignments = list(itertools.permutations(range(4)))
    reports = []
    for seed in q.HELD_OUT_SEEDS:
        records = []
        for i in range(256):
            order = assignments[i % 24]
            metrics = {"Episode/" + key: value for key, value in dict(
                success_rate=1., collision_rate=0., crash_rate=0., timeout_rate=0.,
                duration_s=30., min_separation_m=.4).items()}
            records.append({"env_id": i, "metrics": metrics, "start_kind": "configured",
                            "start_positions": [[6.2, y, .025] for y in (-.9, -.3, .3, .9)],
                            "drones": [{"target_index": j, "target": [1., .5*(j-1.5), 0.],
                                        "landed": True, "land_ok": True, "touchdown_error_m": .1} for j in order]})
        reports.append({"seed": seed, "checkpoint_sha256": "model", "normalizer_sha256": "normalizer",
                        "pilot": "policy", "config_source": "checkpoint", "overrides": ["env.collision_substeps=True"],
                        "summary": {"complete": True, "episodes": 256, "quotas": [1]*256, "counts": [1]*256,
                                    "metrics": records[0]["metrics"].copy()}, "records": records})
    return reports, env


class QualificationTests(unittest.TestCase):
    def test_changed_landing_gate_or_controller_is_rejected(self):
        baseline = {"env": {"num_drones": 1, "sim": {"dt": .002},
                    "dynamics": {"domain_randomization": True}, "land_dwell_s": .5,
                    "land_max_xy_error": .15, "auto_land": True, "decimation": 37,
                    "rc": {"fit_path": "/original/motion_fit.yaml", "max_rc": 100}},
                    "assets": {k: {"sha256": k} for k in ("motion_fit", "room_map", "room_tags")}}
        candidate = copy.deepcopy(baseline)
        candidate["env"].update(num_drones=4, rew_collision=-2000., peer_order_version=2)
        candidate["env"]["rc"]["fit_path"] = "/candidate/motion_fit.yaml"
        q.validate_challenge(candidate, baseline)
        changes = [lambda e: e.update(land_dwell_s=0.),
                   lambda e: e.update(land_max_xy_error=1.),
                   lambda e: e.update(auto_land=False),
                   lambda e: e.update(decimation=20),
                   lambda e: e["rc"].update(max_rc=50)]
        for change in changes:
            bad = copy.deepcopy(candidate)
            change(bad["env"])
            with self.assertRaises(ValueError):
                q.validate_challenge(bad, baseline)

    def test_complete_cohort_and_exact_target_boundary(self):
        reports, env = fixtures()
        for i in range(10):
            row = reports[0]["records"][i]
            row["metrics"]["Episode/success_rate"] = 0.
            row["metrics"]["Episode/collision_rate"] = 1.
        reports[0]["summary"]["metrics"].update({"Episode/success_rate": 246/256, "Episode/collision_rate": 10/256})
        result = q.audit_reports(reports, env, "model", "normalizer")
        self.assertTrue(result["qualification_passed"])
        self.assertEqual(result["all_four_successes"], 1014)
        reports[0]["records"][10]["metrics"].update({"Episode/success_rate": 0., "Episode/collision_rate": 1.})
        reports[0]["summary"]["metrics"].update({"Episode/success_rate": 245/256, "Episode/collision_rate": 11/256})
        self.assertFalse(q.audit_reports(reports, env, "model", "normalizer")["qualification_passed"])

    def test_invalid_evidence_is_rejected(self):
        reports, env = fixtures()
        changes = [lambda r: r[0]["summary"].update(complete=False),
                   lambda r: r[0]["records"].pop(),
                   lambda r: r[0].update(seed=r[1]["seed"]),
                   lambda r: r[0].update(checkpoint_sha256="different"),
                   lambda r: r[0].update(overrides=[]),
                   lambda r: r[0]["records"][0]["metrics"].update({"Episode/collision_rate": 1.}),
                   lambda r: r[0]["records"][0]["drones"][0].update(touchdown_error_m=.4),
                   lambda r: r[0]["records"][0]["drones"][0].update(target=[6., -.75, 0.]),
                   lambda r: r[0]["records"][0].update(start_kind="hover_landing")]
        for change in changes:
            bad = copy.deepcopy(reports)
            change(bad)
            with self.assertRaises(ValueError):
                q.audit_reports(bad, env, "model", "normalizer")


if __name__ == "__main__":
    unittest.main()

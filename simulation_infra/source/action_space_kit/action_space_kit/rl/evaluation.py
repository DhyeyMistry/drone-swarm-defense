"""Episode-balanced evaluation, independent of simulator and training dependencies."""

import copy
import math


class EpisodeCohort:
    """Take a fixed quota from each simulator slot, never favoring faster episodes."""

    def __init__(self, num_envs, episodes, retain_records=True):
        if num_envs < 1 or episodes < 1:
            raise ValueError("num_envs and episodes must be positive")
        self.quotas = [episodes // num_envs + (i < episodes % num_envs) for i in range(num_envs)]
        self.counts = [0] * num_envs
        self._last_ids = [-1] * num_envs
        self.retain_records = retain_records
        self._metric_sums = {}
        self._metric_counts = {}
        self._targets = {}
        self.records = []

    def add(self, record):
        env_id = int(record["env_id"])
        episode_id = int(record["episode_id"])
        if episode_id <= self._last_ids[env_id]:
            raise ValueError(f"Duplicate or out-of-order episode {(env_id, episode_id)}")
        self._last_ids[env_id] = episode_id
        if self.counts[env_id] >= self.quotas[env_id]:
            return False
        self.counts[env_id] += 1
        if self.retain_records:
            self.records.append(record)
        for key, value in record.get("metrics", {}).items():
            self._metric_sums[key] = self._metric_sums.get(key, 0.0) + value
            self._metric_counts[key] = self._metric_counts.get(key, 0) + 1
        for drone in record.get("drones", []):
            key = str(drone["target_index"])
            row = self._targets.setdefault(key, {"episodes": 0, "successes": 0, "touchdown_errors_m": []})
            row["episodes"] += 1
            row["successes"] += int(drone["land_ok"])
            if drone["landed"] and self.retain_records:
                row["touchdown_errors_m"].append(drone["touchdown_error_m"])
        return True

    @property
    def complete(self):
        return self.counts == self.quotas

    def summary(self):
        metrics = {key: value / self._metric_counts[key] for key, value in sorted(self._metric_sums.items())}
        targets = copy.deepcopy(self._targets)
        for row in targets.values():
            n, k = row["episodes"], row["successes"]
            p, z = k / n, 1.96
            den = 1 + z * z / n
            center = (p + z * z / (2 * n)) / den
            radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            row.update(success_rate=p, wilson_95=[center - radius, center + radius])
        return {
            "episodes": sum(self.counts),
            "complete": self.complete,
            "quotas": self.quotas,
            "counts": self.counts,
            "metrics": metrics,
            "targets": targets,
        }

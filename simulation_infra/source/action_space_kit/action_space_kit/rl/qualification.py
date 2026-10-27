"""Audit complete four-drone evaluation cohorts against the unchanged room challenge."""

import math
import statistics

HELD_OUT_SEEDS = (17291, 41423, 89519, 130073)
CHALLENGE_FIELDS = (
    "episode_length_s", "spawn_box_center", "spawn_box_size", "spawn_range", "spawn_min_separation",
    "spawn_near_target_radius", "spawn_short_fraction", "spawn_hover_fraction", "start_on_floor",
    "takeoff_at_reset", "rest_height", "takeoff_height", "takeoff_speed", "spawn_yaw_noise",
    "spot_distance", "spot_spacing", "spot_count", "random_assignment", "fixed_spot_index",
    "success_radius", "collision_distance", "waypoint_height", "command_rate_hz", "command_jitter",
    "tag10_height", "wall_x", "side_wall_y", "room_y_min", "room_x_max", "ceiling", "wall_clearance",
    "min_height", "max_tilt", "land_speed", "touchdown_height", "lost_after", "yaw_fix_noise",
    "height_noise", "fix_noise", "fix_bias", "link_loss_range", "tag_sensor", "dynamics", "kalman",
    "decimation", "waypoint_radius", "waypoint_max_speed", "speed_trust_std", "terminate_fast_entry",
    "max_fix_age", "land_threshold", "land_confirm_steps", "auto_land", "land_gate_enabled",
    "land_max_xy_error", "land_max_horizontal_speed", "land_height_tolerance", "land_max_position_std",
    "land_max_velocity_std", "land_max_observation_age", "land_dwell_s",
)


def validate_challenge(candidate, reference):
    env, baseline = candidate["env"], reference["env"]
    if env["num_drones"] != 4:
        raise ValueError("Qualification requires four drones in the same room")
    different = [key for key in CHALLENGE_FIELDS if env.get(key) != baseline.get(key)]
    if different:
        raise ValueError(f"Challenge differs from the preserved full-room reference: {different}")
    if env["sim"]["dt"] != baseline["sim"]["dt"]:
        raise ValueError("Physics step changed")
    # The fit is copied into each run, so compare its content below and retain all
    # other controller settings here. A different gate or controller is a different task.
    controller = lambda cfg: {k: v for k, v in cfg.get("rc", {}).items() if k != "fit_path"}
    if controller(env) != controller(baseline):
        raise ValueError("RC controller settings changed")
    for name in ("motion_fit", "room_map", "room_tags"):
        if candidate["assets"][name]["sha256"] != reference["assets"][name]["sha256"]:
            raise ValueError(f"Challenge asset changed: {name}")
    if not env["dynamics"]["domain_randomization"]:
        raise ValueError("Domain randomization is required")


def wilson_interval(successes, count):
    p, z = successes / count, 1.96
    den = 1 + z*z/count
    center = (p + z*z/(2*count)) / den
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / den
    return [center-half, center+half]


def audit_reports(reports, env, checkpoint_hash, normalizer_hash, seeds=HELD_OUT_SEEDS, episodes_per_seed=256):
    """Recompute success from individual records; reject incomplete or incomparable evidence."""
    if len(reports) != len(seeds) or sorted(r["seed"] for r in reports) != sorted(seeds):
        raise ValueError("Missing, duplicate, or unexpected evaluation seeds")
    records = []
    for report in reports:
        summary = report["summary"]
        if (not summary["complete"] or summary["episodes"] != episodes_per_seed
                or len(report["records"]) != episodes_per_seed
                or summary["quotas"] != [1] * episodes_per_seed
                or summary["counts"] != summary["quotas"]):
            raise ValueError("Qualification needs one complete episode from every evaluation room")
        if sorted(row["env_id"] for row in report["records"]) != list(range(episodes_per_seed)):
            raise ValueError("Evaluation room coverage is incomplete or duplicated")
        if (report["checkpoint_sha256"] != checkpoint_hash or report["normalizer_sha256"] != normalizer_hash
                or report["pilot"] != "policy" or report["config_source"] != "checkpoint"
                or report["overrides"] != ["env.collision_substeps=True"]):
            raise ValueError("Mixed checkpoints or changed evaluation settings")
        for row in report["records"]:
            drones, metrics = row["drones"], row["metrics"]
            if len(drones) != 4 or sorted(d["target_index"] for d in drones) != [0, 1, 2, 3]:
                raise ValueError("Every trial must assign four distinct landing pads")
            for drone in drones:
                expected = [env["spot_distance"], env["spot_spacing"] * (drone["target_index"] - 1.5), 0.]
                if len(drone["target"]) != 3 or any(abs(a-b) > 1e-5 for a, b in zip(drone["target"], expected)):
                    raise ValueError("Assigned pad location changed")
            if row["start_kind"] != "configured" or len(row["start_positions"]) != 4:
                raise ValueError("Short or hovering start in full-room qualification")
            for p in row["start_positions"]:
                radius = math.hypot(p[0], p[1])
                if not env["spawn_range"][0] - .001 <= radius <= env["spawn_range"][1] + .001:
                    raise ValueError("Start outside the full-room range")
                if abs(p[2] - env["rest_height"]) > .001:
                    raise ValueError("Trial did not start on the floor")
            for key in ("success_rate", "collision_rate", "crash_rate", "timeout_rate"):
                if metrics["Episode/" + key] not in (0., 1.):
                    raise ValueError(f"Invalid per-episode indicator: {key}")
            if not 0. < metrics["Episode/duration_s"] <= env["episode_length_s"] + .15:
                raise ValueError("Invalid episode duration")
            if metrics["Episode/success_rate"]:
                valid = all(d["landed"] and d["land_ok"] and 0 <= d["touchdown_error_m"] < env["success_radius"]
                            for d in drones)
                clean = not any(metrics["Episode/" + k] for k in ("collision_rate", "crash_rate", "timeout_rate"))
                separated = metrics["Episode/min_separation_m"] >= env["collision_distance"] - 1e-5
                if not (valid and clean and separated):
                    raise ValueError("A claimed success violates touchdown or collision requirements")
        for key in ("success_rate", "collision_rate", "crash_rate", "timeout_rate"):
            observed = sum(r["metrics"]["Episode/" + key] for r in report["records"]) / episodes_per_seed
            if not math.isclose(observed, summary["metrics"]["Episode/" + key], abs_tol=1e-12):
                raise ValueError("Summary does not match individual episode records")
        records.extend(report["records"])
    count = len(records)
    assignments = {tuple(d["target_index"] for d in row["drones"]) for row in records}
    if count >= 1024 and len(assignments) != 24:
        raise ValueError("Qualification did not cover all 24 drone-to-pad assignments")
    counts = {key: int(sum(r["metrics"]["Episode/" + key] for r in records))
              for key in ("success_rate", "collision_rate", "crash_rate", "timeout_rate")}
    successful_times = [r["metrics"]["Episode/duration_s"] for r in records if r["metrics"]["Episode/success_rate"]]
    rate = counts["success_rate"] / count
    return {"episodes": count, "seeds": list(seeds), "distinct_assignments": len(assignments), "all_four_successes": counts["success_rate"],
            "joint_success_rate": rate, "joint_success_wilson_95": wilson_interval(counts["success_rate"], count),
            "collision_failures": counts["collision_rate"], "crash_failures": counts["crash_rate"],
            "timeouts": counts["timeout_rate"], "success_mean_duration_s": statistics.mean(successful_times) if successful_times else None,
            "qualification_passed": count >= 1024 and rate >= .99,
            "qualification": "Observed simulation reliability; does not establish real-flight reliability."}

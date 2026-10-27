"""Portable run configuration and provenance. Loading config never unpickles Python objects."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tarfile
from pathlib import Path

import yaml

_SKIP = {"action_spaces", "observation_spaces", "state_space", "ui_window_class_type", "log_dir"}
_MISSING = object()


def _plain(value):
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        values = [_plain(v) for v in value]
        return values if all(v is not _MISSING for v in values) else _MISSING
    if isinstance(value, dict):
        return {k: p for k, v in value.items() if k not in _SKIP and (p := _plain(v)) is not _MISSING}
    return _MISSING


def _apply(obj, values):
    for key, value in values.items():
        if key in _SKIP or not hasattr(obj, key):
            continue
        old = getattr(obj, key)
        if isinstance(value, dict) and hasattr(old, "__dict__"):
            _apply(old, value)
        elif not callable(old):
            setattr(obj, key, tuple(value) if isinstance(old, tuple) and isinstance(value, list) else value)


class _LegacyLoader(yaml.SafeLoader):
    pass


_LegacyLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
# Ignore serialized Box/ndarray objects; dimensions come from the registered task and PPO checks them.
_LegacyLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def _override_values(env_cfg, agent_cfg, overrides):
    """Hydra already parsed overrides. Preserve those exact values while restoring a run."""
    values = []
    for arg in overrides:
        path = arg.split("=", 1)[0].lstrip("+").split(".")
        if len(path) < 2 or path[0] not in ("env", "agent"):
            continue
        obj = env_cfg if path[0] == "env" else agent_cfg
        for key in path[1:]:
            obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
        values.append((path, obj))
    return values


def restore_run_config(checkpoint, env_cfg, agent_cfg, overrides=()):
    root = Path(checkpoint).resolve().parent
    explicit = _override_values(env_cfg, agent_cfg, overrides)
    saved = root / "params/run.json"
    if saved.exists():
        data = json.loads(saved.read_text())
        config, agent = data["env"], data["agent"]
        for key, asset in data.get("assets", {}).items():
            path = root / asset["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"Snapshot asset hash mismatch: {path}")
            if key == "motion_fit":
                config["rc"]["fit_path"] = str(path)
            elif key in ("room_map", "room_tags"):
                config[key + "_path"] = str(path)
    elif (root / "params/env.yaml").exists():
        config = yaml.load((root / "params/env.yaml").read_text(), Loader=_LegacyLoader)
        agent = yaml.safe_load((root / "params/agent.yaml").read_text())
        # Old runs predate versioned rewards/gates. Don't silently score them on new defaults.
        config.setdefault("reward_version", 1)
        config.setdefault("land_gate_enabled", False)
        config.setdefault("terminate_fast_entry", "waypoint_max_speed" in config)
        if "waypoint_max_speed" not in config:
            config["waypoint_max_speed"] = 1e6
            config["speed_trust_std"] = 1e6
        if "tag_sensor" in config:
            config["tag_sensor"].setdefault("blur_max_loss", 0.0)
        config.setdefault("auto_land", False)
        print(
            "[config] LEGACY run: external map/fit and code were not snapshotted; "
            "exact historical replay cannot be guaranteed."
        )
        # Legacy fit paths point at another checkout/machine. Use the current fit explicitly.
        if "rc" in config:
            config["rc"]["fit_path"] = env_cfg.rc.fit_path
    else:
        raise FileNotFoundError(f"No saved config beside {checkpoint}; use --config-source current explicitly.")
    if hasattr(env_cfg, "num_drones") and config.get("num_drones", env_cfg.num_drones) != env_cfg.num_drones:
        raise ValueError(
            "Saved task has a different drone count; a central policy cannot change its input/output layout"
        )
    # Older runs checked proximity only once per RC update. Preserve replay unless
    # evaluation/training explicitly opts into physics-substep collision checks.
    config.setdefault("collision_substeps", False)
    config.setdefault("peer_order_version", 1)  # retain historical nearest-neighbor slot ordering
    _apply(env_cfg, config)
    agent_cfg.update(agent)
    for path, value in explicit:
        obj = env_cfg if path[0] == "env" else agent_cfg
        for key in path[1:-1]:
            obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
        if isinstance(obj, dict):
            obj[path[-1]] = value
        else:
            setattr(obj, path[-1], value)
    print(f"[config] Restored {root}; explicit overrides: {list(overrides)}")


def snapshot_run(log_dir, env_cfg, agent_cfg, task, argv):
    root = Path(log_dir)
    params = root / "params"
    params.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[5]
    config = _plain(env_cfg.to_dict())
    assets = {}
    if hasattr(env_cfg, "rc"):
        deploy = Path(os.environ.get("AS_DEPLOY_CONFIG", repo / "drone_deployment_infra/config"))
        paths = {"motion_fit": Path(env_cfg.rc.fit_path)}
        if hasattr(env_cfg, "room_map_path"):
            paths.update(
                room_map=Path(env_cfg.room_map_path or deploy / "map.yaml"),
                room_tags=Path(env_cfg.room_tags_path or deploy / "tags.yaml"),
            )
        for name, path in paths.items():
            dest = root / "assets" / (name + ".yaml")
            dest.parent.mkdir(exist_ok=True)
            content = path.read_bytes()
            dest.write_bytes(content)
            assets[name] = {"path": str(dest.relative_to(root)), "sha256": hashlib.sha256(content).hexdigest()}

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True)

    source = {}
    for name in git("ls-files").splitlines():
        path = repo / name
        if path.suffix in (".py", ".yaml", ".toml") and path.is_file():
            source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Include newly added modules, which are not in ls-files before the fixes are committed.
    for name in git("ls-files", "--others", "--exclude-standard").splitlines():
        path = repo / name
        if path.suffix in (".py", ".yaml", ".toml") and path.is_file():
            source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = {
        "schema": 1,
        "task": task,
        "env": config,
        "agent": _plain(agent_cfg),
        "assets": assets,
        "git_commit": git("rev-parse", "HEAD").strip(),
        "source_sha256": source,
        "argv": list(argv),
        "dependencies": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]
        },
    }
    (params / "run.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (params / "working.diff").write_text(git("diff", "HEAD"))
    with tarfile.open(params / "source.tar.gz", "w:gz") as archive:
        for name in sorted(source):
            archive.add(repo / name, arcname=name)
    return data

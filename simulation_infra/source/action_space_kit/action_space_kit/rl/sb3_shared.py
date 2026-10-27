"""Stable-Baselines3 wrappers for MARL envs: a shared per-agent policy, or one central policy.

``Sb3CentralVecEnv`` (agent cfg ``central: true``) is one controller for all agents: one row
per env, all observations in, all actions out, team reward. The rest of this note describes
the shared per-agent wrapper.


Isaac Lab's ``Sb3VecEnvWrapper`` only takes single-agent envs, and its
``multi_agent_to_single_agent`` concatenates all agents into one centralized policy (16-D
action for four drones), which no single real drone could run. Here each agent of each
env becomes its own SB3 env row instead: row = env * num_agents + agent. Every drone then
runs the same policy on its own observation, which is what gets flown on the Tellos.

All agents of an env share its reset, so a row is done exactly when its env is.

Episode statistics the env publishes in ``extras['log']`` are collected and written to the
SB3 logger by :class:`LogExtrasCallback`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from isaaclab.envs import DirectMARLEnv


class Sb3SharedPolicyVecEnv(VecEnv):
    """Flatten a :class:`DirectMARLEnv` with homogeneous agents into num_envs * num_agents SB3 rows."""

    def __init__(self, env: gym.Env):
        if not isinstance(env.unwrapped, DirectMARLEnv):
            raise ValueError(f"Expected a DirectMARLEnv, got {type(env.unwrapped)}")
        self.env = env
        self.unwrapped_env: DirectMARLEnv = env.unwrapped
        self.agents = list(self.unwrapped_env.possible_agents)
        self.num_agents = len(self.agents)
        self.num_isaac_envs = self.unwrapped_env.num_envs
        self.sim_device = self.unwrapped_env.device

        obs_spaces = [self.unwrapped_env.observation_spaces[a] for a in self.agents]
        act_spaces = [self.unwrapped_env.action_spaces[a] for a in self.agents]
        if any(s.shape != obs_spaces[0].shape for s in obs_spaces) or any(
            s.shape != act_spaces[0].shape for s in act_spaces
        ):
            raise ValueError("A shared policy needs every agent to have the same observation and action shapes.")
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_spaces[0].shape, dtype=np.float32)
        action_space = act_spaces[0]
        super().__init__(self.num_isaac_envs * self.num_agents, observation_space, action_space)

        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)
        self.log_buffer: list[dict[str, float]] = []
        self.episode_buffer: list[dict] = []
        self._episode_ids = np.zeros(self.num_isaac_envs, dtype=np.int64)

    # -- helpers ------------------------------------------------------------------

    def _flatten(self, per_agent: dict[str, torch.Tensor]) -> torch.Tensor:
        """dict of (E, ...) tensors -> (E * A, ...), row = env * A + agent."""
        stacked = torch.stack([per_agent[a] for a in self.agents], dim=1)
        return stacked.reshape(self.num_envs, *stacked.shape[2:])

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _collect_extras(self, extras, env_dones):
        # Consume each emission once. Isaac's extras dict persists across steps.
        log = extras.pop("log", None)
        if log and np.any(env_dones):
            self.log_buffer.append(dict(log))
        records = {int(r["env_id"]): r for r in extras.pop("episode_records", [])}
        for env_id in np.flatnonzero(env_dones):
            self._episode_ids[env_id] += 1
            record = records.get(int(env_id), {"env_id": int(env_id), "metrics": {}})
            record["episode_id"] = int(self._episode_ids[env_id])
            self.episode_buffer.append(record)
        terminal = extras.pop("terminal_observations", None)
        if np.any(env_dones) and terminal is None:
            raise RuntimeError("Missing pre-reset observation. Add TerminalObservationMixin to the task.")
        return terminal

    def _reset_statistics(self):
        self._ep_rew_buf[:] = 0.0
        self._ep_len_buf[:] = 0
        self.log_buffer.clear()
        self.episode_buffer.clear()
        self._episode_ids[:] = 0
        self.unwrapped_env.extras.pop("log", None)
        self.unwrapped_env.extras.pop("episode_records", None)

    # -- VecEnv API ---------------------------------------------------------------

    def reset(self) -> VecEnvObs:  # noqa: D102
        obs_dict, _ = self.env.reset()
        self._reset_statistics()
        return self._to_numpy(self._flatten(obs_dict))

    def step_async(self, actions):  # noqa: D102
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(np.asarray(actions))
        actions = actions.to(device=self.sim_device, dtype=torch.float32)
        actions = actions.view(self.num_isaac_envs, self.num_agents, -1)
        self._async_actions = {a: actions[:, i] for i, a in enumerate(self.agents)}

    def step_wait(self) -> VecEnvStepReturn:  # noqa: D102
        obs_dict, rew_dict, term_dict, trunc_dict, extras = self.env.step(self._async_actions)
        obs = self._to_numpy(self._flatten(obs_dict))
        rewards = self._to_numpy(self._flatten(rew_dict))
        terminated = self._to_numpy(self._flatten(term_dict))
        truncated = self._to_numpy(self._flatten(trunc_dict))
        dones = terminated | truncated

        terminal = self._collect_extras(extras, dones.reshape(self.num_isaac_envs, self.num_agents).any(-1))
        final_obs = {}
        if terminal is not None:
            for j, env_id in enumerate(self._to_numpy(terminal["env_ids"])):
                for a, name in enumerate(self.agents):
                    final_obs[int(env_id) * self.num_agents + a] = self._to_numpy(terminal["observations"][name][j])

        self._ep_rew_buf += rewards
        self._ep_len_buf += 1
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for idx in dones.nonzero()[0]:
            infos[idx]["episode"] = {"r": self._ep_rew_buf[idx], "l": self._ep_len_buf[idx]}
            infos[idx]["TimeLimit.truncated"] = bool(truncated[idx] and not terminated[idx])
            infos[idx]["terminal_observation"] = final_obs[idx]
        self._ep_rew_buf[dones] = 0.0
        self._ep_len_buf[dones] = 0
        return obs, rewards, dones, infos

    def close(self):  # noqa: D102
        self.env.close()

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        n = self.num_envs if indices is None else len(self._get_indices(indices))
        return [getattr(self.unwrapped_env, attr_name)] * n

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        if method_name == "render":
            return self.env.render()
        return getattr(self.unwrapped_env, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        return [False] * self.num_envs

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")

    def seed(self, seed: int | None = None):  # noqa: D102
        return [self.unwrapped_env.seed(seed)] * self.num_envs


class Sb3CentralVecEnv(Sb3SharedPolicyVecEnv):
    """One central policy for all agents of a :class:`DirectMARLEnv`: one SB3 row per env.

    The observation is every agent's observation concatenated in agent order, the action is
    every agent's action concatenated (agent i owns slice [i * act_dim, (i + 1) * act_dim)),
    and the reward is the team reward, the sum over agents. This is a single controller (the
    laptop) flying all the drones at once.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        A = self.num_agents
        obs_dim = int(np.prod(self.observation_space.shape))
        act = self.action_space
        self._act_dim = int(np.prod(act.shape))
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(A * obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(np.tile(act.low, A), np.tile(act.high, A), dtype=np.float32)
        VecEnv.__init__(self, self.num_isaac_envs, observation_space, action_space)
        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)

    def _central(self, per_agent: dict[str, torch.Tensor]) -> torch.Tensor:
        """dict of (E, obs_dim) -> (E, A * obs_dim), agent order."""
        return torch.cat([per_agent[a] for a in self.agents], dim=-1)

    def reset(self) -> VecEnvObs:  # noqa: D102
        obs_dict, _ = self.env.reset()
        self._reset_statistics()
        return self._to_numpy(self._central(obs_dict))

    def step_async(self, actions):  # noqa: D102
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(np.asarray(actions))
        actions = actions.to(device=self.sim_device, dtype=torch.float32)
        actions = actions.view(self.num_isaac_envs, self.num_agents, self._act_dim)
        self._async_actions = {a: actions[:, i] for i, a in enumerate(self.agents)}

    def step_wait(self) -> VecEnvStepReturn:  # noqa: D102
        obs_dict, rew_dict, term_dict, trunc_dict, extras = self.env.step(self._async_actions)
        obs = self._to_numpy(self._central(obs_dict))
        rewards = self._to_numpy(torch.stack([rew_dict[a] for a in self.agents], dim=1).sum(dim=1))
        a0 = self.agents[0]  # every agent of an env shares its reset
        terminated = self._to_numpy(term_dict[a0])
        truncated = self._to_numpy(trunc_dict[a0])
        dones = terminated | truncated

        terminal = self._collect_extras(extras, dones)
        final_obs = {}
        if terminal is not None:
            values = self._to_numpy(self._central(terminal["observations"]))
            final_obs = dict(zip(self._to_numpy(terminal["env_ids"]), values))

        self._ep_rew_buf += rewards
        self._ep_len_buf += 1
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for idx in dones.nonzero()[0]:
            infos[idx]["episode"] = {"r": self._ep_rew_buf[idx], "l": self._ep_len_buf[idx]}
            infos[idx]["TimeLimit.truncated"] = bool(truncated[idx] and not terminated[idx])
            infos[idx]["terminal_observation"] = final_obs[idx]
        self._ep_rew_buf[dones] = 0.0
        self._ep_len_buf[dones] = 0
        return obs, rewards, dones, infos


class LogExtrasCallback(BaseCallback):
    """Write the env's ``extras['log']`` statistics to the SB3 logger once per rollout.

    Each entry is weighted by its ``Episode/count`` (episodes finished that step) when present.
    """

    def __init__(self, vec_env: Sb3SharedPolicyVecEnv, verbose: int = 0):
        super().__init__(verbose)
        self._vec_env = vec_env
        self._episode_totals = {}

    def _on_step(self) -> bool:
        def count(prefix, success):
            total = self._episode_totals.setdefault(prefix, [0., 0])
            total[0] += float(success)
            total[1] += 1

        for record in self._vec_env.episode_buffer:
            if "start_kind" in record:
                count(f"Curriculum/{record['start_kind']}", record["metrics"].get("Episode/success_rate", 0.))
            for drone in record.get("drones", []):
                if "target_index" in drone:
                    count(f"Targets/pad_{drone['target_index']}", drone["land_ok"])
        # Detailed records are for evaluation; do not retain an entire training run.
        self._vec_env.episode_buffer.clear()
        return True

    def _on_rollout_end(self) -> None:
        buf = self._vec_env.log_buffer
        if not buf:
            return
        weights = np.array([entry.get("Episode/count", 1.0) for entry in buf])
        keys = {k for entry in buf for k in entry if k != "Episode/count"}
        for key in sorted(keys):
            vals = np.array([entry.get(key, np.nan) for entry in buf])
            mask = ~np.isnan(vals)
            if mask.any():
                self.logger.record(key, float(np.average(vals[mask], weights=weights[mask])))
        self.logger.record("Episode/count", float(weights.sum()))
        for prefix, (successes, count) in self._episode_totals.items():
            self.logger.record(prefix + "/success_rate", successes / count)
            self.logger.record(prefix + "/count", count)
        self._episode_totals.clear()
        buf.clear()

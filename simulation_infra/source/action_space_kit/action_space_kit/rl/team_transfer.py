"""Initialize a joint room policy from independent copies of a trained actor.

The central observation is frame-major: [frame, drone, feature]. The single
policy expects [frame, feature]. Transfer must preserve that ordering. The new
networks are dense and trainable; their cross-drone weights start at zero.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn


def extend_policy_observations(source, target, *, old_width, new_width, frame_stack, num_drones):
    """Append features to each drone/frame while preserving the complete trained policy."""
    if new_width <= old_width:
        raise ValueError("New observation block must be wider")
    if source.observation_space.shape != (old_width * frame_stack * num_drones,):
        raise ValueError("Source observation layout mismatch")
    if target.observation_space.shape != (new_width * frame_stack * num_drones,):
        raise ValueError("Target observation layout mismatch")
    source_state, target_state = source.state_dict(), target.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("Policy architectures differ")
    keys = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}
    if not keys.issubset(source_state):
        raise ValueError("Expected flat MLP actor and critic")
    with torch.no_grad():
        for key, value in target_state.items():
            original = source_state[key]
            if key in keys:
                value.zero_()
                columns = (torch.arange(frame_stack * num_drones, device=value.device)[:, None] * new_width
                           + torch.arange(old_width, device=value.device)).flatten()
                value[:, columns] = original
            else:
                value.copy_(original)
    target.load_state_dict(target_state)
    target.optimizer.state.clear()


def extend_observation_normalizer(source, venv, *, old_width, new_width, num_drones):
    """Preserve old per-drone statistics; new metre, m/s and std features use unit scaling."""
    if source.obs_rms.mean.shape != (old_width * num_drones,):
        raise ValueError("Source normalization layout mismatch")
    if venv.observation_space.shape != (new_width * num_drones,):
        raise ValueError("Target normalization layout mismatch")
    target = VecNormalize(venv, training=False, norm_obs=source.norm_obs, norm_reward=source.norm_reward,
                          clip_obs=source.clip_obs, clip_reward=source.clip_reward,
                          gamma=source.gamma, epsilon=source.epsilon)
    mean = np.zeros((num_drones, new_width))
    var = np.ones_like(mean)
    mean[:, :old_width] = source.obs_rms.mean.reshape(num_drones, old_width)
    var[:, :old_width] = source.obs_rms.var.reshape(num_drones, old_width)
    target.obs_rms.mean, target.obs_rms.var = mean.flatten(), var.flatten()
    target.obs_rms.count = source.obs_rms.count
    target.ret_rms = copy.deepcopy(source.ret_rms)
    return target


def transfer_policy(source, target, *, observation_dim, frame_stack, num_drones, value_scale=1.0):
    """Copy actor/value networks; value_scale applies to the SUM of single values."""
    if num_drones < 2 or observation_dim < 1 or frame_stack < 1:
        raise ValueError("Invalid team observation layout")
    if source.observation_space.shape != (observation_dim * frame_stack,):
        raise ValueError("Source observation does not match its frame layout")
    if target.observation_space.shape != (observation_dim * frame_stack * num_drones,):
        raise ValueError("Target observation does not match its frame layout")
    action_dim = source.action_space.shape[0]
    if target.action_space.shape != (action_dim * num_drones,):
        raise ValueError("Target action layout does not match the team")
    if source.use_sde or target.use_sde or source.squash_output or target.squash_output:
        raise ValueError("Transfer supports unsquashed diagonal Gaussian MLP policies")
    if not np.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be finite and positive")

    def copy_branch(old, new):
        if len(old) != len(new):
            raise ValueError("Network depths differ")
        linear_index = 0
        for original, expanded in zip(old, new):
            if type(original) is not type(expanded):
                raise ValueError("Network activation/layout differs")
            if not isinstance(original, nn.Linear):
                if list(original.parameters()):
                    raise ValueError("Unsupported parameterized activation")
                continue
            if expanded.weight.shape != (original.out_features * num_drones,
                                          original.in_features * num_drones):
                raise ValueError("Every hidden layer must be widened by the drone count")
            expanded.weight.zero_()
            expanded.bias.copy_(original.bias.repeat(num_drones))
            for drone in range(num_drones):
                rows = slice(drone * original.out_features, (drone + 1) * original.out_features)
                if linear_index == 0:
                    columns = (torch.arange(frame_stack, device=expanded.weight.device)[:, None]
                               * num_drones * observation_dim + drone * observation_dim
                               + torch.arange(observation_dim, device=expanded.weight.device)).flatten()
                    expanded.weight[rows, columns] = original.weight
                else:
                    columns = slice(drone * original.in_features, (drone + 1) * original.in_features)
                    expanded.weight[rows, columns].copy_(original.weight)
            linear_index += 1
        if not linear_index:
            raise ValueError("At least one hidden layer is required")

    with torch.no_grad():
        copy_branch(source.mlp_extractor.policy_net, target.mlp_extractor.policy_net)
        copy_branch(source.mlp_extractor.value_net, target.mlp_extractor.value_net)
        width = source.action_net.in_features
        target.action_net.weight.zero_()
        for drone in range(num_drones):
            target.action_net.weight[drone * action_dim:(drone + 1) * action_dim,
                                     drone * width:(drone + 1) * width].copy_(source.action_net.weight)
        target.action_net.bias.copy_(source.action_net.bias.repeat(num_drones))
        target.log_std.copy_(source.log_std.repeat(num_drones))
        target.value_net.weight.copy_(source.value_net.weight.repeat(1, num_drones) * value_scale)
        target.value_net.bias.copy_(source.value_net.bias * num_drones * value_scale)
    # Optimizer moments from either policy must not survive a transfer.
    target.optimizer.state.clear()


def expand_normalizer(source, venv, num_drones):
    """Tile input statistics and retain mean per-drone normalized reward units.

The simulator sums team rewards. Scaling the return standard deviation by N
makes the new normalized reward the mean of the old per-drone units. The new
critic must therefore initially predict the mean of the original values.
"""
    if isinstance(source.obs_rms, dict):
        raise ValueError("Only flat room observations are supported")
    dim = source.obs_rms.mean.size
    if venv.observation_space.shape != (dim * num_drones,):
        raise ValueError("Normalizer input layout does not match the team")
    target = VecNormalize(venv, training=False, norm_obs=source.norm_obs, norm_reward=source.norm_reward,
                          clip_obs=source.clip_obs, clip_reward=source.clip_reward,
                          gamma=source.gamma, epsilon=source.epsilon)
    target.obs_rms.mean = np.tile(source.obs_rms.mean, num_drones)
    target.obs_rms.var = np.tile(source.obs_rms.var, num_drones)
    target.obs_rms.count = source.obs_rms.count
    divisor = num_drones if source.norm_reward else 1
    target.ret_rms.mean = source.ret_rms.mean * divisor
    target.ret_rms.var = (source.ret_rms.var + source.epsilon) * divisor**2 - source.epsilon
    target.ret_rms.count = source.ret_rms.count
    return target, 1.0 / divisor


class CriticWarmupCallback(BaseCallback):
    """Fit joint returns before allowing PPO to change the pretrained flight actor."""

    def __init__(self, iterations):
        super().__init__()
        if iterations < 1:
            raise ValueError("Critic warmup requires at least one rollout")
        self.iterations = iterations
        self.rollouts = 0

    def _on_training_start(self):
        policy = self.model.policy
        self.actor = (policy.actor_parameters() if hasattr(policy, "actor_parameters") else
                      list(policy.mlp_extractor.policy_net.parameters())
                      + list(policy.action_net.parameters()) + [policy.log_std])
        self.original_flags = [parameter.requires_grad for parameter in self.actor]

    def _on_rollout_start(self):
        self.rollouts += 1
        frozen = self.rollouts <= self.iterations
        for parameter, original in zip(self.actor, self.original_flags):
            parameter.requires_grad_(original and not frozen)
        if self.rollouts in (1, self.iterations + 1):
            print(f"[critic-warmup] rollout={self.rollouts}; actor_frozen={frozen}", flush=True)
        self.logger.record("train/actor_frozen", float(frozen))

    def _on_step(self):
        return True

    def _on_training_end(self):
        for parameter, original in zip(self.actor, self.original_flags):
            parameter.requires_grad_(original)

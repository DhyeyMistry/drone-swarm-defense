"""Explicit PPO warm-start changes that preserve the deterministic actor."""

import math

import torch


def prepare_finetune(model, action_std=None, value_scale=None):
    policy = model.policy
    for name, value in (("action_std", action_std), ("value_scale", value_scale)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if action_std is not None:
        with torch.no_grad():
            policy.log_std.fill_(math.log(action_std))
        policy.optimizer.state.pop(policy.log_std, None)
    if value_scale is not None:
        # Convert the critic to the new reward units while retaining its learned features.
        with torch.no_grad():
            policy.value_net.weight.mul_(value_scale)
            policy.value_net.bias.mul_(value_scale)
        for parameter in (*policy.value_net.parameters(), *policy.mlp_extractor.value_net.parameters()):
            policy.optimizer.state.pop(parameter, None)

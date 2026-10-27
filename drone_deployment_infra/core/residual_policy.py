"""SB3 policy with a frozen learned pilot and a trainable command-space correction.

No hardware APIs. The base mean is clipped BEFORE adding the correction and
Gaussian exploration, so an oversized base output cannot hide braking samples.
"""

import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn


class ResidualExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, residual_width, device):
        super().__init__(feature_dim, net_arch, activation_fn, device)
        self.base_width = self.latent_dim_pi
        self.residual_net = nn.Sequential(
            nn.Linear(feature_dim, residual_width), activation_fn(),
            nn.Linear(residual_width, residual_width), activation_fn()).to(device)
        self.latent_dim_pi += residual_width

    def forward_actor(self, features):
        return torch.cat((self.policy_net(features), self.residual_net(features)), dim=-1)


class ResidualActionHead(nn.Module):
    def __init__(self, base_width, residual_width, action_dim, bound_mean=False):
        super().__init__()
        self.base_width = base_width
        self.bound_mean = bound_mean
        self.base = nn.Linear(base_width, action_dim)
        self.correction = nn.Linear(residual_width, action_dim)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, latent):
        base = self.base(latent[..., :self.base_width]).clamp(-1., 1.)
        correction = self.correction(latent[..., self.base_width:])
        mean = base + correction
        return mean.tanh() if self.bound_mean else mean


class ResidualActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, residual_width=128, bound_mean=False, **kwargs):
        self.residual_width = residual_width
        self.bound_mean = bound_mean
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ResidualExtractor(
            self.features_dim, self.net_arch, self.activation_fn, self.residual_width, self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if not isinstance(self.action_dist, DiagGaussianDistribution) or self.squash_output:
            raise ValueError("Residual pilot requires an unsquashed diagonal Gaussian policy")
        if not ((self.action_space.low == -1.).all() and (self.action_space.high == 1.).all()):
            raise ValueError("Residual pilot requires actions in [-1, 1]")
        self.action_net = ResidualActionHead(
            self.mlp_extractor.base_width, self.residual_width, self.action_space.shape[0], self.bound_mean).to(self.device)
        self.mlp_extractor.policy_net.requires_grad_(False)
        self.action_net.base.requires_grad_(False)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def actor_parameters(self):
        """Only the correction branch and exploration are trainable actor parameters."""
        return (list(self.mlp_extractor.residual_net.parameters())
                + list(self.action_net.correction.parameters()) + [self.log_std])

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "residual_width": self.residual_width,
                "bound_mean": self.bound_mean}


def copy_residual_state(source, target):
    """Copy learned parameters into a new residual head contract; commands can change."""
    if not hasattr(source.action_net, "correction"):
        raise ValueError("Expected an existing residual policy")
    if source.observation_space.shape != target.observation_space.shape or source.action_space.shape != target.action_space.shape:
        raise ValueError("Residual policy spaces must match")
    target.load_state_dict(source.state_dict())
    target.optimizer.state.clear()


def copy_learned_pilot(source, target):
    """Retain the existing shared pilot and critic; initialize correction to zero."""
    if source.observation_space.shape != target.observation_space.shape:
        raise ValueError("Pilot and residual policy observations must match")
    if source.action_space.shape != target.action_space.shape:
        raise ValueError("Pilot and residual policy actions must match")
    target.mlp_extractor.policy_net.load_state_dict(source.mlp_extractor.policy_net.state_dict())
    target.mlp_extractor.value_net.load_state_dict(source.mlp_extractor.value_net.state_dict())
    target.action_net.base.load_state_dict(source.action_net.state_dict())
    target.value_net.load_state_dict(source.value_net.state_dict())
    with torch.no_grad():
        target.log_std.copy_(source.log_std)
        target.action_net.correction.weight.zero_()
        target.action_net.correction.bias.zero_()
    target.optimizer.state.clear()

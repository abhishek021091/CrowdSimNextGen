"""Continuous action distribution head for CrowdNav++'s policy.

Ported from ``DiagGaussian`` in rl/networks/distributions.py of
github.com/Shuijing725/CrowdNav_Prediction_AttnGraph. Matches navcore's
actual action space: ``CrowdSimEnvConfig.action_mode = ActionMode.VELOCITY``
uses a continuous 2D ``Box`` action space (vx, vy) -- the same case the
original routes to ``DiagGaussian`` (its ``Categorical``/``Bernoulli``
siblings handle discrete/multi-binary action spaces navcore doesn't use
and are not ported).

Simplification from the original, not a behavior change:
    The original computes a state-independent log-std by running a
    zeros tensor through an ``AddBias`` module (a generic "add a
    learnable per-element bias" layer, reused here only because it
    happened to exist already for another purpose in that codebase).
    That is mathematically just a learnable ``log_std`` parameter with
    extra indirection. This port uses a plain ``nn.Parameter`` directly.
    Also, rather than hand-rolling a ``FixedNormal`` subclass to get
    "log-prob summed over action dimensions," this wraps
    ``torch.distributions.Normal`` in ``torch.distributions.Independent``,
    which provides that summation as a documented, standard composition
    rather than custom subclass logic -- same math, less bespoke code.
"""

from __future__ import annotations

from dataclasses import dataclass

from networkx import config
import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal


@dataclass(slots=True, frozen=True)
class DiagGaussianHeadConfig:
    """Hyperparameters for :class:`DiagGaussianHead`.

    Attributes:
        input_dim: Width of the actor features this head reads (the
            output width of the actor MLP in ``ActorCriticHeads``).
        action_dim: Number of continuous action dimensions. 2 for
            navcore's velocity action space (vx, vy).
    """

    input_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim!r}.")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim!r}.")


class DiagGaussianHead(nn.Module):
    LOG_STD_MIN = -3.0
    LOG_STD_MAX = 0.5

    def __init__(self, config: DiagGaussianHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.mean_linear = nn.Linear(config.input_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), -1.0))

    def forward(self, actor_features: Tensor) -> Independent:
        if actor_features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"actor_features' last dim is {actor_features.shape[-1]}, "
                f"but this head was configured for input_dim="
                f"{self.config.input_dim}."
            )
        mean = self.mean_linear(actor_features)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return Independent(Normal(mean, std), 1)


def select_action(
    distribution: Independent, deterministic: bool = False
) -> tuple[Tensor, Tensor]:
    """Sample (or deterministically select) an action, with its log-prob.

    Free function, not a method on any ``nn.Module``, since action
    selection is a policy-usage concern, not a learnable component of
    the network itself.

    Args:
        distribution: The output of ``DiagGaussianHead.forward``.
        deterministic: If ``True``, return the distribution's mean
            (mode) instead of a stochastic sample -- typically used at
            evaluation/deployment time, not during training rollout
            collection.

    Returns:
        ``(action, log_prob)``. ``log_prob`` is already summed over the
        action dimension (one scalar per batch element), matching what
        a PPO update would need -- computing it here, alongside
        selection, is what the original's ``Policy.act`` does too.
    """
    action = distribution.mean if deterministic else distribution.sample()
    log_prob = distribution.log_prob(action)
    return action, log_prob

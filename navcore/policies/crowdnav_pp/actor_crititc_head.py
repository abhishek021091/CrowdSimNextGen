"""Actor/critic heads for CrowdNav++'s interaction-graph policy.

Ported from the ``self.actor``/``self.critic``/``self.critic_linear``
definitions in ``selfAttn_merge_SRNN.__init__`` (rl/networks/
selfAttn_srnn_temp_node.py of
github.com/Shuijing725/CrowdNav_Prediction_AttnGraph).

Kept decoupled from the action distribution on purpose, mirroring a
separation already present in the original:
    The original's own ``Policy`` wrapper (rl/networks/model.py) keeps
    the network "base" (everything up through these actor/critic
    features) separate from ``self.dist`` (``Categorical`` /
    ``DiagGaussian`` / ``Bernoulli``, chosen based on the Gym action
    space's type). This class reproduces that same boundary: it returns
    raw actor features and a scalar value, not a distribution -- the
    caller applies whichever action-distribution head fits (navcore's
    velocity action space uses ``DiagGaussianHead``, a sibling module).
    This is what lets these heads be reused unchanged if navcore ever
    adds a discrete action mode.

Design note the original makes, kept as-is:
    Actor and critic are two entirely separate MLP towers from
    ``RecurrentNodeUpdate``'s output onward -- no shared trunk below the
    split. This is more parameters than a shared-trunk architecture, but
    it is what the original publishes and benchmarks, so it is kept
    faithfully rather than "optimized" into a shared trunk here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class ActorCriticHeadsConfig:
    """Hyperparameters for :class:`ActorCriticHeads`.

    Attributes:
        input_dim: Width of the input (``RecurrentNodeUpdate``'s
            ``output_size``).
        hidden_size: Width of both MLP towers' hidden layers. Equal to
            ``input_dim`` in the original (256 in its defaults) -- kept
            as a separate parameter here since there is no structural
            reason the two must match, even though the original's
            values happen to coincide.
    """

    input_dim: int
    hidden_size: int = 256

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim!r}.")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size!r}.")


def _orthogonal_linear(
    in_features: int, out_features: int, gain: float = np.sqrt(2)
) -> nn.Linear:
    """Build a ``Linear`` layer with the original's init: orthogonal
    weights (gain ``sqrt(2)``, standard for a layer feeding a Tanh),
    zero bias.
    """
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer


class ActorCriticHeads(nn.Module):
    """Two-tower MLP producing actor features and a scalar value estimate.

    Attributes:
        config: This module's hyperparameters.
    """

    def __init__(self, config: ActorCriticHeadsConfig) -> None:
        super().__init__()
        self.config = config

        self.actor = nn.Sequential(
            _orthogonal_linear(config.input_dim, config.hidden_size),
            nn.Tanh(),
            _orthogonal_linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            _orthogonal_linear(config.input_dim, config.hidden_size),
            nn.Tanh(),
            _orthogonal_linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )
        # gain=0.01 (near-linear at init) on the final value projection
        # matches the original -- a large initial value-head output
        # would otherwise dominate early PPO advantage estimates before
        # the critic has learned anything.
        self.critic_linear = _orthogonal_linear(config.hidden_size, 1, gain=0.01)

    def forward(self, node_output: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(value, actor_features)`` for this tick.

        Args:
            node_output: ``[..., input_dim]`` -- e.g.
                ``RecurrentNodeUpdate``'s output.

        Returns:
            ``value``: ``[..., 1]``, the critic's scalar state-value
            estimate.
            ``actor_features``: ``[..., hidden_size]``, to be passed to
            an action-distribution head (e.g. ``DiagGaussianHead``).

        Raises:
            ValueError: If ``node_output``'s last dimension doesn't
                match ``config.input_dim``.
        """
        if node_output.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"node_output's last dim is {node_output.shape[-1]}, but "
                f"this module was configured for input_dim="
                f"{self.config.input_dim}."
            )
        actor_features = self.actor(node_output)
        value = self.critic_linear(self.critic(node_output))
        return value, actor_features

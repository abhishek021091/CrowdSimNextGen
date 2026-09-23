"""ContextFusionGate: feature-wise learned gate fusing the human and
obstacle interaction contexts into one context vector for the GRU.

Neither branch is trusted unconditionally: a scene with a clear human
threat but no nearby obstacle (or vice versa) should be able to lean the
fused context toward whichever branch actually carries signal this tick,
per-feature rather than with one scalar switch -- hence the gate's shape
is `[B, D]`, not `[B, 1]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class FusionGateConfig:
    """Hyperparameters for :class:`ContextFusionGate`.

    Attributes:
        embedding_dim: Width of both the human and obstacle context
            vectors -- they must already share this width by the time
            they reach this gate (see policy.py's LayerNorm/projection
            steps immediately before fusion).
        hidden_size: Hidden width of the gate's small MLP.
    """

    embedding_dim: int = 256
    hidden_size: int = 128

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")


class ContextFusionGate(nn.Module):
    """Feature-wise sigmoid gate between two equal-width context vectors.

    ``gate = sigmoid(MLP([human_context || obstacle_context]))``,
    ``fused = gate * human_context + (1 - gate) * obstacle_context``.

    Attributes:
        config: This gate's hyperparameters.
        last_gate: The most recent forward call's gate tensor, `[B, D]`,
            exposed for tests/diagnostics (e.g. verifying gate values
            stay in `[0, 1]`, or later inspecting which branch a trained
            policy leans on in which situations). Not read by
            `forward()` itself.
    """

    def __init__(self, config: FusionGateConfig) -> None:
        super().__init__()
        self.config = config
        self.gate_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.embedding_dim),
        )
        self.last_gate: Tensor | None = None

    def forward(self, human_context: Tensor, obstacle_context: Tensor) -> Tensor:
        """Fuse two context vectors with a learned, feature-wise gate.

        Args:
            human_context: `[B, embedding_dim]`, already LayerNorm'd.
            obstacle_context: `[B, embedding_dim]`, already projected
                into the shared embedding space and LayerNorm'd.

        Returns:
            `[B, embedding_dim]` fused context.

        Raises:
            ValueError: On a shape/width mismatch between the two inputs.
        """
        if human_context.shape != obstacle_context.shape:
            raise ValueError(
                f"human_context shape {tuple(human_context.shape)} does not "
                f"match obstacle_context shape {tuple(obstacle_context.shape)}."
            )
        if human_context.shape[-1] != self.config.embedding_dim:
            raise ValueError(
                f"Context vectors' last dim is {human_context.shape[-1]}, "
                f"but this gate was configured for embedding_dim="
                f"{self.config.embedding_dim}."
            )

        gate = torch.sigmoid(
            self.gate_mlp(torch.cat((human_context, obstacle_context), dim=-1))
        )
        self.last_gate = gate
        return gate * human_context + (1.0 - gate) * obstacle_context

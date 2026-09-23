"""RobotObstacleAttention: single dot-product attention pass, robot query
over obstacle tokens.

Deliberately NOT a reuse of `RobotHumanAttention` (see
`navcore.policies.crowdnav_pp.policy`'s module docstring for the fusion-
architecture rationale): humans and obstacles must never share one
attention module, so the robot's query, and the key/value space it
attends over, can specialize independently per modality. Consumes
`RangeImageEncoder`'s obstacle tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class RobotObstacleAttentionConfig:
    """Hyperparameters for :class:`RobotObstacleAttention`.

    Attributes:
        robot_embedding_dim: Width of the incoming robot embedding
            (256, matching `RobotStateEncoder`'s output /
            `CrowdNavPPPolicyConfig.interaction_embedding_dim`).
        obstacle_embedding_dim: Width of each obstacle token (must match
            `RangeImageEncoderConfig.token_embedding_dim`) and of this
            attention's internal query/key/value space -- the obstacle
            branch's own latent space, kept separate from
            `robot_embedding_dim` until after attention (see policy.py's
            "own internal latent space" design decision).
        num_attention_heads: Attention head count. Must evenly divide
            `obstacle_embedding_dim`.
    """

    robot_embedding_dim: int = 256
    obstacle_embedding_dim: int = 128
    num_attention_heads: int = 4

    def __post_init__(self) -> None:
        if self.robot_embedding_dim <= 0 or self.obstacle_embedding_dim <= 0:
            raise ValueError("Embedding dims must be positive.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.obstacle_embedding_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"obstacle_embedding_dim ({self.obstacle_embedding_dim}) must "
                f"be divisible by num_attention_heads "
                f"({self.num_attention_heads})."
            )


class RobotObstacleAttention(nn.Module):
    """Robot-query attention over a fixed-size set of obstacle tokens.

    Attributes:
        config: This module's hyperparameters.
        query_proj: Projects the robot embedding (256) down into the
            obstacle branch's own latent space (128) before it becomes
            the attention query -- required by the architecture spec
            ("robot embedding must first pass through its own
            projection layer 256 -> 128 before becoming the attention
            query").
    """

    def __init__(self, config: RobotObstacleAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.query_proj = nn.Linear(
            config.robot_embedding_dim, config.obstacle_embedding_dim
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config.obstacle_embedding_dim,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )

    def forward(
        self,
        robot_embedding: Tensor,
        obstacle_tokens: Tensor,
        obstacle_mask: Tensor | None = None,
    ) -> Tensor:
        """Attend the (projected) robot embedding over obstacle tokens.

        Args:
            robot_embedding: `[B, robot_embedding_dim]`.
            obstacle_tokens: `[B, num_tokens, obstacle_embedding_dim]`,
                as produced by `RangeImageEncoder`.
            obstacle_mask: Optional `[B, num_tokens]` boolean, True for
                a token that should be attended to. `RangeImageEncoder`
                always produces a fixed number of geometrically
                meaningful tokens (every angular sector is either free
                or blocked -- there is no "padding" token the way a
                variable-size human list has), so this is `None` in the
                common case; accepted for forward compatibility with a
                future variable-token-count encoder.

        Returns:
            `[B, obstacle_embedding_dim]` obstacle context vector.

        Raises:
            ValueError: On a robot/token embedding-dim mismatch.
        """
        if robot_embedding.dim() != 2:
            raise ValueError(
                f"robot_embedding must be [B, robot_embedding_dim], got "
                f"shape {tuple(robot_embedding.shape)}."
            )
        if robot_embedding.shape[-1] != self.config.robot_embedding_dim:
            raise ValueError(
                f"robot_embedding's last dim is {robot_embedding.shape[-1]}, "
                f"but this layer was configured for robot_embedding_dim="
                f"{self.config.robot_embedding_dim}."
            )
        if obstacle_tokens.dim() != 3:
            raise ValueError(
                f"obstacle_tokens must be [B, num_tokens, "
                f"obstacle_embedding_dim], got shape "
                f"{tuple(obstacle_tokens.shape)}."
            )
        if obstacle_tokens.shape[-1] != self.config.obstacle_embedding_dim:
            raise ValueError(
                f"obstacle_tokens' last dim is {obstacle_tokens.shape[-1]}, "
                f"but this layer was configured for obstacle_embedding_dim="
                f"{self.config.obstacle_embedding_dim}."
            )
        if obstacle_tokens.shape[0] != robot_embedding.shape[0]:
            raise ValueError(
                f"obstacle_tokens batch size {obstacle_tokens.shape[0]} does "
                f"not match robot_embedding batch size "
                f"{robot_embedding.shape[0]}."
            )

        query = self.query_proj(robot_embedding).unsqueeze(1)  # [B, 1, D]
        key_padding_mask = ~obstacle_mask if obstacle_mask is not None else None

        context, _ = self.attention(
            query=query,
            key=obstacle_tokens,
            value=obstacle_tokens,
            key_padding_mask=key_padding_mask,
        )
        return context.squeeze(1)  # [B, D]

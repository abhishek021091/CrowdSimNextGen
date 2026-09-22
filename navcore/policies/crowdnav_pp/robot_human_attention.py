"""Robot-human attention for CrowdNav++'s recurrent interaction-graph policy.

Ported from ``EdgeAttention_M`` in
rl/networks/selfAttn_srnn_temp_node.py of
github.com/Shuijing725/CrowdNav_Prediction_AttnGraph. See
``attention.py``'s module docstring (sibling file, ``HumanHumanAttention``)
for the general porting rationale (typed config instead of a shared
``argparse.Namespace``, explicit feature widths instead of hardcoded
per-env branches). This module adds one more deliberate deviation from
the original, noted below.

What this computes:
    A single dot-product attention pass where the robot's own encoded
    state is the *query* and every visible human's (already
    human-human-attended) embedding is a *key/value* pair. The output is
    one crowd-context vector per robot -- "given who I am and how I'm
    moving, which humans matter most right now, and what does the
    weighted-relevant-human-state look like." This is the second half of
    CrowdNav++'s interaction graph; ``HumanHumanAttention`` (sibling
    module) is the first half.

Naming deviation from the original, on purpose:
    The original calls its inputs ``h_temporal``/``h_spatials`` --
    leftover naming from the predecessor DS-RNN architecture
    (``srnn_model.py``), which really did have per-edge RNNs. The actual
    CrowdNav++ network no longer has edge RNNs at all (see this
    project's notes on `srnn_model.py` vs `selfAttn_srnn_temp_node.py`);
    ``h_temporal`` is just the robot's own projected embedding and
    ``h_spatials`` is the humans' attended embeddings. Renamed to
    ``robot_embedding``/``human_embeddings`` here so the code describes
    what it is now, not what it used to be.

Faithfulness note -- attention scaling is NOT standard scaled-dot-product:
    Standard scaled-dot-product attention divides scores by
    ``sqrt(d_k)``. The original *multiplies* the raw dot-product score by
    ``human_count / sqrt(attention_size)`` instead -- i.e. attention
    sharpens as more humans are visible, the opposite direction you'd
    expect from a per-key normalization term. This looks unusual, but
    this port is meant to reproduce the paper's actual behavior, so it
    is kept exactly as published rather than "corrected" to the standard
    convention. Flagged here rather than silently changed or silently
    kept without comment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class RobotHumanAttentionConfig:
    embedding_dim: int
    num_attention_heads: int = 8

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim!r}."
            )
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.embedding_dim % self.num_attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_attention_heads.")


class RobotHumanAttention(nn.Module):
    """Single dot-product attention pass: robot embedding queries humans
    (and, optionally, a static-obstacle summary).

    Attributes:
        config: This layer's hyperparameters.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )

    def forward(
        self,
        robot_embedding: Tensor,
        human_embeddings: Tensor,
        visible_mask: Tensor,
        obstacle_embedding: Tensor | None = None,
        obstacle_mask: Tensor | None = None,
    ) -> Tensor:
        """...
        Args:
            ...
            obstacle_embedding: Optional ``[seq_len, nenv, num_obstacle_tokens,
                embedding_dim]`` -- one key/value token per ObstacleEncoder
                ray (no longer a single pooled summary; pooling was removed
                from ObstacleEncoder because it destroyed left/right
                directional information even in an untrained encoder).
                A ``[seq_len, nenv, embedding_dim]`` tensor (no token axis)
                is still accepted for backward compatibility and treated as
                one always-visible token.
            obstacle_mask: Optional ``[seq_len, nenv, num_obstacle_tokens]``
                boolean, True where that ray actually hit something. A ray
                that hit nothing carries no real geometry and should not
                compete for attention weight as a phantom "obstacle at the
                robot" token -- omit only if every token should count as
                visible (rare; prefer passing the real hit mask).
        """
        seq_len, nenv, human_count, embedding_dim = human_embeddings.shape

        if obstacle_embedding is not None:
            if obstacle_embedding.dim() == 3:
                obstacle_embedding = obstacle_embedding.unsqueeze(2)

            if (
                obstacle_embedding.shape[:2] != (seq_len, nenv)
                or obstacle_embedding.shape[-1] != embedding_dim
            ):
                raise ValueError(
                    f"obstacle_embedding shape {tuple(obstacle_embedding.shape)} "
                    f"is not compatible with expected leading dims "
                    f"{(seq_len, nenv)} and embedding_dim={embedding_dim}."
                )
            num_obstacle_tokens = obstacle_embedding.shape[2]

            if obstacle_mask is None:
                obstacle_mask = visible_mask.new_ones(
                    seq_len, nenv, num_obstacle_tokens
                )
            elif tuple(obstacle_mask.shape) != (seq_len, nenv, num_obstacle_tokens):
                raise ValueError(
                    f"obstacle_mask shape {tuple(obstacle_mask.shape)} does not "
                    f"match obstacle_embedding's token count "
                    f"{(seq_len, nenv, num_obstacle_tokens)}."
                )

            human_embeddings = torch.cat((human_embeddings, obstacle_embedding), dim=2)
            visible_mask = torch.cat((visible_mask, obstacle_mask), dim=2)

        slots = human_embeddings.shape[2]
        batch = seq_len * nenv

        query = robot_embedding.reshape(batch, 1, embedding_dim)
        key = human_embeddings.reshape(batch, slots, embedding_dim)
        mask = visible_mask.reshape(batch, slots)

        # Let standard MultiheadAttention handle projections and optimal scaling
        context, _ = self.attention(
            query=query,
            key=key,
            value=key,  # Key and Value share the same base tensor before internal projection
            key_padding_mask=~mask,
        )

        return context.reshape(seq_len, nenv, embedding_dim)

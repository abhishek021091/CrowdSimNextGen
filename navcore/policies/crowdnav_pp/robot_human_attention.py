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
    """Hyperparameters for :class:`RobotHumanAttention`.

    Attributes:
        embedding_dim: Width of both the robot's own embedding and each
            human's embedding arriving at this layer -- they must match,
            since the query/key projections below share this input
            width. 256 in the original (``human_human_edge_rnn_size``).
        attention_size: Width of the projection space attention scores
            are computed in. 64 in the original.
    """

    embedding_dim: int
    attention_size: int = 64

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim!r}."
            )
        if self.attention_size <= 0:
            raise ValueError(
                f"attention_size must be positive, got {self.attention_size!r}."
            )


class RobotHumanAttention(nn.Module):
    """Single dot-product attention pass: robot embedding queries humans
    (and, optionally, a static-obstacle summary).

    Attributes:
        config: This layer's hyperparameters.
    """

    def __init__(self, config: RobotHumanAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.query_proj = nn.Linear(config.embedding_dim, config.attention_size)
        self.key_proj = nn.Linear(config.embedding_dim, config.attention_size)

    def forward(
        self,
        robot_embedding: Tensor,
        human_embeddings: Tensor,
        visible_mask: Tensor,
        obstacle_embedding: Tensor | None = None,
    ) -> Tensor:
        """Return one crowd(+obstacle)-context vector per robot, per (seq, env) slot.

        Args:
            robot_embedding: ``[seq_len, nenv, 1, embedding_dim]`` -- the
                robot's own embedding.
            human_embeddings: ``[seq_len, nenv, max_human_num,
                embedding_dim]`` -- per-human embeddings.
            visible_mask: ``[seq_len, nenv, max_human_num]`` boolean,
                ``True`` for a real (non-padding) human slot.
            obstacle_embedding: Optional ``[seq_len, nenv, embedding_dim]``
                -- a single per-(seq, env) summary of the robot's static
                surroundings (e.g. ``ObstacleEncoder``'s pooled ray-scan
                embedding, already projected to ``embedding_dim`` by the
                caller). When given, it is appended as one extra,
                always-visible key/value slot alongside the human
                embeddings, so the same softmax that decides which
                humans matter this tick also decides how much weight the
                static-obstacle context deserves -- rather than obstacle
                information reaching the policy through a separate,
                unconditionally-added branch (``RecurrentNodeUpdate``
                used to have one; it's been removed in favor of this).
                ``None`` (the default) recovers the original human-only
                behavior exactly: the temperature below is derived from
                the human count alone, computed *before* this slot is
                appended, so attaching an obstacle token never changes
                attention sharpness for a fixed crowd.

        Returns:
            ``[seq_len, nenv, embedding_dim]`` -- one context vector per
            (seq, env) slot, blending whichever humans (and, if given,
            the static-obstacle summary) the robot's query attends to.

        Raises:
            ValueError: If the inputs' shapes are inconsistent with each
                other or with ``obstacle_embedding``, or if any ``(seq,
                env)`` slot has zero visible humans and no
                ``obstacle_embedding`` was given to fall back on (see
                ``HumanHumanAttention`` for why a fully-masked softmax
                row is rejected rather than silently producing ``NaN``).
        """
        seq_len, nenv, human_count, embedding_dim = human_embeddings.shape
        if tuple(robot_embedding.shape) != (seq_len, nenv, 1, embedding_dim):
            raise ValueError(
                f"robot_embedding shape {tuple(robot_embedding.shape)} does "
                f"not match expected {(seq_len, nenv, 1, embedding_dim)} "
                f"(derived from human_embeddings)."
            )
        if tuple(visible_mask.shape) != (seq_len, nenv, human_count):
            raise ValueError(
                f"visible_mask shape {tuple(visible_mask.shape)} does not "
                f"match human_embeddings' leading dims "
                f"{(seq_len, nenv, human_count)}."
            )
        if embedding_dim != self.config.embedding_dim:
            raise ValueError(
                f"Embeddings have width {embedding_dim}, but this layer "
                f"was configured for embedding_dim={self.config.embedding_dim}."
            )

        # See "Faithfulness note" above / in the module docstring: this
        # scaling is tied to the human count specifically, computed
        # before any obstacle slot is appended below.
        temperature = human_count / (self.config.attention_size**0.5)

        if obstacle_embedding is not None:
            if tuple(obstacle_embedding.shape) != (seq_len, nenv, embedding_dim):
                raise ValueError(
                    f"obstacle_embedding shape "
                    f"{tuple(obstacle_embedding.shape)} does not match "
                    f"expected {(seq_len, nenv, embedding_dim)}."
                )
            human_embeddings = torch.cat(
                (human_embeddings, obstacle_embedding.unsqueeze(2)), dim=2
            )
            visible_mask = torch.cat(
                (visible_mask, visible_mask.new_ones(seq_len, nenv, 1)), dim=2
            )

        fully_masked = visible_mask.sum(dim=-1) == 0
        if bool(fully_masked.any()):
            raise ValueError(
                "visible_mask contains at least one (seq, env) slot with "
                "zero visible humans, and no obstacle_embedding was given "
                "to fall back on -- softmax over zero unmasked keys is "
                "undefined. Pass an obstacle_embedding, or see "
                "HumanHumanAttention's docstring for the same open design "
                "question (synthetic dummy human vs. caller-guaranteed "
                "non-empty crowd)."
            )

        query = self.query_proj(robot_embedding)  # [seq_len, nenv, 1, attention_size]
        key = self.key_proj(human_embeddings)  # [seq_len, nenv, slots, attention_size]

        scores = (query * key).sum(dim=-1)  # [seq_len, nenv, slots]
        scores = scores * temperature

        scores = scores.masked_fill(~visible_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # [seq_len, nenv, slots]

        weighted = (weights.unsqueeze(-1) * human_embeddings).sum(dim=2)

        return weighted

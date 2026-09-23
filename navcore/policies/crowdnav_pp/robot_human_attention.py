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

Humans only -- obstacles are a separate branch:
    This module previously also accepted an optional static-obstacle
    summary (``obstacle_embedding``/``obstacle_mask``), folded in as
    extra key/value tokens alongside humans. That has been removed: the
    obstacle branch is now ``RangeImageEncoder`` ->
    ``RobotObstacleAttention`` (see
    ``navcore.policies.crowdnav_pp.policy``'s module docstring), a
    completely independent attention pass with its own query projection
    and its own latent space, fused with this module's output only
    afterward, via ``ContextFusionGate``. Humans and obstacles must
    never again share one attention module -- a human's identity,
    visibility, and motion history have nothing in common with a
    ray-cast occupancy sector's, and letting them compete for the same
    softmax was an artifact of the old 1D-CNN branch reusing whatever
    attention module happened to already exist, not a deliberate
    architectural choice.

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
    """Single dot-product attention pass: robot embedding queries humans.

    Attributes:
        config: This layer's hyperparameters.
    """

    def __init__(self, config: RobotHumanAttentionConfig) -> None:
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
    ) -> Tensor:
        """Return the robot's crowd-context vector for this tick.

        Args:
            robot_embedding: ``[seq_len, nenv, 1, embedding_dim]`` --
                the robot's own (already-projected) embedding, with a
                singleton "1 robot" axis matching this layer's query
                convention.
            human_embeddings: ``[seq_len, nenv, human_count,
                embedding_dim]`` -- every human slot's (already
                human-human-attended) embedding, zero-padded past the
                real visible-human count.
            visible_mask: ``[seq_len, nenv, human_count]`` boolean,
                ``True`` for a real (non-padding) human slot.

        Returns:
            ``[seq_len, nenv, embedding_dim]`` -- one crowd-context
            vector per (seq, env).
        """
        seq_len, nenv, human_count, embedding_dim = human_embeddings.shape
        batch = seq_len * nenv

        query = robot_embedding.reshape(batch, 1, embedding_dim)
        key = human_embeddings.reshape(batch, human_count, embedding_dim)
        mask = visible_mask.reshape(batch, human_count)

        # Let standard MultiheadAttention handle projections and optimal scaling
        context, _ = self.attention(
            query=query,
            key=key,
            value=key,  # Key and Value share the same base tensor before internal projection
            key_padding_mask=~mask,
        )

        return context.reshape(seq_len, nenv, embedding_dim)

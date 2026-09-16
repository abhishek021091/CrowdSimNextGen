"""Attention modules for CrowdNav++'s recurrent interaction-graph policy.

Architecture ported from Liu et al., "Intention Aware Robot Crowd
Navigation with Attention-Based Interaction Graph" (ICRA 2023). Official
code: github.com/Shuijing725/CrowdNav_Prediction_AttnGraph,
rl/networks/selfAttn_srnn_temp_node.py. Re-implemented against navcore's
own conventions rather than transcribed:

    - The original threads a single monolithic `argparse.Namespace`
      (named `args` everywhere) through every submodule -- every layer
      reaches into a shared, untyped blob for its own hyperparameters
      (`args.attention_size`, `args.human_node_rnn_size`, ...). That is
      exactly the hidden-state / tight-coupling pattern this project
      avoids. Each class here instead takes an explicit, typed config
      dataclass carrying only the fields it actually needs.
    - The original hardcodes each attention layer's input feature width
      by branching on `args.env_name` (a string naming which Gym env
      registration is active) inside the network itself -- e.g. "12 if
      env_name in [...] else 2". That couples the network to a fixed set
      of env-specific encodings baked into a string comparison. Here,
      feature width is an explicit constructor parameter. See the open
      architectural question below for why this matters.

Open architectural question, not resolved here:
    The original's per-human "spatial edge" feature is 12-dimensional in
    its main configuration -- each human's current relative state *plus*
    several steps of predicted future trajectory from a separately
    trained transformer (GST). That predicted-future component is what
    makes the policy "intention aware," and it is not a detail this
    attention layer can supply on its own -- it has to already be present
    in whatever feature vector navcore's ObservationEncoder hands it.
    navcore currently has `TrajectoryTargetRecorder`/
    `FutureTrajectoryTargets` for producing *training labels* for a
    future auxiliary predictor, but nothing yet that runs a live
    predictor at inference time to augment `ObservationEncoder`'s
    neighbor features the way GST does upstream of this network in the
    original. Until that predictor exists, this port can only faithfully
    reproduce CrowdNav++'s *architecture*, not its full intention-aware
    *behavior*. Flagged per the roadmap item "lock observation schema
    before CrowdNav++ model code is written" -- this file intentionally
    does not guess at that schema by hardcoding a feature width.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class HumanHumanAttentionConfig:
    """Hyperparameters for :class:`HumanHumanAttention`.

    Attributes:
        spatial_edge_feature_dim: Width of one human's per-tick feature
            vector, as produced by whatever encodes navcore's
            observation (e.g. a future extension of
            ``ObservationEncoder``'s per-neighbor features). The original
            hardcodes this per env variant (2 with no prediction, 12
            with a 4-step GST prediction concatenated in); left explicit
            here rather than guessed at -- see module docstring.
        embedding_size: Width of the per-human embedding attention
            operates on. 512 in the original.
        num_attention_heads: Number of self-attention heads. Must evenly
            divide ``embedding_size`` (a ``torch.nn.MultiheadAttention``
            requirement). 8 in the original.
    """

    spatial_edge_feature_dim: int
    embedding_size: int = 512
    num_attention_heads: int = 8

    def __post_init__(self) -> None:
        if self.spatial_edge_feature_dim <= 0:
            raise ValueError(
                f"spatial_edge_feature_dim must be positive, got "
                f"{self.spatial_edge_feature_dim!r}."
            )
        if self.embedding_size % self.num_attention_heads != 0:
            raise ValueError(
                f"embedding_size ({self.embedding_size}) must be divisible "
                f"by num_attention_heads ({self.num_attention_heads})."
            )


class HumanHumanAttention(nn.Module):
    """Multi-head self-attention over currently-visible humans' features.

    Lets each human's embedding be informed by every other visible
    human's state before the robot-human attention layer (a sibling
    module, ported separately) reads it -- e.g. two humans converging on
    a narrow gap should influence each other's encoded "danger" even
    before the robot's own attention is applied. This is the
    "human-human" half of CrowdNav++'s interaction graph; the
    "robot-human" half is a separate, smaller module by design (see the
    project's "small classes" preference) and is ported next, once this
    layer's interface is confirmed against navcore's real
    ``ObservationEncoder`` output.

    Padding convention:
        A batch mixes environments with different numbers of currently-
        visible humans, zero-padded up to a fixed ``max_human_num`` --
        the same convention ``navcore.gym_wrapper.observation_encoder``
        already uses for its ``neighbors``/``neighbor_mask`` pair.
        ``forward`` takes an explicit boolean mask so padded slots never
        contribute to another human's attention output.
    """

    def __init__(self, config: HumanHumanAttentionConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Sequential(
            nn.Linear(config.spatial_edge_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.embedding_size),
            nn.ReLU(),
        )
        self.query_proj = nn.Linear(config.embedding_size, config.embedding_size)
        self.key_proj = nn.Linear(config.embedding_size, config.embedding_size)
        self.value_proj = nn.Linear(config.embedding_size, config.embedding_size)
        self.attention = nn.MultiheadAttention(
            config.embedding_size, config.num_attention_heads
        )

    def forward(self, human_features: Tensor, visible_mask: Tensor) -> Tensor:
        """Return each human's self-attended embedding.

        Args:
            human_features: ``[seq_len, nenv, max_human_num,
                spatial_edge_feature_dim]`` -- one feature vector per
                human slot, zero-padded past each environment's actual
                visible-human count.
            visible_mask: ``[seq_len, nenv, max_human_num]`` boolean,
                ``True`` for a real (non-padding) human slot. Matches the
                semantics of ``ObservationEncoder``'s ``neighbor_mask``
                (there, ``1``/``0`` rather than bool -- cast before
                calling this).

        Returns:
            ``[seq_len, nenv, max_human_num, embedding_size]`` -- one
            self-attended embedding per human slot. Padded slots' output
            values are not meaningful and are not zeroed here (matching
            ``nn.MultiheadAttention``'s convention of masking attention
            *weights*, not outputs) -- the caller must re-mask before
            any downstream reduction over the human dimension.

        Raises:
            ValueError: If ``human_features`` and ``visible_mask``
                disagree on ``[seq_len, nenv, max_human_num]``, if the
                last dimension of ``human_features`` doesn't match
                ``config.spatial_edge_feature_dim``, or if any
                ``(seq, env)`` slot has zero visible humans.
        """
        seq_len, nenv, max_human_num, feature_dim = human_features.shape
        if tuple(visible_mask.shape) != (seq_len, nenv, max_human_num):
            raise ValueError(
                f"visible_mask shape {tuple(visible_mask.shape)} does not "
                f"match human_features' leading dims "
                f"{(seq_len, nenv, max_human_num)}."
            )
        if feature_dim != self.config.spatial_edge_feature_dim:
            raise ValueError(
                f"human_features' last dim is {feature_dim}, but this "
                f"layer was configured for spatial_edge_feature_dim="
                f"{self.config.spatial_edge_feature_dim}."
            )

        batch = seq_len * nenv
        flat_mask = visible_mask.reshape(batch, max_human_num)
        fully_masked = flat_mask.sum(dim=-1) == 0
        if bool(fully_masked.any()):
            raise ValueError(
                "human_features contains at least one (seq, env) slot with "
                "zero visible humans. nn.MultiheadAttention produces NaNs "
                "for a fully-masked row; the original CrowdNav++ "
                "implementation works around this with a synthetic "
                "'dummy human' placeholder at slot 0. Whether navcore "
                "should do the same, or whether the caller should "
                "guarantee at least one always-present slot, is an open "
                "design question -- not resolved here, so this fails "
                "loudly instead of silently producing NaNs."
            )

        embedded = self.embed(human_features).reshape(batch, max_human_num, -1)
        # nn.MultiheadAttention (non-batch-first form) is sequence-first:
        # [max_human_num, batch, embedding_size].
        embedded = embedded.transpose(0, 1)

        query = self.query_proj(embedded)
        key = self.key_proj(embedded)
        value = self.value_proj(embedded)

        # key_padding_mask=True means "ignore this key" in PyTorch's
        # convention -- the inverse of navcore's neighbor_mask convention
        # (True means "this is a real neighbor"), hence the negation.
        key_padding_mask = ~flat_mask

        attended, _ = self.attention(
            query, key, value, key_padding_mask=key_padding_mask
        )
        attended = attended.transpose(0, 1)  # [batch, max_human_num, embedding_size]
        return attended.reshape(
            seq_len, nenv, max_human_num, self.config.embedding_size
        )

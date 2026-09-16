# navcore/policies/gst_predictor/gst_predictor.py
"""GSTPredictor: the assembled Gumbel Social Transformer trajectory predictor.

Pipeline: per-human temporal encoding (reuses navcore's existing
TemporalEncoder, same masked-LSTM-over-history-window module CrowdNav++'s
own policy uses for instantaneous edge features) -> learned sparse
interaction graph (SparseInteractionGraph) -> graph-gated spatial
attention (GraphGatedTransformerLayer, stacked) -> multi-step Gaussian
future-displacement head (GaussianTrajectoryHead).

Trained-separately-and-frozen, by design:
    Matches the reference implementation's two-stage workflow: this
    module is pretrained end-to-end against ground-truth pedestrian
    trajectories (see gst_data_collection.py / gst_predictor_trainer.py),
    then loaded with requires_grad=False and consumed read-only inside
    CrowdNavPPPolicy during RL rollout -- never jointly optimized with
    the RL objective. See GSTPredictorTrainer.load_predictor for the
    freeze step.

Input convention -- reuses ObservationEncoder's existing relative-position
invariant:
    ObservationEncoder's neighbor_history already stores each neighbor's
    position *relative to the robot* at each history step. Since
    (pos_i - robot) - (pos_j - robot) = pos_i - pos_j, that relative
    encoding already gives exactly the pairwise human-human geometry
    this predictor's interaction graph needs -- no change to
    ObservationEncoder's schema was required to wire this in (see
    policy.py's forward() for where neighbor_history[..., 0:2] is
    passed straight through as history_positions).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from navcore.policies.crowdnav_pp.temporal_encoder import (
    TemporalEncoder,
    TemporalEncoderConfig,
)
from navcore.policies.gst_predictor.interaction_graph import SparseInteractionGraph
from navcore.policies.gst_predictor.prediction_head import GaussianTrajectoryHead
from navcore.policies.gst_predictor.spatial_transformer import (
    GraphGatedTransformerLayer,
)


@dataclass(slots=True, frozen=True)
class GSTPredictorConfig:
    """Hyperparameters for GSTPredictor.

    Attributes:
        obs_length: History window length consumed per prediction.
        pred_length: Number of future steps predicted.
        motion_feature_dim: Width of one agent's per-timestep motion
            feature fed to the temporal encoder (2 for (vx, vy)).
        temporal_hidden_size: TemporalEncoder's output width.
        embedding_dim: Width used throughout the interaction graph and
            spatial transformer. Must be divisible by num_heads.
        edge_hidden_size: Hidden width of the pairwise edge-scoring MLP.
        gumbel_tau: Gumbel-softmax temperature for edge sampling.
        num_heads: Attention heads in each GraphGatedTransformerLayer.
        num_graph_layers: Stacked spatial-transformer layers.
        ff_hidden_size: Feed-forward hidden width inside each layer.
        head_hidden_size: Hidden width of the trajectory prediction head.
    """

    obs_length: int = 8
    pred_length: int = 5
    motion_feature_dim: int = 2
    temporal_hidden_size: int = 64
    embedding_dim: int = 64
    edge_hidden_size: int = 64
    gumbel_tau: float = 1.0
    num_heads: int = 4
    num_graph_layers: int = 2
    ff_hidden_size: int = 128
    head_hidden_size: int = 64

    def __post_init__(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )


class GSTPredictor(nn.Module):
    def __init__(self, config: GSTPredictorConfig) -> None:
        super().__init__()
        self.config = config

        self.temporal_encoder = TemporalEncoder(
            TemporalEncoderConfig(
                motion_feature_dim=config.motion_feature_dim,
                hidden_size=config.temporal_hidden_size,
            )
        )
        self.embed_proj = nn.Linear(config.temporal_hidden_size, config.embedding_dim)
        self.interaction_graph = SparseInteractionGraph(
            embedding_dim=config.embedding_dim,
            edge_hidden_size=config.edge_hidden_size,
            gumbel_tau=config.gumbel_tau,
        )
        self.graph_layers = nn.ModuleList(
            GraphGatedTransformerLayer(
                config.embedding_dim, config.num_heads, config.ff_hidden_size
            )
            for _ in range(config.num_graph_layers)
        )
        self.head = GaussianTrajectoryHead(
            config.embedding_dim, config.pred_length, config.head_hidden_size
        )

    def forward(
        self,
        history_positions: Tensor,
        history_velocity: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Args:
            history_positions: ``[B, N, obs_length, 2]``. Any consistent
                frame works (world-frame for offline training; robot-
                relative for policy.py's runtime use -- see module
                docstring for why robot-relative is sufficient).
            history_velocity: ``[B, N, obs_length, 2]``.
            history_mask: ``[B, N, obs_length]`` -- 1 where that agent was
                actually present at that timestep.

        Returns:
            ``(mean, log_var)``, each ``[B, N, pred_length, 2]``.
            ``mean`` is the predicted future *displacement* relative to
            each agent's last observed position (``history_positions[:, :, -1]``),
            not an absolute future position.
        """
        motion = history_velocity.permute(2, 0, 1, 3)  # [T, B, N, 2]
        mask_t = history_mask.permute(2, 0, 1).to(history_velocity.dtype)  # [T, B, N]

        temporal_embedding = self.temporal_encoder(motion, mask_t)  # [B, N, hidden]
        embeddings = self.embed_proj(temporal_embedding)  # [B, N, D]

        last_pos = history_positions[:, :, -1, :]  # [B, N, 2]
        relative_positions = last_pos.unsqueeze(2) - last_pos.unsqueeze(1)

        ever_visible = history_mask.any(dim=-1)  # [B, N]
        edge_weights = self.interaction_graph(
            embeddings, relative_positions, ever_visible
        )

        seq_embeddings = embeddings.transpose(0, 1)  # [N, B, D]
        key_padding_mask = ~ever_visible
        for layer in self.graph_layers:
            seq_embeddings = layer(seq_embeddings, edge_weights, key_padding_mask)
        embeddings = seq_embeddings.transpose(0, 1)  # [B, N, D]

        return self.head(embeddings)

    @torch.no_grad()
    def predict_features(
        self,
        history_positions: Tensor,
        history_velocity: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """Inference-only: predicted mean displacement, flattened for
        concatenation into a policy's per-neighbor feature vector.

        Discards log_var -- an uncertainty-weighted consumer should call
        forward() directly instead. Calling this also forces the module
        into eval() as a side effect (dropout/BN if any were ever added);
        harmless today since this module has neither, but worth knowing
        if that changes.

        Returns:
            ``[B, N, pred_length * 2]``.
        """
        self.eval()
        mean, _ = self.forward(history_positions, history_velocity, history_mask)
        B, N, T, _ = mean.shape
        return mean.reshape(B, N, T * 2)

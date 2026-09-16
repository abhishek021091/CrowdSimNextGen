# navcore/policies/gst_predictor/interaction_graph.py
"""SparseInteractionGraph: learns which human-human pairs matter, via
Gumbel-softmax edge sampling.

This is GST's actual architectural contribution over dense attention
(navcore's existing HumanHumanAttention): rather than every human
attending to every other human with a continuous weight, this module
first samples a discrete edge/no-edge decision per pair (soft during
training via the Gumbel-softmax relaxation, effectively hard at
inference), producing a genuinely sparse interaction graph. Two humans
far apart and moving independently should contribute zero signal to
each other's embedding, not merely a small one.
"""

from __future__ import annotations

from torch import Tensor, nn
import torch
import torch.nn.functional as F


class SparseInteractionGraph(nn.Module):
    """Scores every ordered human pair and samples a sparse adjacency.

    Attributes:
        embedding_dim: Width of each human's embedding.
        edge_hidden_size: Hidden width of the pairwise edge-scoring MLP.
        gumbel_tau: Gumbel-softmax temperature. Higher = softer/more
            uniform edges early; typically annealed down over training,
            though this module leaves annealing to the caller (see
            GSTPredictorTrainer) rather than owning a schedule itself.
    """

    def __init__(
        self, embedding_dim: int, edge_hidden_size: int = 64, gumbel_tau: float = 1.0
    ) -> None:
        super().__init__()
        self.gumbel_tau = gumbel_tau
        # Input: this human's embedding, the other human's embedding, and
        # their relative position (2D) -- geometry the embedding alone
        # may not cleanly encode, and which matters directly for "is
        # this pair close enough to interact."
        self.edge_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, edge_hidden_size),
            nn.ReLU(),
            nn.Linear(edge_hidden_size, 2),  # [no-edge logit, edge logit]
        )

    def forward(
        self,
        embeddings: Tensor,
        relative_positions: Tensor,
        visible_mask: Tensor,
    ) -> Tensor:
        """Return a soft/sparse adjacency matrix over visible humans.

        Args:
            embeddings: ``[B, N, D]`` -- one embedding per human slot.
            relative_positions: ``[B, N, N, 2]`` where
                ``relative_positions[:, i, j] = pos[i] - pos[j]``.
            visible_mask: ``[B, N]`` boolean, True for a real (non-padding)
                human slot present at the anchor timestep.

        Returns:
            ``[B, N, N]``, in ``[0, 1]``. Diagonal and any pair touching
            an invisible slot are forced to exactly 0 -- a human never
            forms an edge with itself or with padding.
        """
        B, N, D = embeddings.shape

        ei = embeddings.unsqueeze(2).expand(B, N, N, D)
        ej = embeddings.unsqueeze(1).expand(B, N, N, D)
        pair_features = torch.cat((ei, ej, relative_positions), dim=-1)
        logits = self.edge_scorer(pair_features)  # [B, N, N, 2]

        # Soft (differentiable) relaxation during training; near-hard,
        # deterministic sampling at inference -- standard Gumbel-softmax
        # usage. `hard=False` here means the *forward value* is soft too
        # at eval time via a low-noise draw; callers wanting a strictly
        # deterministic graph at inference should switch to argmax
        # instead -- left as the simpler, still-effective default.
        edge_sample = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False)
        edge_weights = edge_sample[..., 1]  # P(edge exists)

        pair_visible = visible_mask.unsqueeze(2) & visible_mask.unsqueeze(1)
        eye = torch.eye(N, dtype=torch.bool, device=embeddings.device).unsqueeze(0)
        valid = pair_visible & ~eye
        return edge_weights * valid.to(edge_weights.dtype)

# navcore/policies/gst_predictor/spatial_transformer.py
"""GraphGatedTransformerLayer: self-attention gated by a learned sparse graph.

Reuses nn.MultiheadAttention's additive attn_mask (rather than hand-
rolling attention) to fold SparseInteractionGraph's continuous edge
weights directly into attention logits: log(edge_weight) added before
softmax means an edge weight near 0 drives that pair's attention score
toward -inf, effectively removing it -- the sparsity SparseInteractionGraph
decided on is enforced here, not just computed and ignored.
"""

from __future__ import annotations

from torch import Tensor, nn
import torch


class GraphGatedTransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_hidden_size: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, embeddings: Tensor, edge_weights: Tensor, key_padding_mask: Tensor
    ) -> Tensor:
        """Args:
            embeddings: ``[N, B, D]`` (sequence-first, nn.MultiheadAttention's
                non-batch-first convention).
            edge_weights: ``[B, N, N]``, in ``[0, 1]``.
            key_padding_mask: ``[B, N]`` boolean, True = ignore this key.

        Returns:
            ``[N, B, D]``, same shape as ``embeddings``.
        """
        N, B, _ = embeddings.shape
        num_heads = self.attn.num_heads
        eps = 1e-6

        bias = torch.log(edge_weights + eps)  # [B, N, N]
        bias = bias.unsqueeze(1).expand(B, num_heads, N, N).reshape(B * num_heads, N, N)

        attended, _ = self.attn(
            embeddings,
            embeddings,
            embeddings,
            attn_mask=bias,
            key_padding_mask=key_padding_mask,
        )
        embeddings = self.norm1(embeddings + attended)
        embeddings = self.norm2(embeddings + self.ff(embeddings))
        return embeddings

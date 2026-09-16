# navcore/policies/gst_predictor/prediction_head.py
"""GaussianTrajectoryHead: multi-step future-displacement prediction with
per-step aleatoric uncertainty.

Uncertainty is a real part of GST's design, not decoration: the
CrowdNav++ paper explicitly downweights uncertain predictions rather
than treating every predicted future position as equally trustworthy.
This head produces (mean, log_var) per future step so that trust; the
current wiring in gst_predictor.py's predict_features() discards
log_var at inference for simplicity -- flagged there, not silently
dropped -- so uncertainty-weighted downstream use is a real, scoped
follow-up, not implemented here.
"""

from __future__ import annotations

import math

from torch import Tensor, nn


class GaussianTrajectoryHead(nn.Module):
    def __init__(
        self, embedding_dim: int, pred_length: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.pred_length = pred_length
        self.mean_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_length * 2),
        )
        self.log_var_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_length * 2),
        )

    def forward(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Args:
            embeddings: ``[B, N, D]``.

        Returns:
            ``(mean, log_var)``, each ``[B, N, pred_length, 2]``. ``log_var``
            is clamped to ``[-10, 10]`` -- unclamped log-variance heads are
            a known source of NLL-training instability (a runaway negative
            log_var makes the NLL loss diverge toward ``-inf``).
        """
        B, N, _ = embeddings.shape
        mean = self.mean_head(embeddings).view(B, N, self.pred_length, 2)
        log_var = self.log_var_head(embeddings).view(B, N, self.pred_length, 2)
        log_var = log_var.clamp(-10.0, 10.0)
        return mean, log_var


def gaussian_nll_loss(
    mean: Tensor, log_var: Tensor, target: Tensor, valid_mask: Tensor
) -> Tensor:
    """Masked, per-step diagonal-Gaussian negative log-likelihood.

    Args:
        mean, log_var, target: ``[B, N, T, 2]``.
        valid_mask: ``[B, N, T]`` -- 0 for any step with no ground-truth
            future position (agent left the episode, or the window ran
            past the end of a recorded trajectory).

    Returns:
        Scalar loss, averaged only over valid (agent, timestep) entries.
    """
    var = log_var.exp()
    nll = 0.5 * (log_var + (target - mean) ** 2 / var + math.log(2 * math.pi))
    nll = nll.sum(dim=-1)  # sum over (x, y) -> [B, N, T]

    mask = valid_mask.to(nll.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (nll * mask).sum() / denom

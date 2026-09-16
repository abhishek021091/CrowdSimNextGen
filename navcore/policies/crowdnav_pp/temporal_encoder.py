"""Per-agent temporal encoder for navcore's GST-equivalent trajectory predictor.

Ported from the ``lstm`` branch of ``st_model.forward`` (gst_updated/
src/gumbel_social_transformer/st_model.py of
github.com/Shuijing725/CrowdNav_Prediction_AttnGraph). See
``navcore/policies/crowdnav_pp/attention.py``'s module docstring for the
general porting rationale shared across this whole port (typed configs,
explicit feature widths, no shared ``argparse.Namespace``).

Which of the original's two LSTM variants this ports, and why:
    The original has two temporal-encoding code paths, selected by
    ``args.temporal``:

    - ``'lstm'``: loops one timestep at a time, explicitly freezing
      (not updating) each pedestrian's hidden/cell state on any
      timestep they aren't visible -- ``ht = htp * mask + ht * (1-mask)``.
      Correct: an invisible pedestrian's belief state simply doesn't
      change until they're seen again.
    - ``'faster_lstm'`` (what the original's actually-shipped, published
      checkpoint uses): zeroes each pedestrian's *input* on invisible
      timesteps, then runs the whole sequence through ``nn.LSTM`` in a
      single call. Faster (one cuDNN call instead of a Python loop), but
      not equivalent -- a zero input still updates the LSTM's hidden
      state through its own recurrence; it does not freeze it. This is a
      genuine, acknowledged approximation traded for training speed.

    navcore is not loading the original's pretrained weights (different
    simulator, different observation schema -- a checkpoint trained on
    their features wouldn't transfer), so there is no compatibility
    reason to inherit the approximation. Per this project's stated
    principle ordering (correctness before performance), this module
    ports the ``'lstm'`` (masked, correct) variant. If profiling later
    shows the per-timestep Python loop is a real bottleneck, the
    ``'faster_lstm'`` approximation is a documented, available fallback
    -- not implemented here, so that choice stays visible rather than
    silently baked in.

Feature convention -- reusing navcore's own history buffers:
    ``ObservationEncoder`` already accumulates ``neighbor_history``
    (``[history_steps, max_neighbors, 5]``: rel_px, rel_py, vx, vy,
    radius) and ``neighbor_history_mask`` for exactly this kind of use.
    This encoder consumes a caller-selected slice of that feature vector
    (``motion_feature_dim``, e.g. 2 for just ``vx, vy``) rather than
    inventing a new history representation -- see
    ``navcore.gym_wrapper.observation_encoder`` for the source buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class TemporalEncoderConfig:
    """Hyperparameters for :class:`TemporalEncoder`.

    Attributes:
        motion_feature_dim: Width of one agent's per-timestep motion
            feature (e.g. 2 for ``(vx, vy)`` sliced out of navcore's
            ``neighbor_history``).
        hidden_size: LSTM hidden state width -- this becomes each
            agent's temporal embedding, consumed downstream by the edge
            selector / spatial attention (a later piece of this port).
    """

    motion_feature_dim: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.motion_feature_dim <= 0:
            raise ValueError(
                f"motion_feature_dim must be positive, got {self.motion_feature_dim!r}."
            )
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size!r}.")


class TemporalEncoder(nn.Module):
    """Masked per-agent LSTM over a fixed-length motion history window.

    Processes ``history_steps`` timesteps one at a time (an explicit
    Python loop, not a single batched ``nn.LSTM`` call -- see module
    docstring for why that's a deliberate correctness choice, not an
    oversight), holding each agent's hidden/cell state exactly constant
    on any timestep that agent wasn't actually visible.
    """

    def __init__(self, config: TemporalEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.cell = nn.LSTMCell(config.motion_feature_dim, config.hidden_size)

    def forward(self, motion_history: Tensor, history_mask: Tensor) -> Tensor:
        """Return each agent's final hidden state after the whole window.

        Args:
            motion_history: ``[history_steps, nenv, max_agents,
                motion_feature_dim]`` -- oldest timestep first, matching
                navcore's ``neighbor_history`` convention.
            history_mask: ``[history_steps, nenv, max_agents]`` boolean,
                ``True`` where that agent was actually visible at that
                timestep -- matches ``neighbor_history_mask``'s
                semantics (there, ``1``/``0``; cast before calling).

        Returns:
            ``[nenv, max_agents, hidden_size]`` -- each agent's temporal
            embedding after processing the full history window. An
            agent invisible for the *entire* window gets an all-zero
            embedding (LSTM state never leaves its zero initialization),
            which downstream masking (this same ``history_mask``'s last
            timestep, or a caller-derived "ever visible" mask) must
            still account for -- this encoder does not decide what an
            "absent agent" embedding should mean to its callers.

        Raises:
            ValueError: If ``motion_history`` and ``history_mask``
                disagree on ``[history_steps, nenv, max_agents]``, or if
                the last dimension of ``motion_history`` doesn't match
                ``config.motion_feature_dim``.
        """
        history_steps, nenv, max_agents, feature_dim = motion_history.shape
        if tuple(history_mask.shape) != (history_steps, nenv, max_agents):
            raise ValueError(
                f"history_mask shape {tuple(history_mask.shape)} does not "
                f"match motion_history's leading dims "
                f"{(history_steps, nenv, max_agents)}."
            )
        if feature_dim != self.config.motion_feature_dim:
            raise ValueError(
                f"motion_history's last dim is {feature_dim}, but this "
                f"encoder was configured for motion_feature_dim="
                f"{self.config.motion_feature_dim}."
            )

        batch = nenv * max_agents
        hidden = motion_history.new_zeros(batch, self.config.hidden_size)
        cell = motion_history.new_zeros(batch, self.config.hidden_size)

        flat_motion = motion_history.reshape(history_steps, batch, feature_dim)
        flat_mask = history_mask.reshape(history_steps, batch).to(motion_history.dtype)

        for t in range(history_steps):
            new_hidden, new_cell = self.cell(flat_motion[t], (hidden, cell))
            step_mask = flat_mask[t].unsqueeze(
                -1
            )  # [batch, 1], broadcasts over hidden_size
            # Freeze (do not update) any agent's state on a timestep it
            # wasn't visible -- the correctness property the module
            # docstring contrasts against the original's 'faster_lstm'.
            hidden = new_hidden * step_mask + hidden * (1.0 - step_mask)
            cell = new_cell * step_mask + cell * (1.0 - step_mask)

        return hidden.reshape(nenv, max_agents, self.config.hidden_size)

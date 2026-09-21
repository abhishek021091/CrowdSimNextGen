"""Recurrent node update for CrowdNav++'s interaction-graph policy.

Ported from ``EndRNN`` (rl/networks/selfAttn_srnn_temp_node.py of
github.com/Shuijing725/CrowdNav_Prediction_AttnGraph). See
``attention.py``'s module docstring for the general porting rationale.

Scope: inference-time only, by design, not an oversight
-----------------------------------------------------------
The original's GRU update lives inside ``RNNBase._forward_gru``, which
has two branches:

    1. A single-timestep path (``x.size(0) == hxs.size(0)``), used
       during rollout collection: one tick, hidden state carried in and
       out explicitly, with a "done mask" zeroing the incoming hidden
       state wherever an episode just restarted.
    2. A multi-timestep batched path (``x.size(0) == T``), used only
       during PPO's training update: an entire stored rollout of ``T``
       steps is pushed through the GRU in one call, with extra logic
       (the ``has_zeros`` splitting loop) to correctly reset hidden state
       *mid-sequence* wherever an episode boundary falls inside the
       batched window.

This class ports only branch (1). Branch (2)'s correctness can only be
verified against a real training loop that stores multi-step rollouts a
specific way -- and no such loop exists in navcore yet, nor has its
batching scheme been decided. Porting that logic now would mean writing
untestable code that also silently commits navcore to the original
project's specific rollout-storage layout. When navcore's training loop
is designed, the batched path becomes its own class, built and verified
against that actual loop -- not guessed at here. See conversation
history for this explicit tradeoff.

Note this class does NOT carry navcore's usual ``[seq_len, nenv, ...]``
leading-axis convention (unlike ``HumanHumanAttention``/
``RobotHumanAttention``): ``nn.GRUCell``'s semantics are inherently
single-step, so a ``seq_len`` axis has nothing to mean here. Callers
operating on ``[seq_len, nenv, ...]`` tensors from the earlier pieces
index/squeeze ``seq_len`` (always 1 at inference) before calling this.

Static-obstacle context used to enter here as a third, unconditionally-
added branch (``obstacle_embed``). It has moved upstream into
``RobotHumanAttention`` instead: the obstacle summary is now one more
key/value token the robot's query attends over, alongside humans, so the
network learns *when* obstacle context matters rather than always
receiving it at full strength. This class is back to its original
two-branch (robot, crowd-context) GRU input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class RecurrentNodeUpdateConfig:
    input_dim: int
    node_embedding_size: int = 64
    rnn_hidden_size: int = 128
    output_size: int = 256

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "node_embedding_size",
            "rnn_hidden_size",
            "output_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive, got {getattr(self, name)!r}."
                )


class RecurrentNodeUpdate(nn.Module):
    def __init__(self, config: RecurrentNodeUpdateConfig) -> None:
        super().__init__()
        self.config = config

        self.robot_embed = nn.Sequential(
            nn.Linear(config.input_dim, config.node_embedding_size), nn.ReLU()
        )
        self.context_embed = nn.Sequential(
            nn.Linear(config.input_dim, config.node_embedding_size), nn.ReLU()
        )

        self.gru_cell = nn.GRUCell(
            config.node_embedding_size * 2, config.rnn_hidden_size
        )
        for name, param in self.gru_cell.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        self.output_linear = nn.Linear(config.rnn_hidden_size, config.output_size)

    def initial_hidden_state(
        self, nenv: int, device: torch.device | None = None
    ) -> Tensor:
        return torch.zeros(nenv, self.config.rnn_hidden_size, device=device)

    def forward(
        self,
        robot_embedding: Tensor,
        crowd_context: Tensor,
        hidden_state: Tensor,
        not_done_mask: Tensor,
        obstacle_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Advance the recurrent state by one tick.

        Args:
            robot_embedding: ``[nenv, input_dim]``.
            crowd_context: ``[nenv, input_dim]``.
            hidden_state: ``[nenv, rnn_hidden_size]``.
            not_done_mask: ``[nenv]`` -- ``1.0`` to carry hidden_state
                forward, ``0.0`` to reset it before this tick.
            obstacle_context: ``[nenv, input_dim]``, required if and
                only if ``config.use_obstacle_context`` is True (e.g.
                an ``ObstacleEncoder`` embedding, projected to
                ``input_dim``). ``None`` when obstacle encoding is
                disabled for this policy.

        Returns:
            ``(output, new_hidden_state)``.

        Raises:
            ValueError: On any shape mismatch, or if
                ``obstacle_context``'s presence disagrees with
                ``config.use_obstacle_context``.
        """
        nenv, input_dim = robot_embedding.shape
        if input_dim != self.config.input_dim:
            raise ValueError(
                f"robot_embedding's last dim is {input_dim}, but this "
                f"layer was configured for input_dim={self.config.input_dim}."
            )
        if tuple(crowd_context.shape) != (nenv, self.config.input_dim):
            raise ValueError(
                f"crowd_context shape {tuple(crowd_context.shape)} does not "
                f"match expected {(nenv, self.config.input_dim)}."
            )
        if tuple(hidden_state.shape) != (nenv, self.config.rnn_hidden_size):
            raise ValueError(
                f"hidden_state shape {tuple(hidden_state.shape)} does not "
                f"match expected {(nenv, self.config.rnn_hidden_size)}."
            )
        if tuple(not_done_mask.shape) != (nenv,):
            raise ValueError(
                f"not_done_mask shape {tuple(not_done_mask.shape)} does not "
                f"match expected {(nenv,)}."
            )

        concat = torch.cat(
            (self.robot_embed(robot_embedding), self.context_embed(crowd_context)),
            dim=-1,
        )

        reset_hidden = hidden_state * not_done_mask.unsqueeze(-1)
        new_hidden = self.gru_cell(concat, reset_hidden)
        output = self.output_linear(new_hidden)

        return output, new_hidden

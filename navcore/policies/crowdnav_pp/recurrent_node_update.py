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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class RecurrentNodeUpdateConfig:
    """Hyperparameters for :class:`RecurrentNodeUpdate`.

    Attributes:
        input_dim: Width of both the robot embedding and the crowd
            context vector arriving at this layer (the ``embedding_dim``
            shared by ``RobotStateEncoder``/``RobotHumanAttention``).
        node_embedding_size: Width each of the two inputs is separately
            projected to before concatenation. 64 in the original
            (``human_node_embedding_size``).
        rnn_hidden_size: GRU hidden state width. 128 in the original
            (``human_node_rnn_size``).
        output_size: Width of the final projected output, fed to the
            actor/critic heads downstream. 256 in the original
            (``human_node_output_size``).
    """

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
    """One recurrent tick: fold robot + crowd-context into GRU memory.

    This is where the network gets memory across ticks -- everything
    upstream (``RobotStateEncoder``, ``HumanHumanAttention``,
    ``RobotHumanAttention``) is a pure per-tick function of the current
    observation; this class is the only stateful piece, and its state
    (the GRU hidden vector) is owned by the caller, not this module --
    consistent with the project's "no hidden state inside a component;
    the caller threads it explicitly" convention (mirrors how ``Step``
    threads simulation state rather than a ``Mission`` keeping its own
    copy of the world).
    """

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
        # Matches the original's RNNBase initialization: orthogonal
        # weights, zero biases -- a standard, well-behaved RNN init, kept
        # faithfully rather than left at PyTorch's default.
        for name, param in self.gru_cell.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        self.output_linear = nn.Linear(config.rnn_hidden_size, config.output_size)

    def initial_hidden_state(
        self, nenv: int, device: torch.device | None = None
    ) -> Tensor:
        """Return a zeroed hidden state for ``nenv`` parallel environments.

        Convenience for callers starting a fresh rollout -- equivalent
        to, but more discoverable than, ``torch.zeros(nenv,
        config.rnn_hidden_size)``.
        """
        return torch.zeros(nenv, self.config.rnn_hidden_size, device=device)

    def forward(
        self,
        robot_embedding: Tensor,
        crowd_context: Tensor,
        obstacle_context: Tensor,
        hidden_state: Tensor,
        not_done_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Advance the recurrent state by one tick.

        Args:
            robot_embedding: ``[nenv, input_dim]`` -- e.g.
                ``RobotStateEncoder``'s output for this tick, with the
                ``seq_len`` axis already squeezed out by the caller.
            crowd_context: ``[nenv, input_dim]`` -- e.g.
                ``RobotHumanAttention``'s output for this tick, same
                squeeze convention.
            hidden_state: ``[nenv, rnn_hidden_size]`` -- the previous
                tick's hidden state (see :meth:`initial_hidden_state` for
                episode start).
            not_done_mask: ``[nenv]``, ``1.0`` to carry ``hidden_state``
                forward, ``0.0`` to reset it to zero before this tick's
                update -- i.e. ``1.0 - done`` from the previous tick's
                step result. Matches the original's ``masks`` convention.

        Returns:
            ``(output, new_hidden_state)``: ``output`` is
            ``[nenv, output_size]``, ready for the actor/critic heads;
            ``new_hidden_state`` is ``[nenv, rnn_hidden_size]``, to be
            passed back in on the next tick.

        Raises:
            ValueError: If any input's shape is inconsistent with
                ``config`` or with the others.
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
        if tuple(obstacle_context.shape) != (nenv, self.config.input_dim):
            raise ValueError(
                f"obstacle_context shape {tuple(obstacle_context.shape)} does not "
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

        robot_branch = self.robot_embed(robot_embedding)
        context_branch = self.context_embed(crowd_context)
        obstacle_branch = self.context_embed(obstacle_context)
        concat = torch.cat((robot_branch, context_branch, obstacle_branch), dim=-1)

        reset_hidden = hidden_state * not_done_mask.unsqueeze(-1)
        new_hidden = self.gru_cell(concat, reset_hidden)
        output = self.output_linear(new_hidden)

        return output, new_hidden

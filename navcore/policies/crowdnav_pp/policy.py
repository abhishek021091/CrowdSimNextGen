"""CrowdNavPPPolicyConfig / CrowdNavPPPolicy: the assembled CrowdNav++ actor-critic.

Wires together the seven modules ported from
``Shuijing725/CrowdNav_Prediction_AttnGraph`` (see each sibling module's own
docstring for its individual porting rationale) into the single recurrent
policy the paper describes: per-tick robot + neighbor features in, an
action distribution + value estimate + updated recurrent hidden state out.
This is the file that finally resolves the composition questions each of
those modules deliberately left open for "whoever builds this piece" --
see the three design-decision notes below.

Design decision -- spatial_edge_feature_dim composition:
    ``HumanHumanAttention`` documents that its ``spatial_edge_feature_dim``
    is "whatever encodes navcore's observation," left unresolved pending
    this exact assembly. This class feeds it the concatenation of (a)
    each neighbor's instantaneous ``ObservationEncoder`` features
    (rel_px, rel_py, vx, vy, radius -- 5-dim) and (b) that neighbor's
    temporal embedding from ``TemporalEncoder``, run over
    ``ObservationEncoder``'s ``neighbor_history`` buffer. This gives
    human-human attention both "where/how fast right now" and "how has
    this human been moving," without a live future-trajectory predictor
    -- the GST-based 12-dim variant flagged as an open question in
    ``human_human_attention.py`` is still not implemented; this is the
    "no live predictor yet" branch of that fork, by design.

Design decision -- embedding-width unification:
    ``RobotStateEncoder``, ``RobotHumanAttention``, and
    ``RecurrentNodeUpdate.input_dim`` must all agree on one shared width
    (the robot's own embedding and the crowd-context vector live in the
    same space, per ``RecurrentNodeUpdate.forward``). ``HumanHumanAttention``'s
    internal ``embedding_size`` is independent and, per its own
    docstring, may need "projecting back down" before reaching
    ``RobotHumanAttention`` -- this class owns that projection
    (``self._human_embed_down``) rather than adding it inside either
    attention module, since neither module should need to know about the
    other's configured width.

Design decision -- the "zero visible humans" case:
    Both ``HumanHumanAttention`` and ``RobotHumanAttention`` raise loudly
    on a fully-masked (seq, env) slot rather than silently producing
    NaNs, and both docstrings flag "synthetic dummy human at slot 0" as
    the original paper's own resolution, deliberately left unimplemented
    at that layer. This class implements it here, once, at the boundary
    where real ``ObservationEncoder`` output enters the network: any
    environment currently seeing zero neighbors gets slot 0 forced
    visible before either attention layer runs (see
    ``_substitute_dummy_human`` for the caveat this introduces).
    Architecturally this is the only correct place for it -- doing it
    inside the attention modules would force every caller, even ones
    that can guarantee non-empty visibility another way, to pay for it.

Recurrent-state convention:
    Matches ``RecurrentNodeUpdate``'s own convention (see its docstring):
    the hidden state is threaded explicitly by the caller, never stored
    on this module. ``forward()`` takes and returns it. This keeps
    ``CrowdNavPPPolicy`` a plain, RL-framework-agnostic ``nn.Module`` --
    an SB3/RLlib/TorchRL adapter can wrap it however that framework wants
    its recurrent state carried, without this class knowing any of them
    exist (project's RL-framework-agnostic principle).

Not yet resolved here (flagged, not silently decided):
    - This class is single-tick-only (``seq_len`` is always 1 inside
      ``forward()``), mirroring ``RecurrentNodeUpdate``'s own
      inference-only scope -- see that module's docstring for why the
      batched multi-timestep training path is deferred. A PPO training
      loop built on this class needs its own batched forward path, once
      that loop's rollout-storage layout is decided.
    - No SB3/RLlib adapter yet. This module intentionally stops at "a
      correct, testable ``nn.Module``" -- wiring it into a specific RL
      framework's policy base class is a separate, framework-specific
      decision.
    - No shape/gradient smoke test yet, breaking from this project's
      usual "each module is shape-tested before the next is built"
      workflow -- this is the assembly step itself, so there was nothing
      to test it against until now. Recommended immediate next step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Independent

from navcore.policies.crowdnav_pp.actiton_distribution import (
    DiagGaussianHead,
    DiagGaussianHeadConfig,
    select_action,
)
from navcore.policies.crowdnav_pp.actor_crititc_head import (
    ActorCriticHeads,
    ActorCriticHeadsConfig,
)
from navcore.policies.crowdnav_pp.human_human_attention import (
    HumanHumanAttention,
    HumanHumanAttentionConfig,
)
from navcore.policies.crowdnav_pp.obstacle_encoder import ObstacleEncoder
from navcore.policies.crowdnav_pp.recurrent_node_update import (
    RecurrentNodeUpdate,
    RecurrentNodeUpdateConfig,
)
from navcore.policies.crowdnav_pp.robot_human_attention import (
    RobotHumanAttention,
    RobotHumanAttentionConfig,
)
from navcore.policies.crowdnav_pp.robot_state_encoder import (
    RobotStateEncoder,
    RobotStateEncoderConfig,
)
from navcore.policies.crowdnav_pp.temporal_encoder import (
    TemporalEncoder,
    TemporalEncoderConfig,
)
from navcore.policies.gst_predictor.gst_predictor import GSTPredictor

#: Index range of (vx, vy) within one ObservationEncoder neighbor feature
#: vector -- see navcore.gym_wrapper.observation_encoder's module
#: docstring: layout is (rel_px, rel_py, vx, vy, radius). This is a real
#: coupling point: if that layout ever changes, this slice silently
#: breaks. Left as a slice (matching ObservationEncoder's own raw-tuple
#: convention) rather than invented as a new named-feature abstraction
#: here -- fixing that coupling belongs in ObservationEncoder itself, as
#: a separate, dedicated cleanup, not smuggled into this file.
_NEIGHBOR_MOTION_SLICE = slice(2, 4)


@dataclass(slots=True, frozen=True)
class CrowdNavPPPolicyConfig:
    """Hyperparameters for the assembled CrowdNav++ policy.

    Every cross-module width is derived from a single shared field here
    (see module docstring's "embedding-width unification") rather than
    duplicated per sub-config, so there is no way for two sub-modules to
    end up configured with mismatched widths.

    Attributes:
        robot_feature_dim: Width of ObservationEncoder's ``"robot"``
            feature vector (8 for navcore's current encoder).
        neighbor_feature_dim: Width of one instantaneous neighbor feature
            (5 for navcore's current encoder: rel_px, rel_py, vx, vy,
            radius).
        temporal_hidden_size: Width of each neighbor's TemporalEncoder
            embedding, concatenated onto its instantaneous features to
            form spatial_edge_feature_dim (see module docstring).
        interaction_embedding_dim: Shared embedding width for the
            robot's own encoding, the robot-human attention output, and
            RecurrentNodeUpdate's input.
        human_human_embedding_size: HumanHumanAttention's internal
            embedding width (512 in the original paper).
        human_human_num_heads: HumanHumanAttention's attention head
            count. Must evenly divide human_human_embedding_size.
        robot_human_attention_size: RobotHumanAttention's internal
            projection width (64 in the original).
        node_embedding_size: RecurrentNodeUpdate's per-input projection
            width before concatenation (64 in the original).
        rnn_hidden_size: RecurrentNodeUpdate's GRU hidden width (128 in
            the original).
        node_output_size: RecurrentNodeUpdate's output width, also
            ActorCriticHeads.input_dim (256 in the original).
        actor_critic_hidden_size: Width of both actor/critic MLP towers,
            and DiagGaussianHead's input width (256 in the original).
        action_dim: Number of continuous action dimensions (2 for
            navcore's (vx, vy) velocity action space).
    """

    robot_feature_dim: int = 8
    neighbor_feature_dim: int = 5
    temporal_hidden_size: int = 32
    interaction_embedding_dim: int = 256
    human_human_embedding_size: int = 512
    human_human_num_heads: int = 8
    robot_human_attention_size: int = 64
    node_embedding_size: int = 64
    rnn_hidden_size: int = 128
    node_output_size: int = 256
    actor_critic_hidden_size: int = 256
    action_dim: int = 2
    use_gst_prediction: bool = False
    gst_pred_length: int = 5
    use_obstacle_encoder: bool = False

    @property
    def spatial_edge_feature_dim(self) -> int:
        base = self.neighbor_feature_dim + self.temporal_hidden_size
        if self.use_gst_prediction:
            base += self.gst_pred_length * 2
        return base


class CrowdNavPPPolicy(nn.Module):
    """The full CrowdNav++ recurrent actor-critic, assembled from its parts.

    A plain ``nn.Module`` with an explicit, RL-framework-agnostic
    interface: pass in one tick's batched observation plus the previous
    recurrent hidden state, get back an action distribution, a value
    estimate, and the new hidden state. See module docstring for the
    composition decisions this class resolves.

    Attributes:
        config: This policy's hyperparameters.
        gst_predictor: A pretrained GSTPredictor for making future trajectory predictions.
        obstacle_encoder: A pretrained ObstacleEncoder for encoding obstacle information.
    """

    def __init__(
        self,
        config: CrowdNavPPPolicyConfig,
        gst_predictor: GSTPredictor | None = None,
        obstacle_encoder: ObstacleEncoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        if config.use_gst_prediction and gst_predictor is None:
            raise ValueError(
                "use_gst_prediction=True requires a pretrained gst_predictor "
                "(see GSTPredictorTrainer.load_predictor) -- GST is trained "
                "separately, never jointly with this policy."
            )
        if config.use_obstacle_encoder and obstacle_encoder is None:
            raise ValueError(
                "use_obstacle_encoder=True requires a pretrained obstacle_encoder "
                "(see ObstacleEncoderTrainer.load_encoder) -- ObstacleEncoder is trained "
                "separately, never jointly with this policy."
            )
        self.gst_predictor = gst_predictor
        self.obstacle_encoder = obstacle_encoder
        self._obstacle_embed_down: nn.Module = nn.Identity()
        if obstacle_encoder is not None:
            self._obstacle_embed_down = (
                nn.Identity()
                if obstacle_encoder.config.output_dim
                == config.interaction_embedding_dim
                else nn.Linear(
                    obstacle_encoder.config.output_dim, config.interaction_embedding_dim
                )
            )
        self.robot_encoder = RobotStateEncoder(
            RobotStateEncoderConfig(
                robot_feature_dim=config.robot_feature_dim,
                embedding_dim=config.interaction_embedding_dim,
            )
        )
        self.temporal_encoder = TemporalEncoder(
            TemporalEncoderConfig(
                motion_feature_dim=(
                    _NEIGHBOR_MOTION_SLICE.stop - _NEIGHBOR_MOTION_SLICE.start
                ),
                hidden_size=config.temporal_hidden_size,
            )
        )
        self.human_human_attention = HumanHumanAttention(
            HumanHumanAttentionConfig(
                spatial_edge_feature_dim=config.spatial_edge_feature_dim,
                embedding_size=config.human_human_embedding_size,
                num_attention_heads=config.human_human_num_heads,
            )
        )
        self._human_embed_down: nn.Module = (
            nn.Identity()
            if config.human_human_embedding_size == config.interaction_embedding_dim
            else nn.Linear(
                config.human_human_embedding_size, config.interaction_embedding_dim
            )
        )
        self.robot_human_attention = RobotHumanAttention(
            RobotHumanAttentionConfig(
                embedding_dim=config.interaction_embedding_dim,
                attention_size=config.robot_human_attention_size,
            )
        )
        self.recurrent_update = RecurrentNodeUpdate(
            RecurrentNodeUpdateConfig(
                input_dim=config.interaction_embedding_dim,
                node_embedding_size=config.node_embedding_size,
                rnn_hidden_size=config.rnn_hidden_size,
                output_size=config.node_output_size,
                use_obstacle_context=config.use_obstacle_encoder,
            )
        )
        self.actor_critic_heads = ActorCriticHeads(
            ActorCriticHeadsConfig(
                input_dim=config.node_output_size,
                hidden_size=config.actor_critic_hidden_size,
            )
        )
        self.action_head = DiagGaussianHead(
            DiagGaussianHeadConfig(
                input_dim=config.actor_critic_hidden_size,
                action_dim=config.action_dim,
            )
        )

    def initial_hidden_state(
        self, nenv: int, device: torch.device | None = None
    ) -> Tensor:
        """Return a zeroed recurrent hidden state for ``nenv`` environments.

        Thin passthrough to ``RecurrentNodeUpdate.initial_hidden_state``
        -- exposed here so callers never need to reach past this class
        into its sub-modules.
        """
        return self.recurrent_update.initial_hidden_state(nenv, device)

    def forward(
        self,
        robot_features: Tensor,
        neighbor_features: Tensor,
        neighbor_mask: Tensor,
        neighbor_history: Tensor,
        neighbor_history_mask: Tensor,
        hidden_state: Tensor,
        not_done_mask: Tensor,
        ray_features: Tensor | None = None,
    ) -> tuple[Independent, Tensor, Tensor]:
        """Run one recurrent tick for a batch of environments.

        Args:
            robot_features: ``[nenv, robot_feature_dim]`` -- e.g.
                ObservationEncoder's ``"robot"`` output, stacked across
                environments.
            neighbor_features: ``[nenv, max_neighbors, neighbor_feature_dim]``
                -- ``"neighbors"``, stacked.
            neighbor_mask: ``[nenv, max_neighbors]`` -- ``"neighbor_mask"``,
                stacked. Nonzero for a real (non-padding) neighbor.
            neighbor_history: ``[nenv, history_steps, max_neighbors,
                neighbor_feature_dim]`` -- ``"neighbor_history"``, stacked.
                Note the axis order: ObservationEncoder produces
                history_steps first per environment; stacking across
                environments puts nenv first instead. This method
                transposes back to TemporalEncoder's expected
                ``[history_steps, nenv, ...]`` order internally.
            neighbor_history_mask: ``[nenv, history_steps, max_neighbors]``
                -- ``"neighbor_history_mask"``, stacked.
            hidden_state: ``[nenv, rnn_hidden_size]`` -- previous tick's
                recurrent state (see ``initial_hidden_state`` for episode
                start).
            not_done_mask: ``[nenv]`` -- ``1.0`` to carry ``hidden_state``
                forward, ``0.0`` to reset it (this tick starts a new
                episode for that environment). Forwarded verbatim to
                ``RecurrentNodeUpdate``.

        Returns:
            ``(distribution, value, new_hidden_state)``: ``distribution``
            is this tick's action distribution (``Independent(Normal(...),
            1)``, from ``DiagGaussianHead``); ``value`` is ``[nenv, 1]``;
            ``new_hidden_state`` is ``[nenv, rnn_hidden_size]``, to be
            passed back in next tick.
        """
        neighbor_mask = neighbor_mask.bool()
        neighbor_history_mask = neighbor_history_mask.bool()

        motion_history = neighbor_history[..., _NEIGHBOR_MOTION_SLICE].transpose(0, 1)
        history_mask = neighbor_history_mask.transpose(0, 1).to(neighbor_history.dtype)
        temporal_embedding = self.temporal_encoder(motion_history, history_mask)

        if self.config.use_gst_prediction:
            assert self.gst_predictor is not None
            # neighbor_history[..., 0:2] is ObservationEncoder's rel_px/rel_py --
            # robot-relative, which is sufficient for pairwise human-human
            # geometry (see gst_predictor.py's module docstring). transpose(1, 2)
            # swaps [nenv, history_steps, max_neighbors, ...] into GSTPredictor's
            # expected [nenv, max_neighbors, history_steps, ...].
            hist_pos = neighbor_history[..., 0:2].transpose(1, 2)
            hist_vel = neighbor_history[..., 2:4].transpose(1, 2)
            hist_mask = neighbor_history_mask.transpose(1, 2)
            pred_features = self.gst_predictor.predict_features(
                hist_pos, hist_vel, hist_mask
            )
            spatial_edge_features = torch.cat(
                (neighbor_features, temporal_embedding, pred_features), dim=-1
            )
        else:
            spatial_edge_features = torch.cat(
                (neighbor_features, temporal_embedding), dim=-1
            )

        # HumanHumanAttention/RobotHumanAttention carry an explicit
        # seq_len axis (see their own docstrings); this policy is
        # single-tick-only (see module docstring), so seq_len is always
        # exactly 1 here -- added before those two calls, squeezed back
        # off immediately after.
        human_features = spatial_edge_features.unsqueeze(0)
        visible_mask = neighbor_mask.unsqueeze(0)
        visible_mask = self._substitute_dummy_human(visible_mask)

        human_embeddings = self.human_human_attention(human_features, visible_mask)
        human_embeddings = self._human_embed_down(human_embeddings)

        robot_embedding = self.robot_encoder(robot_features).unsqueeze(0).unsqueeze(-2)
        crowd_context = self.robot_human_attention(
            robot_embedding, human_embeddings, visible_mask
        ).squeeze(0)
        robot_embedding = robot_embedding.squeeze(0).squeeze(-2)

        obstacle_context = None
        if self.config.use_obstacle_encoder:
            if ray_features is None:
                raise ValueError(
                    "config.use_obstacle_encoder=True but forward() was "
                    "called without ray_features."
                )
            assert self.obstacle_encoder is not None
            obstacle_embedding = self.obstacle_encoder(ray_features)
            obstacle_context = self._obstacle_embed_down(obstacle_embedding)
        elif ray_features is not None:
            raise ValueError(
                "ray_features was given but config.use_obstacle_encoder=False."
            )

        node_output, new_hidden_state = self.recurrent_update(
            robot_embedding,
            crowd_context,
            hidden_state,
            not_done_mask,
            obstacle_context=obstacle_context,
        )

        value, actor_features = self.actor_critic_heads(node_output)
        distribution = self.action_head(actor_features)

        return distribution, value, new_hidden_state

    def act(
        self,
        robot_features: Tensor,
        neighbor_features: Tensor,
        neighbor_mask: Tensor,
        neighbor_history: Tensor,
        neighbor_history_mask: Tensor,
        hidden_state: Tensor,
        not_done_mask: Tensor,
        ray_features: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convenience wrapper: run ``forward`` and sample an action from it.

        Returns:
            ``(action, log_prob, value, new_hidden_state)`` -- see
            ``select_action`` (``actiton_distribution.py``) for the
            action/log_prob pair's semantics, and ``forward`` for
            ``value``/``new_hidden_state``.
        """
        distribution, value, new_hidden_state = self.forward(
            robot_features,
            neighbor_features,
            neighbor_mask,
            neighbor_history,
            neighbor_history_mask,
            hidden_state,
            not_done_mask,
            ray_features=ray_features,
        )
        action, log_prob = select_action(distribution, deterministic=deterministic)
        return action, log_prob, value, new_hidden_state

    @staticmethod
    def _substitute_dummy_human(visible_mask: Tensor) -> Tensor:
        """Force slot 0 visible wherever an environment sees zero neighbors.

        See module docstring's "zero visible humans" design decision --
        this is the one place navcore implements the original paper's own
        documented workaround for a fully-masked attention row, rather
        than letting either attention module crash on it.

        Caveat, not silently hidden: the substituted slot's *feature*
        vector is left exactly as ObservationEncoder already zero-fills
        padding slots (relative position (0, 0), zero velocity, zero
        radius, zero temporal embedding) -- i.e. the dummy human reads to
        attention as "a human standing exactly on the robot." This is a
        real placeholder artifact, not a neutral no-op; it is an
        accepted approximation for now, worth revisiting if it shows up
        in training diagnostics (e.g. an unexplained bias in learned
        behavior for episodes with sparse crowds).

        Args:
            visible_mask: ``[1, nenv, max_neighbors]``, boolean.

        Returns:
            ``visible_mask``, with slot 0 marked visible for any
            ``(seq, env)`` that had no real visible neighbor. Returned
            unchanged (same object) if every slot already has at least
            one visible neighbor.
        """
        no_humans_visible = ~visible_mask.any(dim=-1)  # [1, nenv]
        if not bool(no_humans_visible.any()):
            return visible_mask

        visible_mask = visible_mask.clone()
        visible_mask[..., 0] |= no_humans_visible
        return visible_mask

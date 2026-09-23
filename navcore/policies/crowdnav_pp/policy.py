"""CrowdNavPPPolicyConfig / CrowdNavPPPolicy: the assembled CrowdNav++ actor-critic.

Wires together the modules ported from
``Shuijing725/CrowdNav_Prediction_AttnGraph`` (see each sibling module's own
docstring for its individual porting rationale) into the recurrent policy
the paper describes, PLUS a completely independent, from-scratch obstacle
branch (see "Obstacle branch" below) that replaced the previous 1D-CNN-
over-ray-features design.

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
    ``RobotStateEncoder``, ``RobotHumanAttention``, ``RobotObstacleAttention``,
    and ``RecurrentNodeUpdate.input_dim`` must all agree on one shared
    width (256 by default, ``interaction_embedding_dim``) for the robot's
    own embedding and every context vector fed into the GRU.
    ``HumanHumanAttention``'s internal ``embedding_size`` is independent
    and, per its own docstring, may need "projecting back down" before
    reaching ``RobotHumanAttention`` -- this class owns that projection
    (``self._human_embed_down``). The obstacle branch keeps its own,
    separate 128-d latent space (``RangeImageEncoderConfig.
    token_embedding_dim`` / ``RobotObstacleAttentionConfig.
    obstacle_embedding_dim``) and is projected up to 256-d only *after*
    ``RobotObstacleAttention`` runs -- see "Obstacle branch" below for why.

Design decision -- the "zero visible humans" case:
    Both ``HumanHumanAttention`` and ``RobotHumanAttention`` raise loudly
    on a fully-masked (seq, env) slot rather than silently producing
    NaNs, and both docstrings flag "synthetic dummy human at slot 0" as
    the original paper's own resolution, deliberately left unimplemented
    at that layer. This class implements it here, once, at the boundary
    where real ``ObservationEncoder`` output enters the network: any
    environment currently seeing zero neighbors gets slot 0 forced
    visible before either attention layer runs (see
    ``_substitute_dummy_human`` for the caveat this introduces). The
    obstacle branch has no equivalent case -- ``RangeImageEncoder``
    always emits a fixed, geometrically meaningful set of tokens (every
    angular sector is either free or blocked), so there is no "zero
    obstacles visible" degenerate case to special-case.

Recurrent-state convention:
    Matches ``RecurrentNodeUpdate``'s own convention (see its docstring):
    the hidden state is threaded explicitly by the caller, never stored
    on this module. ``forward()`` takes and returns it. This keeps
    ``CrowdNavPPPolicy`` a plain, RL-framework-agnostic ``nn.Module`` --
    an SB3/RLlib/TorchRL adapter can wrap it however that framework wants
    its recurrent state carried, without this class knowing any of them
    exist (project's RL-framework-agnostic principle).

====================================================================
Obstacle branch (replaces the previous 1D-CNN-over-ray-features design)
====================================================================

Old design (removed, not merely modified):
    Laser rays -> 1D CNN (``ObstacleEncoder``) -> pooled/per-ray obstacle
    embedding -> concatenated into ``RobotHumanAttention`` as extra
    key/value tokens alongside humans. ``RobotHumanAttention`` no longer
    accepts any obstacle-related argument at all (see its own module
    docstring) -- humans and obstacles must never share one attention
    module again, since nothing about a human's identity/visibility/
    motion history has anything in common with a ray-cast occupancy
    sector's, and the previous sharing was an artifact of reusing
    whatever attention module already existed, not a deliberate choice.

New design, top to bottom:
    1. ``navcore.entities.components.sensors.range_image.
       RangeImageBuilder`` (sensor layer, not this file) converts one
       tick's ``ObstacleScan`` into a binary, robot-centric range image
       ``[1, H, W]``.
    2. ``RangeImageEncoder`` (residual CNN, circular padding along the
       angular/width axis, strided -- never pooled -- spatial
       compression) turns that image into a fixed set of obstacle
       tokens, each carrying a learned angular positional embedding:
       ``[nenv, num_obstacle_tokens, obstacle_embedding_dim]`` (128-d
       by default).
    3. ``RobotObstacleAttention`` is a *completely independent* module
       from ``RobotHumanAttention``: it projects the robot's 256-d
       embedding down into the obstacle branch's own 128-d latent space
       (``query_proj``) before using it as the attention query over the
       obstacle tokens, producing a 128-d obstacle context vector. This
       projection-before-query step, and keeping the whole branch at
       128-d until after attention, is deliberate: the obstacle branch
       should be free to develop its own representation without being
       constrained to live in the human branch's 256-d space from the
       start (this is why the up-projection to 256-d happens strictly
       *after* ``RobotObstacleAttention``, not before it).
    4. Both branches are independently normalized before fusion:
       ``LayerNorm(human_context)`` and
       ``LayerNorm(Linear(obstacle_context))`` (128 -> 256, then
       LayerNorm).
    5. ``ContextFusionGate`` computes a feature-wise (not scalar) sigmoid
       gate from the concatenation of both normalized contexts and
       blends them: ``fused = gate * human + (1 - gate) * obstacle``.
       ``fused`` (not the raw human ``crowd_context``) is what
       ``RecurrentNodeUpdate`` now receives as its ``crowd_context``
       input -- the GRU itself is unchanged; only what feeds it changed.

Not yet resolved here (flagged, not silently decided):
    - This class is single-tick-only (``seq_len`` is always 1 inside
      ``forward()``), mirroring ``RecurrentNodeUpdate``'s own
      inference-only scope -- see that module's docstring for why the
      batched multi-timestep training path is deferred.
    - The legacy 1D-CNN ``ObstacleEncoder``
      (``navcore.policies.crowdnav_pp.obstacle_encoder``) is no longer
      wired into this policy at all. The module file itself is left in
      place (some standalone scripts/tests may still import it) but is
      not part of the default architecture; see its own module docstring.
    - Callers that used to pass ``ray_features``/``use_obstacle_encoder``
      (``CrowdNavPPTrainer``, ``evaluate.py``, the live-demo/smoke-test
      scripts) still need to be migrated to ``range_image``/
      ``use_range_image_obstacles`` -- see this change's migration notes.
      This is a mechanical rename plus threading a
      ``RangeImageBuilder``-produced tensor through instead of
      ``scan_to_features``'s output; not done in this pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
from navcore.policies.crowdnav_pp.fusion_gate import ContextFusionGate, FusionGateConfig
from navcore.policies.crowdnav_pp.human_human_attention import (
    HumanHumanAttention,
    HumanHumanAttentionConfig,
)
from navcore.policies.crowdnav_pp.range_image_encoder import (
    RangeImageEncoder,
    RangeImageEncoderConfig,
)
from navcore.policies.crowdnav_pp.recurrent_node_update import (
    RecurrentNodeUpdate,
    RecurrentNodeUpdateConfig,
)
from navcore.policies.crowdnav_pp.robot_human_attention import (
    RobotHumanAttention,
    RobotHumanAttentionConfig,
)
from navcore.policies.crowdnav_pp.robot_obstacle_attention import (
    RobotObstacleAttention,
    RobotObstacleAttentionConfig,
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

    Every cross-module width that both the human and obstacle branches
    must agree on is derived from ``interaction_embedding_dim`` (see
    module docstring's "embedding-width unification"); the obstacle
    branch's own nested configs additionally cross-check themselves
    against it and against each other in ``__post_init__`` below, so
    there is no way for two sub-modules to end up configured with
    mismatched widths.

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
            robot's own encoding, the robot-human attention output, the
            (projected) robot-obstacle attention output, and
            RecurrentNodeUpdate's input.
        human_human_embedding_size: HumanHumanAttention's internal
            embedding width (512 in the original paper).
        human_human_num_heads: HumanHumanAttention's attention head
            count. Must evenly divide human_human_embedding_size.
        robot_human_num_heads: RobotHumanAttention's attention head
            count. Must evenly divide interaction_embedding_dim.
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
        use_gst_prediction: Whether to augment human-human attention
            with a pretrained GSTPredictor's future-displacement
            prediction -- unrelated to the obstacle branch.
        gst_pred_length: See GSTPredictorConfig.pred_length.
        use_range_image_obstacles: Whether the obstacle branch (range
            image -> RangeImageEncoder -> RobotObstacleAttention ->
            ContextFusionGate) is active. When False, ``forward()``
            feeds the GRU the human branch's context directly, with no
            obstacle awareness at all -- matching an obstacle-free
            training config (e.g. ``include_static_obstacles=False``).
        range_image_encoder: Nested config for ``RangeImageEncoder``.
            Its ``token_embedding_dim`` must equal
            ``robot_obstacle_attention.obstacle_embedding_dim``.
        robot_obstacle_attention: Nested config for
            ``RobotObstacleAttention``. Its ``robot_embedding_dim`` must
            equal ``interaction_embedding_dim``.
        fusion_gate: Nested config for ``ContextFusionGate``. Its
            ``embedding_dim`` must equal ``interaction_embedding_dim``.
    """

    robot_feature_dim: int = 8
    neighbor_feature_dim: int = 5
    temporal_hidden_size: int = 32
    interaction_embedding_dim: int = 256
    human_human_embedding_size: int = 512
    human_human_num_heads: int = 8
    robot_human_num_heads: int = 8
    node_embedding_size: int = 64
    rnn_hidden_size: int = 128
    node_output_size: int = 256
    actor_critic_hidden_size: int = 256
    action_dim: int = 2
    use_gst_prediction: bool = False
    gst_pred_length: int = 5

    use_range_image_obstacles: bool = False
    range_image_encoder: RangeImageEncoderConfig = field(
        default_factory=RangeImageEncoderConfig
    )
    robot_obstacle_attention: RobotObstacleAttentionConfig = field(
        default_factory=RobotObstacleAttentionConfig
    )
    fusion_gate: FusionGateConfig = field(default_factory=FusionGateConfig)

    def __post_init__(self) -> None:
        if not self.use_range_image_obstacles:
            return
        if self.robot_obstacle_attention.robot_embedding_dim != (
            self.interaction_embedding_dim
        ):
            raise ValueError(
                "robot_obstacle_attention.robot_embedding_dim "
                f"({self.robot_obstacle_attention.robot_embedding_dim}) must "
                f"equal interaction_embedding_dim "
                f"({self.interaction_embedding_dim})."
            )
        if self.fusion_gate.embedding_dim != self.interaction_embedding_dim:
            raise ValueError(
                f"fusion_gate.embedding_dim ({self.fusion_gate.embedding_dim}) "
                f"must equal interaction_embedding_dim "
                f"({self.interaction_embedding_dim})."
            )
        if (
            self.range_image_encoder.token_embedding_dim
            != self.robot_obstacle_attention.obstacle_embedding_dim
        ):
            raise ValueError(
                "range_image_encoder.token_embedding_dim "
                f"({self.range_image_encoder.token_embedding_dim}) must equal "
                "robot_obstacle_attention.obstacle_embedding_dim "
                f"({self.robot_obstacle_attention.obstacle_embedding_dim})."
            )

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
    composition decisions this class resolves, and for the obstacle
    branch's full design.

    Attributes:
        config: This policy's hyperparameters.
        gst_predictor: A pretrained GSTPredictor for making future
            trajectory predictions (human branch only).
        range_image_encoder: The obstacle branch's CNN, present only
            when ``config.use_range_image_obstacles`` is True.
        robot_obstacle_attention: The obstacle branch's attention
            module, present only when
            ``config.use_range_image_obstacles`` is True.
        fusion_gate: Combines the human and obstacle contexts, present
            only when ``config.use_range_image_obstacles`` is True.
    """

    def __init__(
        self,
        config: CrowdNavPPPolicyConfig,
        gst_predictor: GSTPredictor | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        if config.use_gst_prediction and gst_predictor is None:
            raise ValueError(
                "use_gst_prediction=True requires a pretrained gst_predictor "
                "(see GSTPredictorTrainer.load_predictor) -- GST is trained "
                "separately, never jointly with this policy."
            )
        self.gst_predictor = gst_predictor

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
                num_attention_heads=config.robot_human_num_heads,
            )
        )
        self.human_context_norm = nn.LayerNorm(config.interaction_embedding_dim)

        # -- obstacle branch: independent of everything above except the
        # shared robot embedding -- see module docstring.
        self.range_image_encoder: RangeImageEncoder | None = None
        self.robot_obstacle_attention: RobotObstacleAttention | None = None
        self.obstacle_context_proj: nn.Module | None = None
        self.obstacle_context_norm: nn.Module | None = None
        self.fusion_gate: ContextFusionGate | None = None
        if config.use_range_image_obstacles:
            self.range_image_encoder = RangeImageEncoder(config.range_image_encoder)
            self.robot_obstacle_attention = RobotObstacleAttention(
                config.robot_obstacle_attention
            )
            # Projection happens strictly AFTER attention (128 -> 256),
            # never before -- see module docstring's "own internal
            # latent space" rationale.
            self.obstacle_context_proj = nn.Linear(
                config.robot_obstacle_attention.obstacle_embedding_dim,
                config.interaction_embedding_dim,
            )
            self.obstacle_context_norm = nn.LayerNorm(config.interaction_embedding_dim)
            self.fusion_gate = ContextFusionGate(config.fusion_gate)

        self.recurrent_update = RecurrentNodeUpdate(
            RecurrentNodeUpdateConfig(
                input_dim=config.interaction_embedding_dim,
                node_embedding_size=config.node_embedding_size,
                rnn_hidden_size=config.rnn_hidden_size,
                output_size=config.node_output_size,
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
        range_image: Tensor | None = None,
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
            neighbor_history_mask: ``[nenv, history_steps, max_neighbors]``
                -- ``"neighbor_history_mask"``, stacked.
            hidden_state: ``[nenv, rnn_hidden_size]`` -- previous tick's
                recurrent state (see ``initial_hidden_state`` for episode
                start).
            not_done_mask: ``[nenv]`` -- ``1.0`` to carry ``hidden_state``
                forward, ``0.0`` to reset it (this tick starts a new
                episode for that environment). Forwarded verbatim to
                ``RecurrentNodeUpdate``.
            range_image: ``[nenv, 1, H, W]`` -- ``"range_image"``, as
                produced by
                ``navcore.entities.components.sensors.range_image.
                RangeImageBuilder`` and stacked across environments.
                Required exactly when ``config.use_range_image_obstacles``
                is True; must be omitted otherwise.

        Returns:
            ``(distribution, value, new_hidden_state)``: ``distribution``
            is this tick's action distribution (``Independent(Normal(...),
            1)``, from ``DiagGaussianHead``); ``value`` is ``[nenv, 1]``;
            ``new_hidden_state`` is ``[nenv, rnn_hidden_size]``, to be
            passed back in next tick.

        Raises:
            ValueError: If ``range_image`` is given but the obstacle
                branch is disabled, or required but missing.
        """
        if self.config.use_range_image_obstacles and range_image is None:
            raise ValueError(
                "config.use_range_image_obstacles=True but forward() was "
                "called without range_image."
            )
        if not self.config.use_range_image_obstacles and range_image is not None:
            raise ValueError(
                "range_image was given but config.use_range_image_obstacles=False."
            )

        neighbor_mask = neighbor_mask.bool()
        neighbor_history_mask = neighbor_history_mask.bool()

        motion_history = neighbor_history[..., _NEIGHBOR_MOTION_SLICE].transpose(0, 1)
        history_mask = neighbor_history_mask.transpose(0, 1).to(neighbor_history.dtype)
        temporal_embedding = self.temporal_encoder(motion_history, history_mask)

        if self.config.use_gst_prediction:
            assert self.gst_predictor is not None
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

        robot_embedding_seq = (
            self.robot_encoder(robot_features).unsqueeze(0).unsqueeze(-2)
        )  # [1, nenv, 1, D]
        crowd_context = self.robot_human_attention(
            robot_embedding_seq, human_embeddings, visible_mask
        ).squeeze(0)  # [nenv, D]
        robot_embedding = robot_embedding_seq.squeeze(0).squeeze(-2)  # [nenv, D]

        human_context = self.human_context_norm(crowd_context)  # [nenv, D]

        if self.config.use_range_image_obstacles:
            assert (
                self.range_image_encoder is not None
                and self.robot_obstacle_attention is not None
                and self.obstacle_context_proj is not None
                and self.obstacle_context_norm is not None
                and self.fusion_gate is not None
                and range_image is not None
            )
            obstacle_tokens = self.range_image_encoder(range_image)
            obstacle_context_raw = self.robot_obstacle_attention(
                robot_embedding, obstacle_tokens
            )  # [nenv, obstacle_embedding_dim]
            obstacle_context = self.obstacle_context_norm(
                self.obstacle_context_proj(obstacle_context_raw)
            )  # [nenv, D]
            fused_context = self.fusion_gate(human_context, obstacle_context)
        else:
            fused_context = human_context

        node_output, new_hidden_state = self.recurrent_update(
            robot_embedding,
            fused_context,
            hidden_state,
            not_done_mask,
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
        range_image: Tensor | None = None,
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
            range_image=range_image,
        )
        action, log_prob = select_action(distribution, deterministic=deterministic)
        return action, log_prob, value, new_hidden_state

    @staticmethod
    def _substitute_dummy_human(visible_mask: Tensor) -> Tensor:
        """Force slot 0 visible wherever an environment sees zero neighbors.

        See module docstring's "zero visible humans" design decision --
        this is the one place navcore implements the original paper's own
        documented workaround for a fully-masked attention row, rather
        than letting either attention module crash on it. Obstacle-branch
        equivalent: none needed (see module docstring).

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

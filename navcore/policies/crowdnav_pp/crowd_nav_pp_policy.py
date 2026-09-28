"""CrowdNavPPPolicyConfig / CrowdNavPPPolicy: the assembled CrowdNav++ actor-critic.

Wires together the seven modules ported from
``Shuijing725/CrowdNav_Prediction_AttnGraph`` (see each sibling module's own
docstring for its individual porting rationale) into the single recurrent
policy the paper describes: per-tick robot + neighbor features in, an
action distribution + value estimate + updated recurrent hidden state out.
This is the file that finally resolves the composition questions each of
those modules deliberately left open for "whoever builds this piece" --
see the design-decision notes below.

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

Design decision -- how static obstacles reach the network (``ObstacleMode``):
    Three mutually exclusive modes, selected by
    ``CrowdNavPPPolicyConfig.obstacle_mode``:

    - ``NONE``: obstacles are invisible to the policy.
    - ``ENCODER``: ray features go through ``ObstacleEncoder`` (1D CNN,
      one embedding per ray) and enter ``RobotHumanAttention`` as extra
      key/value tokens. Kept as the ablation baseline.
    - ``POINT_TOKENS``: every ray that hit something becomes a
      pseudo-human token (rel_px, rel_py, 0, 0, hit_radius) plus an
      explicit ``is_obstacle`` flag, is concatenated onto the real human
      tokens, and flows through ``HumanHumanAttention`` and
      ``RobotHumanAttention`` exactly like a pedestrian (see
      ``ObstacleTokenizer``). Obstacle tokens get a zero temporal
      embedding (ray index has no persistent identity, so a per-ray
      history would be meaningless) and zero GST prediction (a static
      point's predicted displacement is genuinely zero).

    Known caveat of POINT_TOKENS: pedestrians ignore obstacles in this
    simulator (their planner is built without obstacles), so the
    obstacle->human direction of human-human attention carries no signal
    about pedestrian motion. The useful consumer is the robot. Making
    obstacle tokens key/value-only in human-human attention is the
    natural, cheaper follow-up ablation; it is not implemented here.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn
from torch.distributions import Independent

from navcore.entities.components.sensors.obstacle_detector import RAY_FEATURE_DIM
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
from navcore.policies.crowdnav_pp.obstacle_encoder import (
    ObstacleEncoder,
    ObstacleEncoderConfig,
)
from navcore.policies.crowdnav_pp.obstacle_tokenizer import (
    ObstacleTokenizer,
    ObstacleTokenizerConfig,
)
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


class ObstacleMode(Enum):
    """How static obstacles (ray-cast hits) reach the policy."""

    NONE = "none"
    ENCODER = "encoder"
    POINT_TOKENS = "point_tokens"


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
        obstacle_mode: How ray-cast obstacle hits reach the network
            (see module docstring's ``ObstacleMode`` note).
        obstacle_max_range: The ``ObstacleDetectorConfig.max_range`` the
            ray scans are cast with. Only read in ``POINT_TOKENS`` mode,
            to convert normalized hit offsets back to meters. MUST match
            the env's ``obstacle_max_range`` -- a mismatch silently
            rescales every obstacle position.
        obstacle_hit_radius: Radius given to every obstacle token, in
            meters. Only read in ``POINT_TOKENS`` mode.
    """

    robot_feature_dim: int = 8
    neighbor_feature_dim: int = 5
    temporal_hidden_size: int = 32
    interaction_embedding_dim: int = 256
    human_human_embedding_size: int = 512
    human_human_num_heads: int = 8
    robot_human_num_heads: int = 8
    robot_human_attention_size: int = 64
    node_embedding_size: int = 64
    rnn_hidden_size: int = 128
    node_output_size: int = 256
    actor_critic_hidden_size: int = 256
    action_dim: int = 2
    use_gst_prediction: bool = False
    gst_pred_length: int = 5
    obstacle_mode: ObstacleMode = ObstacleMode.NONE
    obstacle_max_range: float = 5.0
    obstacle_hit_radius: float = 0.1

    @property
    def uses_ray_features(self) -> bool:
        """Whether ``forward()`` needs ``ray_features`` in this mode."""
        return self.obstacle_mode is not ObstacleMode.NONE

    @property
    def spatial_edge_feature_dim(self) -> int:
        base = self.neighbor_feature_dim + self.temporal_hidden_size
        if self.use_gst_prediction:
            base += self.gst_pred_length * 2
        if self.obstacle_mode is ObstacleMode.POINT_TOKENS:
            base += 1  # explicit is_obstacle type flag
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
        obstacle_encoder: ObstacleEncoder, present only in ``ObstacleMode.ENCODER``.
        obstacle_tokenizer: Ray-hit tokenizer, present only in ``ObstacleMode.POINT_TOKENS``.
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
        self.gst_predictor = gst_predictor

        # Injected encoder wins (matches the gst_predictor pattern: built
        # once, possibly pretrained, injected read-only). Only build a
        # default when ENCODER mode actually needs one and none was given.
        # An injected encoder in any other mode would register parameters
        # that are never used, so it is rejected rather than ignored.
        self.obstacle_encoder = obstacle_encoder
        if config.obstacle_mode is ObstacleMode.ENCODER:
            if self.obstacle_encoder is None:
                self.obstacle_encoder = ObstacleEncoder(
                    ObstacleEncoderConfig(
                        ray_feature_dim=RAY_FEATURE_DIM,
                        embedding_dim=config.interaction_embedding_dim,
                    )
                )
        elif obstacle_encoder is not None:
            raise ValueError(
                f"An obstacle_encoder was injected but obstacle_mode="
                f"{config.obstacle_mode.name}; only ObstacleMode.ENCODER "
                f"uses one."
            )

        self.obstacle_tokenizer: ObstacleTokenizer | None = (
            ObstacleTokenizer(
                ObstacleTokenizerConfig(
                    max_range=config.obstacle_max_range,
                    hit_radius=config.obstacle_hit_radius,
                )
            )
            if config.obstacle_mode is ObstacleMode.POINT_TOKENS
            else None
        )

        # Down-project only if the encoder's output width doesn't already
        # match interaction_embedding_dim -- the default-constructed
        # encoder above is already built at that width, so this is a
        # no-op (Identity) unless a caller injects a mismatched one.
        self._obstacle_embed_down: nn.Module = nn.Identity()
        if self.obstacle_encoder is not None:
            obstacle_output_dim = self.obstacle_encoder.config.output_dim
            if obstacle_output_dim != config.interaction_embedding_dim:
                self._obstacle_embed_down = nn.Linear(
                    obstacle_output_dim, config.interaction_embedding_dim
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
                num_attention_heads=config.robot_human_num_heads,
            )
        )
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
            ray_features: ``[nenv, num_rays, RAY_FEATURE_DIM]`` --
                ``"ray_features"``, stacked. Required iff
                ``config.obstacle_mode`` is not ``NONE``.

        Returns:
            ``(distribution, value, new_hidden_state)``: ``distribution``
            is this tick's action distribution (``Independent(Normal(...),
            1)``, from ``DiagGaussianHead``); ``value`` is ``[nenv, 1]``;
            ``new_hidden_state`` is ``[nenv, rnn_hidden_size]``, to be
            passed back in next tick.

        Raises:
            ValueError: If ``ray_features`` presence disagrees with
                ``config.obstacle_mode``.
        """
        mode = self.config.obstacle_mode
        if self.config.uses_ray_features and ray_features is None:
            raise ValueError(
                f"obstacle_mode={mode.name} but forward() was called "
                f"without ray_features."
            )
        if not self.config.uses_ray_features and ray_features is not None:
            raise ValueError("ray_features was given but obstacle_mode=NONE.")

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

        # Token set fed to human-human attention. In POINT_TOKENS mode it
        # is [real humans | obstacle hits]; humans stay first so that
        # _substitute_dummy_human's slot 0 is always a human slot.
        token_features = spatial_edge_features
        token_mask = neighbor_mask
        if mode is ObstacleMode.POINT_TOKENS:
            assert self.obstacle_tokenizer is not None and ray_features is not None
            human_flag = spatial_edge_features.new_zeros(
                *spatial_edge_features.shape[:-1], 1
            )
            obstacle_geometry, obstacle_hit_mask = self.obstacle_tokenizer.tokenize(
                ray_features
            )
            token_features = torch.cat(
                (
                    torch.cat((spatial_edge_features, human_flag), dim=-1),
                    self._pad_obstacle_tokens(obstacle_geometry),
                ),
                dim=1,
            )
            token_mask = torch.cat((neighbor_mask, obstacle_hit_mask), dim=1)

        # HumanHumanAttention/RobotHumanAttention carry an explicit
        # seq_len axis (see their own docstrings); this policy is
        # single-tick-only (see module docstring), so seq_len is always
        # exactly 1 here -- added before those two calls, squeezed back
        # off immediately after.
        human_features = token_features.unsqueeze(0)
        visible_mask = token_mask.unsqueeze(0)
        visible_mask = self._substitute_dummy_human(visible_mask)

        human_embeddings = self.human_human_attention(human_features, visible_mask)
        human_embeddings = self._human_embed_down(human_embeddings)

        obstacle_embedding_for_attention = None
        obstacle_mask_for_attention = None
        if mode is ObstacleMode.ENCODER:
            assert self.obstacle_encoder is not None and ray_features is not None
            # [nenv, num_rays, output_dim] -- one token per ray, not a
            # single pooled summary (see ObstacleEncoder module docstring:
            # pooling was confirmed via probe to destroy left/right
            # directional info even untrained).
            raw_obstacle_embedding = self.obstacle_encoder(ray_features)
            obstacle_context = self._obstacle_embed_down(raw_obstacle_embedding)
            obstacle_embedding_for_attention = obstacle_context.unsqueeze(0)
            # ray_features[..., 0] is scan_to_features' hit_mask channel --
            # mask out rays that hit nothing rather than feeding them in as
            # phantom always-visible obstacle tokens.
            obstacle_mask_for_attention = (ray_features[..., 0] > 0.5).unsqueeze(0)

        robot_embedding = self.robot_encoder(robot_features).unsqueeze(0).unsqueeze(-2)
        crowd_context = self.robot_human_attention(
            robot_embedding,
            human_embeddings,
            visible_mask,
            obstacle_embedding=obstacle_embedding_for_attention,
            obstacle_mask=obstacle_mask_for_attention,
        ).squeeze(0)
        robot_embedding = robot_embedding.squeeze(0).squeeze(-2)

        node_output, new_hidden_state = self.recurrent_update(
            robot_embedding,
            crowd_context,
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

    def _pad_obstacle_tokens(self, obstacle_geometry: Tensor) -> Tensor:
        """Widen geometric obstacle tokens to ``spatial_edge_feature_dim``.

        Layout matches a human token: ``[geometry(5) | temporal embedding |
        GST prediction (if enabled) | is_obstacle flag]``. The temporal and
        GST slots are zero -- ray index has no persistent identity, so no
        motion history exists, and a static point's predicted displacement
        is zero. The flag is the only reliable human/obstacle
        discriminator: hit radius (0.3) equals a nominal pedestrian radius.

        Args:
            obstacle_geometry: ``[nenv, num_rays, 5]`` from
                ``ObstacleTokenizer.tokenize``.

        Returns:
            ``[nenv, num_rays, spatial_edge_feature_dim]``.
        """
        nenv, num_rays, _ = obstacle_geometry.shape
        padding_dim = self.config.temporal_hidden_size
        if self.config.use_gst_prediction:
            padding_dim += self.config.gst_pred_length * 2
        padding = obstacle_geometry.new_zeros(nenv, num_rays, padding_dim)
        flag = obstacle_geometry.new_ones(nenv, num_rays, 1)
        return torch.cat((obstacle_geometry, padding, flag), dim=-1)

    @staticmethod
    def _substitute_dummy_human(visible_mask: Tensor) -> Tensor:
        """Force slot 0 visible wherever an environment sees zero tokens.

        See module docstring's "zero visible humans" design decision --
        this is the one place navcore implements the original paper's own
        documented workaround for a fully-masked attention row, rather
        than letting either attention module crash on it. In
        ``POINT_TOKENS`` mode the mask covers humans *and* obstacle hits,
        so the substitution only fires when the robot sees neither.

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
            visible_mask: ``[1, nenv, num_tokens]``, boolean.

        Returns:
            ``visible_mask``, with slot 0 marked visible for any
            ``(seq, env)`` that had no real visible token. Returned
            unchanged (same object) if every slot already has at least
            one visible token.
        """
        no_humans_visible = ~visible_mask.any(dim=-1)  # [1, nenv]
        if not bool(no_humans_visible.any()):
            return visible_mask

        visible_mask = visible_mask.clone()
        visible_mask[..., 0] |= no_humans_visible
        return visible_mask

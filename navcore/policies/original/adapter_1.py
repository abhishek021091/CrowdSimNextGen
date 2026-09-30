"""CrowdSimNextGen adapter for the official CrowdNav++ policy.

Architecture (unchanged from the design goal, but the original forward is now
*reused*, not copied):

    navcore observation
        -> ``_build_inputs``            (observation conversion only)
        -> official ``base.forward``    (SpatialEdgeSelfAttn, EdgeAttention_M,
                                         robot_linear, EndRNN, actor, critic --
                                         100% untouched code)
              \\-- forward-pre-hook on ``base.humanNodeRNN`` (EndRNN):
                   fused = obstacle_fusion([crowd_context || obstacle_context])
        -> ``dist``                     (official DiagGaussian)

Obstacle rays never reach ``spatial_attn``, ``attn``, ``detected_human_num``,
the human masks or human ordering: they are encoded in ``_encode_obstacles``
from ``ray_features`` alone and only meet the human pipeline's *output*
(the 256-d crowd context) inside the hook.

state_dict layout: ``base.*`` and ``dist.*`` are exactly the official keys;
the extension adds only ``obstacle_encoder.*`` / ``obstacle_fusion.*``.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor, nn

from navcore.entities.components.sensors.obstacle_detector import RAY_FEATURE_DIM
from navcore.policies.obstacle_mode import ObstacleMode

from .obstacle_tokenizer import ObstacleTokenizer, ObstacleTokenizerConfig
from .policy import Policy

__all__ = [
    "CrowdNavPPPolicy",
    "CrowdNavPPPolicyConfig",
    "ObstacleMode",
    "ObstacleTokenEncoder",
    "build_action_space",
    "build_base_args",
    "build_obs_space",
]

# -- constants dictated by the official CrowdSimPredRealGST-v0 layout ---------
#: Predicted future steps per human (12-d spatial edge = 2 * (1 + 5)).
_PRED_STEPS = 5
_SPATIAL_EDGE_DIM = 2 * (1 + _PRED_STEPS)
_ROBOT_NODE_DIM = 7
_TEMPORAL_EDGE_DIM = 2
#: Width of robot_linear's output == EndRNN's encoder_linear input.
_CONTEXT_DIM = 256
_EDGE_RNN_SIZE = 256
#: Official env replaces inf padding of spatial edges by 15 (as far as I
#: remember crowd_sim_var_num.py -- verify). Only visible for the dummy human.
_FAR_AWAY = 15.0
#: Official GST prediction interval; also used by the constant-velocity fallback.
_CV_DT = 0.25
_OBSTACLE_PREFIXES = ("obstacle_encoder.", "obstacle_fusion.")


@dataclass(slots=True, frozen=True)
class CrowdNavPPPolicyConfig:
    """Hyperparameters for CrowdNavPPPolicy.

    Fields marked *unused* are kept for public-API compatibility with the
    from-scratch policy; the official backbone hard-codes those values.
    """

    robot_feature_dim: int = 8
    neighbor_feature_dim: int = 5
    max_neighbors: int = 10
    temporal_hidden_size: int = 32  # unused
    interaction_embedding_dim: int = 256  # unused
    human_human_embedding_size: int = 512  # unused (official: 512)
    human_human_num_heads: int = 8  # unused (official: 8)
    robot_human_num_heads: int = 8  # unused
    robot_human_attention_size: int = 64
    node_embedding_size: int = 64
    rnn_hidden_size: int = 128
    node_output_size: int = 256
    actor_critic_hidden_size: int = 256  # unused (official: == node_output_size)
    action_dim: int = 2
    use_gst_prediction: bool = False
    gst_pred_length: int = _PRED_STEPS
    obstacle_mode: ObstacleMode = ObstacleMode.NONE
    obstacle_max_range: float = 5.0
    obstacle_hit_radius: float = 0.1  # tokenizer only; 12-d tokens carry no radius
    obstacle_context_dim: int = 128
    use_obstacle_encoder: bool | None = None

    def __post_init__(self) -> None:
        if self.use_obstacle_encoder is not None:
            if self.use_obstacle_encoder and self.obstacle_mode is ObstacleMode.NONE:
                object.__setattr__(self, "obstacle_mode", ObstacleMode.ENCODER)
            elif (
                not self.use_obstacle_encoder
                and self.obstacle_mode is ObstacleMode.ENCODER
            ):
                object.__setattr__(self, "obstacle_mode", ObstacleMode.NONE)
        else:
            object.__setattr__(
                self, "use_obstacle_encoder", self.obstacle_mode is ObstacleMode.ENCODER
            )
        if self.use_gst_prediction and self.gst_pred_length != _PRED_STEPS:
            raise ValueError(
                f"The official 12-d spatial edge requires gst_pred_length="
                f"{_PRED_STEPS}, got {self.gst_pred_length}."
            )

    @property
    def uses_ray_features(self) -> bool:
        return self.obstacle_mode is not ObstacleMode.NONE

    @property
    def spatial_edge_feature_dim(self) -> int:
        return _SPATIAL_EDGE_DIM


def build_base_args(config: CrowdNavPPPolicyConfig) -> types.SimpleNamespace:
    """The ``args`` namespace the official ``selfAttn_merge_SRNN`` reads.

    ``no_cuda=True`` on purpose: the official ``__init__`` unconditionally
    builds ``dummy_human_mask`` with ``.cuda()`` when ``no_cuda`` is False.
    With ``sort_humans=True`` that tensor is never read, so this avoids a crash
    on CPU-only hosts and a stray CUDA tensor that ``.to(device)`` never moves.
    """
    return types.SimpleNamespace(
        human_node_rnn_size=config.rnn_hidden_size,
        human_human_edge_rnn_size=_EDGE_RNN_SIZE,
        human_node_output_size=config.node_output_size,
        human_node_input_size=3,
        human_human_edge_input_size=2,
        human_node_embedding_size=config.node_embedding_size,
        human_human_edge_embedding_size=64,
        attention_size=config.robot_human_attention_size,
        seq_length=30,
        num_processes=1,
        num_mini_batch=1,
        use_self_attn=True,
        use_hr_attn=True,
        env_name="CrowdSimPredRealGST-v0",
        sort_humans=True,
        no_cuda=True,
    )


def build_obs_space(config: CrowdNavPPPolicyConfig) -> dict[str, spaces.Box]:
    inf = np.inf
    return {
        "robot_node": spaces.Box(-inf, inf, shape=(1, _ROBOT_NODE_DIM)),
        "temporal_edges": spaces.Box(-inf, inf, shape=(1, _TEMPORAL_EDGE_DIM)),
        "spatial_edges": spaces.Box(
            -inf, inf, shape=(config.max_neighbors, _SPATIAL_EDGE_DIM)
        ),
        "visible_masks": spaces.Box(-inf, inf, shape=(config.max_neighbors,)),
        "detected_human_num": spaces.Box(-inf, inf, shape=(1,)),
    }


def build_action_space(config: CrowdNavPPPolicyConfig) -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(config.action_dim,), dtype=np.float32)


def masked_mean(features: Tensor, mask: Tensor) -> Tensor:
    """Mean over dim 1 of ``[B, R, D]`` features where ``mask [B, R]`` is True.

    All-False rows give an exact zero vector (numerator is zero, denominator is
    clamped to 1), so no separate ``has_hits`` gate is needed.
    """
    m = mask.unsqueeze(-1).to(features.dtype)
    return (features * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class ObstacleTokenEncoder(nn.Module):
    """Per-token MLP + permutation-invariant masked mean over ray hits."""

    def __init__(
        self,
        token_dim: int = _SPATIAL_EDGE_DIM,
        hidden_dim: int = 64,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.output_dim = output_dim

    def forward(self, obstacle_tokens: Tensor, hit_mask: Tensor) -> Tensor:
        """``[B, R, token_dim]`` + ``[B, R]`` bool -> ``[B, output_dim]``."""
        return masked_mean(self.mlp(obstacle_tokens), hit_mask)


class _SpatialAttentionView:
    """Read-only view exposing ``.embed`` for trainer/test metrics."""

    def __init__(self, spatial_attn: nn.Module) -> None:
        self.embed = spatial_attn.embedding_layer


class _ActionHeadView:
    """Read-only view exposing ``log_std`` of the official DiagGaussian.

    NOTE: the official head does not clamp log_std; LOG_STD_MIN/MAX are only
    what the trainer's *metric* clamps to, not what the policy uses.
    """

    LOG_STD_MIN = -3.0
    LOG_STD_MAX = 0.5

    def __init__(self, dist: nn.Module) -> None:
        self._dist = dist

    @property
    def log_std(self) -> Tensor:
        return self._dist.logstd._bias.reshape(-1)


class CrowdNavPPPolicy(Policy):
    """Official CrowdNav++ ``Policy`` + a decoupled obstacle branch.

    Attributes:
        config: Adapter hyperparameters.
        obstacle_tokenizer / obstacle_encoder / obstacle_fusion: Extension
            modules; ``None`` in ``ObstacleMode.NONE``.
    """

    def __init__(
        self,
        config: CrowdNavPPPolicyConfig | None = None,
        gst_predictor: nn.Module | None = None,
        obstacle_encoder: nn.Module | None = None,
    ) -> None:
        config = config or CrowdNavPPPolicyConfig()
        if config.obstacle_mode is ObstacleMode.POINT_TOKENS and obstacle_encoder:
            raise ValueError(
                "An obstacle_encoder was injected but obstacle_mode=POINT_TOKENS; "
                "only ObstacleMode.ENCODER uses an injected obstacle_encoder."
            )
        if config.use_gst_prediction and gst_predictor is None:
            raise ValueError("use_gst_prediction=True requires a gst_predictor.")

        super().__init__(
            build_obs_space(config),
            build_action_space(config),
            base="selfAttn_merge_srnn",
            base_kwargs=build_base_args(config),
        )
        self.config = config
        self.human_num = config.max_neighbors
        self.gst_predictor = gst_predictor

        # Official: human_node_final_linear only serves an auxiliary loss.
        # Kept in the state_dict (official key), excluded from optimisation.
        self.base.human_node_final_linear.requires_grad_(False)

        self.obstacle_tokenizer: ObstacleTokenizer | None = None
        self.obstacle_encoder: nn.Module | None = None
        self.obstacle_fusion: nn.Linear | None = None
        self.obstacle_dim = config.obstacle_context_dim
        self._pending_obstacle_context: Tensor | None = None

        if config.obstacle_mode is ObstacleMode.POINT_TOKENS:
            self.obstacle_tokenizer = ObstacleTokenizer(
                ObstacleTokenizerConfig(
                    max_range=config.obstacle_max_range,
                    hit_radius=config.obstacle_hit_radius,
                )
            )
            self.obstacle_encoder = ObstacleTokenEncoder(output_dim=self.obstacle_dim)
        elif config.obstacle_mode is ObstacleMode.ENCODER:
            if obstacle_encoder is None:
                from navcore.policies.crowdnav_pp.obstacle_encoder import (
                    ObstacleEncoder,
                    ObstacleEncoderConfig,
                )

                obstacle_encoder = ObstacleEncoder(
                    ObstacleEncoderConfig(
                        ray_feature_dim=RAY_FEATURE_DIM, embedding_dim=self.obstacle_dim
                    )
                )
            self.obstacle_encoder = obstacle_encoder
            self.obstacle_dim = getattr(
                obstacle_encoder.config,
                "output_dim",
                getattr(obstacle_encoder.config, "embedding_dim", self.obstacle_dim),
            )

        if self.obstacle_encoder is not None:
            self.obstacle_fusion = nn.Linear(
                _CONTEXT_DIM + self.obstacle_dim, _CONTEXT_DIM
            )
            with torch.no_grad():
                self.obstacle_fusion.bias.zero_()
                self.obstacle_fusion.weight[:, :_CONTEXT_DIM] = torch.eye(_CONTEXT_DIM)
                nn.init.orthogonal_(
                    self.obstacle_fusion.weight[:, _CONTEXT_DIM:], gain=1.0
                )
            # The single extension point into the official forward.
            self.base.humanNodeRNN.register_forward_pre_hook(
                self._inject_obstacle_context
            )

    # -- adapter views -------------------------------------------------------

    @property
    def human_human_attention(self) -> _SpatialAttentionView:
        return _SpatialAttentionView(self.base.spatial_attn)

    @property
    def action_head(self) -> _ActionHeadView:
        return _ActionHeadView(self.dist)

    def initial_hidden_state(
        self, nenv: int = 1, device: torch.device | None = None
    ) -> Tensor:
        return torch.zeros(nenv, self.config.rnn_hidden_size, device=device)

    # -- checkpoint loading --------------------------------------------------

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Strict load with ONE explicit allowance.

        An official (or pre-obstacle) checkpoint has no ``obstacle_*`` keys. That
        is tolerated only if *every* obstacle key is absent; a partially present
        obstacle branch, any other missing key, or any unexpected key raises.
        Shape mismatches always raise (torch does this even for strict=False).
        """
        if "policy_state_dict" in state_dict:
            state_dict = state_dict["policy_state_dict"]
        if strict:
            expected = set(self.state_dict())
            given = set(state_dict)
            missing = expected - given
            unexpected = given - expected
            obstacle_keys = {k for k in expected if k.startswith(_OBSTACLE_PREFIXES)}
            tolerated = obstacle_keys if missing >= obstacle_keys else set()
            fatal = missing - tolerated
            if fatal or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch. missing={sorted(fatal)[:10]} "
                    f"unexpected={sorted(unexpected)[:10]}"
                )
        return super().load_state_dict(state_dict, strict=False, **kwargs)

    # -- observation conversion ----------------------------------------------

    def _build_inputs(
        self,
        robot: Tensor,
        neighbors: Tensor,
        neighbor_mask: Tensor,
        neighbor_history: Tensor,
        neighbor_history_mask: Tensor,
    ) -> dict[str, Tensor]:
        """navcore observation -> official input dict (humans only).

        Assumes ``neighbor_mask`` is a contiguous prefix (nearest first), which
        is what ObservationEncoder produces and what the official
        ``sort_humans=True`` path (``detected_human_num`` as a prefix length)
        requires.
        """
        B, N = neighbors.shape[:2]
        device, dtype = robot.device, robot.dtype

        temporal_edges = robot[:, 2:4].unsqueeze(1)  # (vx, vy)
        zeros = torch.zeros(B, 1, device=device, dtype=dtype)
        theta = torch.atan2(robot[:, 7:8], robot[:, 6:7])
        # (px, py, radius, gx, gy, v_pref, theta). Robot-relative frame:
        # px = py = 0 and the goal is relative -- see report (deviation).
        robot_node = torch.cat(
            (
                zeros,
                zeros,
                robot[:, 5:6],
                robot[:, 0:1],
                robot[:, 1:2],
                robot[:, 4:5],
                theta,
            ),
            dim=-1,
        ).unsqueeze(1)

        rel_pos, vel = neighbors[..., 0:2], neighbors[..., 2:4]
        if self.config.use_gst_prediction:
            assert self.gst_predictor is not None
            pred = self.gst_predictor.predict_features(
                neighbor_history[..., 0:2].transpose(1, 2),
                neighbor_history[..., 2:4].transpose(1, 2),
                neighbor_history_mask.transpose(1, 2),
            ).view(B, N, _PRED_STEPS, 2)
            future = rel_pos.unsqueeze(2) + pred
        else:
            steps = torch.arange(1, _PRED_STEPS + 1, device=device, dtype=dtype)
            future = rel_pos.unsqueeze(2) + vel.unsqueeze(2) * (
                steps.view(1, 1, -1, 1) * _CV_DT
            )
        spatial_edges = torch.cat((rel_pos, future.reshape(B, N, -1)), dim=-1)

        mask = neighbor_mask.bool()
        if N < self.human_num:
            pad = self.human_num - N
            spatial_edges = torch.cat(
                (spatial_edges, spatial_edges.new_zeros(B, pad, _SPATIAL_EDGE_DIM)), 1
            )
            mask = torch.cat((mask, mask.new_zeros(B, pad)), 1)
        else:
            spatial_edges = spatial_edges[:, : self.human_num]
            mask = mask[:, : self.human_num]

        # Padding -> far away (official fills inf with 15); then the official
        # "at least one human" rule. No .any() -> no GPU sync.
        spatial_edges = torch.where(
            mask.unsqueeze(-1), spatial_edges, spatial_edges.new_full((), _FAR_AWAY)
        )
        first = mask[:, :1] | ~mask.any(dim=-1, keepdim=True)
        mask = torch.cat((first, mask[:, 1:]), dim=1)

        return {
            "robot_node": robot_node,
            "temporal_edges": temporal_edges,
            "spatial_edges": spatial_edges,
            "visible_masks": mask.to(dtype),
            "detected_human_num": mask.sum(dim=-1, keepdim=True, dtype=torch.int32),
        }

    # -- obstacle branch (never touches human tensors) -------------------------

    def _encode_obstacles(self, ray_features: Tensor) -> Tensor | None:
        """``[B, R, 3]`` rays -> ``[1, B, 1, D]`` obstacle context (seq_len=1)."""
        mode = self.config.obstacle_mode
        if mode is ObstacleMode.NONE:
            return None
        assert self.obstacle_encoder is not None
        if mode is ObstacleMode.POINT_TOKENS:
            assert self.obstacle_tokenizer is not None
            geometry, hit = self.obstacle_tokenizer.tokenize(ray_features)
            pos = geometry[..., 0:2]
            # static point: predicted future == current position, x5
            tokens = torch.cat((pos, pos.repeat(1, 1, _PRED_STEPS)), dim=-1)
            context = self.obstacle_encoder(tokens, hit)
        else:
            hit = ray_features[..., 0] > 0.5
            context = masked_mean(self.obstacle_encoder(ray_features), hit)
        return context.unsqueeze(0).unsqueeze(2)

    def _inject_obstacle_context(self, module: nn.Module, args: tuple):
        """Forward-pre-hook on EndRNN: fuse obstacle context into crowd context.

        EndRNN.forward(robot_s, h_spatial_other, h, masks); only argument 1
        (the crowd context produced by the untouched human pipeline) is replaced.
        No-op unless ``forward`` staged a context, so direct ``self.base(...)``
        calls keep exact official behaviour.
        """
        context = self._pending_obstacle_context
        if context is None:
            return None
        robot_states, crowd_context, hidden, masks = args
        assert self.obstacle_fusion is not None
        fused = self.obstacle_fusion(torch.cat((crowd_context, context), dim=-1))
        return robot_states, fused, hidden, masks

    # -- policy API ----------------------------------------------------------------

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
    ) -> tuple[object, Tensor, Tensor]:
        """One recurrent tick for ``B`` environments (delegates to official base)."""
        if self.config.uses_ray_features and ray_features is None:
            raise ValueError("ray_features required in obstacle mode.")
        if not self.config.uses_ray_features and ray_features is not None:
            raise ValueError("ray_features was given but obstacle_mode is NONE.")

        B = robot_features.size(0)
        inputs = self._build_inputs(
            robot_features,
            neighbor_features,
            neighbor_mask,
            neighbor_history,
            neighbor_history_mask,
        )
        # Only sizes of the edge-RNN state are read by the official forward.
        rnn_hxs = {
            "human_node_rnn": hidden_state.reshape(B, 1, -1),
            "human_human_edge_rnn": hidden_state.new_zeros(B, 1, _EDGE_RNN_SIZE),
        }
        # Official code fixes nenv at construction (num_processes); we batch freely.
        self.base.nenv = B
        self._pending_obstacle_context = (
            self._encode_obstacles(ray_features) if ray_features is not None else None
        )
        try:
            value, actor_features, new_rnn_hxs = self.base(
                inputs, rnn_hxs, not_done_mask.reshape(B, 1), infer=True
            )
        finally:
            self._pending_obstacle_context = None

        new_hidden = new_rnn_hxs["human_node_rnn"].reshape(B, -1)
        return self.dist(actor_features), value, new_hidden

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
        """navcore API: ``(action, log_prob[B], value, new_hidden)``.

        Deliberately different signature from the official ``Policy.act``.
        """
        distribution, value, new_hidden = self.forward(
            robot_features,
            neighbor_features,
            neighbor_mask,
            neighbor_history,
            neighbor_history_mask,
            hidden_state,
            not_done_mask,
            ray_features=ray_features,
        )
        action = distribution.mode() if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value, new_hidden

    def get_value(self, *args, **kwargs):
        raise NotImplementedError(
            "Use forward()/act(); official-format inputs would skip the obstacle branch."
        )

    def evaluate_actions(self, *args, **kwargs):
        raise NotImplementedError(
            "Use forward(); official-format inputs would skip the obstacle branch."
        )

"""CrowdSimNextGen adapter for the official CrowdNav++ policy.

Architecture::

    navcore observation
        -> ``_build_inputs``  (observation conversion only)
             real humans        [B, max_humans, 12]  \\
                                                      +-- cat --> [B, max_humans + R, 12]
             obstacle ray hits  [B, R, 12]           /   (pseudo-humans)
        -> official ``base.forward``  (100% untouched code)
        -> ``dist``  (official DiagGaussian)

Obstacles are *not* a separate branch. Every obstacle ray becomes one
pseudo-human slot, laid out exactly like a stationary human: position
relative to the robot, zero velocity, constant-velocity "future" (= the
position repeated), so it flows through human embedding, human-human
attention, robot-human attention, the GRU and the actor/critic heads
identically to a real human. The network cannot tell them apart.

Fixed slot count: a ray that hit nothing keeps its slot as a far-away,
masked dummy (the same convention already used for padded human slots),
so ``num slots == max_neighbors + obstacle_num_rays`` at every timestep.

Why ``sort_humans=False``: with ``True`` the official attention treats
validity as a *prefix length* (``detected_human_num``). Ray hits are
scattered across the ray array, so their validity is not a prefix. The
``False`` path takes an arbitrary boolean mask, has no parameters, and is
identical to the prefix path for prefix masks -- so pretrained weights are
unaffected.

Known consequences (not bugs):
    * The official ``EdgeAttention_M`` scales its logits by
      ``num_slots / sqrt(attention_size)``. More slots => sharper robot-human
      attention than pretrained weights saw. Weights load; behaviour drifts
      until fine-tuned.
    * The official 12-d edge has no radius channel, so ``obstacle_hit_radius``
      is carried in the 5-d token but dropped here, exactly like real human
      radii are.

state_dict layout: exactly the official ``base.*`` / ``dist.*`` keys in every
obstacle mode.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field

import numpy as np
import torch
from gymnasium import spaces
from torch import Tensor, nn

from navcore.policies.obstacle_mode import ObstacleMode
from navcore.entities.components.sensors.ray_spec import (
    RaySpec,
    check_ray_counts,
    default_ray_spec,
)


from .obstacle_tokenizer import ObstacleTokenizer, ObstacleTokenizerConfig
from .policy import Policy

__all__ = [
    "CrowdNavPPPolicy",
    "CrowdNavPPPolicyConfig",
    "ObstacleMode",
    "build_action_space",
    "build_base_args",
    "build_obs_space",
    "constant_velocity_edges",
]

# -- constants dictated by the official CrowdSimPredRealGST-v0 layout ---------
#: Predicted future steps per human (12-d spatial edge = 2 * (1 + 5)).
_PRED_STEPS = 5
_SPATIAL_EDGE_DIM = 2 * (1 + _PRED_STEPS)
_ROBOT_NODE_DIM = 7
_TEMPORAL_EDGE_DIM = 2
_EDGE_RNN_SIZE = 256
#: Official env replaces inf padding of spatial edges by 15. Used for every
#: invalid slot (padded human OR obstacle ray that hit nothing).
_FAR_AWAY = 15.0
#: Official GST prediction interval; also used by the constant-velocity fallback.
_CV_DT = 0.25
#: Keys written by the removed obstacle side-branch; such checkpoints are rejected.
_LEGACY_OBSTACLE_PREFIXES = ("obstacle_encoder.", "obstacle_fusion.")


@dataclass(slots=True, frozen=True)
class CrowdNavPPPolicyConfig:
    """Hyperparameters for CrowdNavPPPolicy.

    Fields marked *unused* are kept for public-API compatibility with the
    from-scratch policy; the official backbone hard-codes those values.

    Attributes:
        obstacle_num_rays: Number of obstacle pseudo-human slots. Must equal
            the env's ``obstacle_num_rays``; it fixes the obs-space shape.
            Only used when ``obstacle_mode`` is ``POINT_TOKENS``.
        obstacle_hit_radius: Radius stored in obstacle tokens (metres). Has no
            effect on the official 12-d layout (see module docstring).
        use_obstacle_encoder: Deprecated alias; ENCODER mode was removed.
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
    ray_spec: RaySpec = field(
        default_factory=default_ray_spec
    )  # replaces obstacle_max_range + obstacle_num_rays
    obstacle_hit_radius: float = 0.3
    use_obstacle_encoder: bool | None = None

    @property
    def obstacle_num_rays(self) -> int:
        return self.ray_spec.num_rays

    @property
    def obstacle_max_range(self) -> float:
        return self.ray_spec.max_range

    @property
    def num_obstacle_slots(self) -> int:
        return self.ray_spec.num_rays if self.uses_ray_features else 0

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
        if self.obstacle_mode is ObstacleMode.ENCODER:
            raise ValueError(
                "ObstacleMode.ENCODER was removed from the official-port adapter: "
                "obstacles are now pseudo-humans (ObstacleMode.POINT_TOKENS)."
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
    def total_slots(self) -> int:
        """Fixed slot count entering human-human attention."""
        return self.max_neighbors + self.num_obstacle_slots

    @property
    def spatial_edge_feature_dim(self) -> int:
        return _SPATIAL_EDGE_DIM


def build_base_args(config: CrowdNavPPPolicyConfig) -> types.SimpleNamespace:
    """The ``args`` namespace the official ``selfAttn_merge_SRNN`` reads.

    ``no_cuda=True`` on purpose: the official ``__init__`` unconditionally
    builds ``dummy_human_mask`` with ``.cuda()`` when ``no_cuda`` is False.

    ``sort_humans=False``: validity is passed as an arbitrary mask instead of a
    prefix length, which is required because obstacle-ray hits are not a
    contiguous prefix (see module docstring).
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
        sort_humans=False,
        no_cuda=True,
    )


def build_obs_space(config: CrowdNavPPPolicyConfig) -> dict[str, spaces.Box]:
    inf = np.inf
    slots = config.total_slots
    return {
        "robot_node": spaces.Box(-inf, inf, shape=(1, _ROBOT_NODE_DIM)),
        "temporal_edges": spaces.Box(-inf, inf, shape=(1, _TEMPORAL_EDGE_DIM)),
        "spatial_edges": spaces.Box(-inf, inf, shape=(slots, _SPATIAL_EDGE_DIM)),
        "visible_masks": spaces.Box(-inf, inf, shape=(slots,)),
        "detected_human_num": spaces.Box(-inf, inf, shape=(1,)),
    }


def build_action_space(config: CrowdNavPPPolicyConfig) -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(config.action_dim,), dtype=np.float32)


def _edges_from_future(rel_pos: Tensor, future: Tensor) -> Tensor:
    """``[B,N,2]`` + ``[B,N,5,2]`` -> official 12-d edge ``[B,N,12]``."""
    B, N = rel_pos.shape[:2]
    return torch.cat((rel_pos, future.reshape(B, N, -1)), dim=-1)


def constant_velocity_edges(rel_pos: Tensor, vel: Tensor) -> Tensor:
    """12-d spatial edges under a constant-velocity assumption.

    Used for humans when GST is off, and for obstacle pseudo-humans always
    (``vel == 0`` => the future equals the current position).
    """
    steps = torch.arange(1, _PRED_STEPS + 1, device=rel_pos.device, dtype=rel_pos.dtype)
    future = rel_pos.unsqueeze(2) + vel.unsqueeze(2) * (
        steps.view(1, 1, -1, 1) * _CV_DT
    )
    return _edges_from_future(rel_pos, future)


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
    """Official CrowdNav++ ``Policy`` whose human slots include obstacle hits.

    Attributes:
        config: Adapter hyperparameters.
        max_humans: Real-human slots (``config.max_neighbors``).
        num_obstacle_slots: Obstacle pseudo-human slots (0 in NONE mode).
        obstacle_tokenizer: Ray-hit tokenizer; ``None`` in ``ObstacleMode.NONE``.
    """

    def __init__(
        self,
        config: CrowdNavPPPolicyConfig | None = None,
        gst_predictor: nn.Module | None = None,
    ) -> None:
        config = config or CrowdNavPPPolicyConfig()
        if config.use_gst_prediction and gst_predictor is None:
            raise ValueError("use_gst_prediction=True requires a gst_predictor.")

        super().__init__(
            build_obs_space(config),
            build_action_space(config),
            base="selfAttn_merge_srnn",
            base_kwargs=build_base_args(config),
        )
        self.config = config
        self.max_humans = config.max_neighbors
        self.num_obstacle_slots = config.num_obstacle_slots
        self.gst_predictor = gst_predictor

        # Official: human_node_final_linear only serves an auxiliary loss.
        # Kept in the state_dict (official key), excluded from optimisation.
        self.base.human_node_final_linear.requires_grad_(False)

        self.obstacle_tokenizer: ObstacleTokenizer | None = None
        if config.obstacle_mode is ObstacleMode.POINT_TOKENS:
            self.obstacle_tokenizer = ObstacleTokenizer(
                ObstacleTokenizerConfig(
                    max_range=config.ray_spec.max_range,
                    hit_radius=config.obstacle_hit_radius,
                    num_rays=config.ray_spec.num_rays,
                )
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
        """Load official-layout weights; reject the removed obstacle side-branch."""
        if "policy_state_dict" in state_dict:
            state_dict = state_dict["policy_state_dict"]
        legacy = sorted(
            k for k in state_dict if k.startswith(_LEGACY_OBSTACLE_PREFIXES)
        )
        if legacy:
            raise RuntimeError(
                "Checkpoint contains keys from the removed obstacle side-branch "
                f"(e.g. {legacy[:3]}); it was trained with a different "
                "architecture and cannot be loaded."
            )
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    # -- observation conversion ----------------------------------------------

    def _human_slots(
        self,
        neighbors: Tensor,
        neighbor_mask: Tensor,
        neighbor_history: Tensor,
        neighbor_history_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Real humans -> ``(edges [B,max_humans,12], mask [B,max_humans] bool)``.

        Assumes nothing about mask contiguity. Pads/truncates to ``max_humans``.
        """
        B, N = neighbors.shape[:2]
        rel_pos, vel = neighbors[..., 0:2], neighbors[..., 2:4]
        if self.config.use_gst_prediction:
            assert self.gst_predictor is not None
            pred = self.gst_predictor.predict_features(
                neighbor_history[..., 0:2].transpose(1, 2),
                neighbor_history[..., 2:4].transpose(1, 2),
                neighbor_history_mask.transpose(1, 2),
            ).view(B, N, _PRED_STEPS, 2)
            edges = _edges_from_future(rel_pos, rel_pos.unsqueeze(2) + pred)
        else:
            edges = constant_velocity_edges(rel_pos, vel)

        mask = neighbor_mask.bool()
        if N < self.max_humans:
            pad = self.max_humans - N
            edges = torch.cat((edges, edges.new_zeros(B, pad, _SPATIAL_EDGE_DIM)), 1)
            mask = torch.cat((mask, mask.new_zeros(B, pad)), 1)
        else:
            edges = edges[:, : self.max_humans]
            mask = mask[:, : self.max_humans]
        return edges, mask

    def _obstacle_slots(self, ray_features: Tensor) -> tuple[Tensor, Tensor]:
        """Ray scan -> ``(edges [B,R,12], hit mask [B,R] bool)`` pseudo-humans.

        One slot per ray, in ray order. Rays that hit nothing have zero features
        and ``mask=False``; ``_build_inputs`` turns them into far-away dummies
        together with padded human slots.
        """
        assert self.obstacle_tokenizer is not None
        check_ray_counts(
            {
                "CrowdNavPPPolicy.config.ray_spec.num_rays": self.num_obstacle_slots,
                "observation['ray_features'] (env sensor)": ray_features.shape[1],
            },
            context="CrowdNavPPPolicy._obstacle_slots",
        )
        tokens, hit = self.obstacle_tokenizer.tokenize(ray_features)
        # tokens: (rel_px, rel_py, vx=0, vy=0, radius). Radius is dropped here,
        # like it is for real humans (the official edge has no radius channel).
        edges = constant_velocity_edges(tokens[..., 0:2], tokens[..., 2:4])
        return edges, hit

    def _build_inputs(
        self,
        robot: Tensor,
        neighbors: Tensor,
        neighbor_mask: Tensor,
        neighbor_history: Tensor,
        neighbor_history_mask: Tensor,
        ray_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """navcore observation -> official input dict.

        ``spatial_edges`` is ``[real humans | obstacle pseudo-humans]`` with a
        fixed ``config.total_slots`` (=``max_humans + R``) at every call.
        """
        B = neighbors.shape[0]
        device, dtype = robot.device, robot.dtype

        temporal_edges = robot[:, 2:4].unsqueeze(1)  # (vx, vy)
        zeros = torch.zeros(B, 1, device=device, dtype=dtype)
        theta = torch.atan2(robot[:, 7:8], robot[:, 6:7])
        # (px, py, radius, gx, gy, v_pref, theta). Robot-relative frame:
        # px = py = 0 and the goal is relative.
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

        edges, mask = self._human_slots(
            neighbors, neighbor_mask, neighbor_history, neighbor_history_mask
        )
        if self.obstacle_tokenizer is not None:
            assert ray_features is not None
            obs_edges, obs_mask = self._obstacle_slots(ray_features)
            edges = torch.cat((edges, obs_edges), dim=1)
            mask = torch.cat((mask, obs_mask), dim=1)

        # Every invalid slot (padded human or non-hit ray) -> far away; then the
        # official "at least one visible slot" rule. No .any() -> no GPU sync.
        edges = torch.where(mask.unsqueeze(-1), edges, edges.new_full((), _FAR_AWAY))
        first = mask[:, :1] | ~mask.any(dim=-1, keepdim=True)
        mask = torch.cat((first, mask[:, 1:]), dim=1)

        return {
            "robot_node": robot_node,
            "temporal_edges": temporal_edges,
            "spatial_edges": edges,
            "visible_masks": mask.to(dtype),
            # Unused with sort_humans=False; kept for official-input parity.
            "detected_human_num": mask.sum(dim=-1, keepdim=True, dtype=torch.int32),
        }

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
            ray_features,
        )
        # Only sizes of the edge-RNN state are read by the official forward.
        rnn_hxs = {
            "human_node_rnn": hidden_state.reshape(B, 1, -1),
            "human_human_edge_rnn": hidden_state.new_zeros(B, 1, _EDGE_RNN_SIZE),
        }
        # Official code fixes nenv at construction (num_processes); we batch freely.
        self.base.nenv = B
        value, actor_features, new_rnn_hxs = self.base(
            inputs, rnn_hxs, not_done_mask.reshape(B, 1), infer=True
        )
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
            "Use forward()/act(); official-format inputs would skip the obstacle slots."
        )

    def evaluate_actions(self, *args, **kwargs):
        raise NotImplementedError(
            "Use forward(); official-format inputs would skip the obstacle slots."
        )

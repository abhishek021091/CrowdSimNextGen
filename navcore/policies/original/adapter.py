"""CrowdSimNextGen Adapter for the official CrowdNav++ policy with decoupled obstacle integration.

Architecture:
1. Real Humans -> Untouched CrowdNav++ Attention Pipeline (SpatialEdgeSelfAttn + EdgeAttention_M)
   -> 256-dim Original Crowd Context.
2. Obstacle Rays -> Pseudo-human Conversion (12-dim tokens with static future positions)
   -> Separate Lightweight Obstacle Encoder (MLP) -> Aggregated Obstacle Context.
3. Concatenate: [Original Crowd Context || Obstacle Context] -> Linear Obstacle Fusion.
4. Untouched Original EndRNN (human_node_rnn GRU) -> Actor / Critic -> Action Distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import types

import numpy as np
import torch
from torch import Tensor, nn
from gymnasium import spaces

from navcore.entities.components.sensors.obstacle_detector import RAY_FEATURE_DIM
from navcore.policies.original.obstacle_tokenizer import (
    ObstacleTokenizer,
    ObstacleTokenizerConfig,
)
from navcore.policies.original.policy import Policy
from navcore.policies.original.selfAttn_srnn_temp_node import reshapeT


class ObstacleMode(Enum):
    """How static obstacles (ray-cast hits) reach the policy."""

    NONE = "none"
    ENCODER = "encoder"
    POINT_TOKENS = "point_tokens"


@dataclass(slots=True, frozen=True)
class CrowdNavPPPolicyConfig:
    """Hyperparameters for CrowdNavPPPolicy."""

    robot_feature_dim: int = 8
    neighbor_feature_dim: int = 5
    max_neighbors: int = 10
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

    @property
    def uses_ray_features(self) -> bool:
        """Whether forward() needs ray_features in this mode."""
        return self.obstacle_mode is not ObstacleMode.NONE

    @property
    def spatial_edge_feature_dim(self) -> int:
        return 12


class ObstacleTokenEncoder(nn.Module):
    """Lightweight obstacle encoder for pseudo-human obstacle tokens.

    Encodes each valid ray hit into a feature embedding and aggregates them
    via permutation-invariant masked pooling across rays to produce a single
    fixed-size obstacle context vector.
    """

    def __init__(
        self,
        token_dim: int = 12,
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
        """Encode and aggregate obstacle tokens.

        Args:
            obstacle_tokens: [B, num_rays, token_dim] pseudo-human obstacle tokens.
            hit_mask: [B, num_rays] boolean mask indicating valid ray hits.

        Returns:
            [B, output_dim] fixed-size obstacle context vector.
        """
        features = self.mlp(obstacle_tokens)  # [B, num_rays, output_dim]
        mask = hit_mask.unsqueeze(-1).to(features.dtype)  # [B, num_rays, 1]
        masked_features = features * mask

        # Permutation-invariant masked mean pooling
        hit_counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = masked_features.sum(dim=1) / hit_counts

        # If zero hits, return exact zero vector
        has_hits = hit_mask.any(dim=-1, keepdim=True).to(features.dtype)
        return pooled * has_hits


class CrowdNavPPPolicy(Policy):
    """CrowdNav++ recurrent policy adapter with decoupled obstacle integration.

    Inherits from the official untouched CrowdNav++ Policy class:
    - 100% parameter name and hierarchy compatibility with original checkpoints.
    - Checkpoint loading works directly with 0 missing and 0 unexpected keys for base & dist.
    - Real humans only flow through the untouched CrowdNav++ attention layers.
    - Obstacle tokens are encoded separately and concatenated immediately before EndRNN.
    """

    def __init__(
        self,
        config: CrowdNavPPPolicyConfig | None = None,
        gst_predictor: nn.Module | None = None,
        obstacle_encoder: nn.Module | None = None,
    ) -> None:
        if config is None:
            config = CrowdNavPPPolicyConfig()
        self.config = config

        base_kwargs = types.SimpleNamespace(
            human_node_rnn_size=config.rnn_hidden_size,
            human_human_edge_rnn_size=256,
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
            no_cuda=False,
        )

        obs_space_dict = {
            "robot_node": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7)),
            "temporal_edges": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2)),
            "spatial_edges": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(config.max_neighbors, 12),
            ),
            "visible_masks": spaces.Box(
                low=-np.inf, high=np.inf, shape=(config.max_neighbors,)
            ),
            "detected_human_num": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        }
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.action_dim,), dtype=np.float32
        )

        if (
            config.obstacle_mode == ObstacleMode.POINT_TOKENS
            and obstacle_encoder is not None
        ):
            raise ValueError(
                "An obstacle_encoder was injected but obstacle_mode=POINT_TOKENS; "
                "only ObstacleMode.ENCODER uses an injected obstacle_encoder."
            )

        super().__init__(
            obs_space_dict,
            action_space,
            base="selfAttn_merge_srnn",
            base_kwargs=base_kwargs,
        )

        # In original selfAttn_merge_SRNN, human_node_final_linear was only defined
        # for an auxiliary prediction loss and is not part of the RL actor-critic graph.
        self.base.human_node_final_linear.requires_grad_(False)

        self.human_num = config.max_neighbors
        self.gst_predictor = gst_predictor

        # 1. Obstacle Tokenizer (POINT_TOKENS mode)
        self.obstacle_tokenizer = (
            ObstacleTokenizer(
                ObstacleTokenizerConfig(
                    max_range=config.obstacle_max_range,
                    hit_radius=config.obstacle_hit_radius,
                )
            )
            if config.obstacle_mode == ObstacleMode.POINT_TOKENS
            else None
        )

        # 2. Obstacle Encoder and Fusion
        self.obstacle_dim = config.obstacle_context_dim
        if config.obstacle_mode == ObstacleMode.POINT_TOKENS:
            self.obstacle_encoder = ObstacleTokenEncoder(
                token_dim=12, hidden_dim=64, output_dim=self.obstacle_dim
            )
            # Linear fusion immediately before EndRNN: [256 + obstacle_dim -> 256]
            self.obstacle_fusion = nn.Linear(256 + self.obstacle_dim, 256)
            with torch.no_grad():
                self.obstacle_fusion.bias.zero_()
                self.obstacle_fusion.weight[:, :256] = torch.eye(256)
                nn.init.orthogonal_(self.obstacle_fusion.weight[:, 256:], gain=1.0)
        elif config.obstacle_mode == ObstacleMode.ENCODER:
            # Ablation baseline
            if obstacle_encoder is None:
                from navcore.policies.crowdnav_pp.obstacle_encoder import (
                    ObstacleEncoder as CNNEncoder,
                    ObstacleEncoderConfig as CNNEncoderConfig,
                )

                self.obstacle_encoder = CNNEncoder(
                    CNNEncoderConfig(
                        ray_feature_dim=RAY_FEATURE_DIM,
                        embedding_dim=self.obstacle_dim,
                    )
                )
            else:
                self.obstacle_encoder = obstacle_encoder
                self.obstacle_dim = getattr(
                    obstacle_encoder.config,
                    "output_dim",
                    getattr(obstacle_encoder.config, "embedding_dim", 128),
                )

            self.obstacle_fusion = nn.Linear(256 + self.obstacle_dim, 256)
            with torch.no_grad():
                self.obstacle_fusion.bias.zero_()
                self.obstacle_fusion.weight[:, :256] = torch.eye(256)
                nn.init.orthogonal_(self.obstacle_fusion.weight[:, 256:], gain=1.0)
        else:
            self.obstacle_encoder = None
            self.obstacle_fusion = None

    @property
    def human_human_attention(self):
        """Adapter property exposing human-human spatial attention."""

        class _HHAWrapper:
            def __init__(self, spatial_attn):
                self.embed = spatial_attn.embedding_layer

        return _HHAWrapper(self.base.spatial_attn)

    @property
    def action_head(self):
        """Adapter property exposing action head parameters for trainer metrics."""

        class _ActionHeadWrapper:
            LOG_STD_MIN = -3.0
            LOG_STD_MAX = 0.5

            def __init__(self, dist):
                self._dist = dist

            @property
            def log_std(self):
                return self._dist.logstd._bias.squeeze()

        return _ActionHeadWrapper(self.dist)

    def initial_hidden_state(
        self, nenv: int = 1, device: torch.device | None = None
    ) -> Tensor:
        """Return zeroed initial hidden state for nenv environments."""
        return torch.zeros(nenv, self.config.rnn_hidden_size, device=device)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load state dict, supporting both raw checkpoints and trainer checkpoints."""
        if "policy_state_dict" in state_dict:
            state_dict = state_dict["policy_state_dict"]
        # If loading an original checkpoint that has no obstacle_ parameters:
        has_obstacle = any(k.startswith("obstacle_") for k in state_dict.keys())
        if not has_obstacle:
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

    def _build_inputs(
        self,
        robot: Tensor,
        neighbors: Tensor,
        neighbor_mask: Tensor,
        neighbor_history: Tensor,
        neighbor_history_mask: Tensor,
        ray_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Convert environment observations to CrowdNav++ input dictionary."""
        if self.config.uses_ray_features and ray_features is None:
            raise ValueError("ray_features required in obstacle mode.")
        if not self.config.uses_ray_features and ray_features is not None:
            raise ValueError("ray_features was given but obstacle_mode is NONE.")

        B = robot.size(0)
        device = robot.device
        dtype = robot.dtype

        # 1. Temporal edges: [B, 1, 2] -> (vx, vy)
        temporal_edges = robot[:, 2:4].unsqueeze(1)

        # 2. Robot node: [B, 1, 7] -> (px=0, py=0, radius, gx, gy, v_pref, theta)
        px = torch.zeros(B, 1, device=device, dtype=dtype)
        py = torch.zeros(B, 1, device=device, dtype=dtype)
        gx = robot[:, 0:1]
        gy = robot[:, 1:2]
        v_pref = robot[:, 4:5]
        radius = robot[:, 5:6]
        theta = torch.atan2(robot[:, 7:8], robot[:, 6:7])
        robot_node = torch.cat(
            [px, py, radius, gx, gy, v_pref, theta], dim=-1
        ).unsqueeze(1)

        # 3. Spatial edges for REAL HUMANS ONLY: [B, max_neighbors, 12]
        N = neighbors.size(1)
        rel_pos = neighbors[..., 0:2]
        vel = neighbors[..., 2:4]

        if self.config.use_gst_prediction and self.gst_predictor is not None:
            hist_pos = neighbor_history[..., 0:2].transpose(1, 2)
            hist_vel = neighbor_history[..., 2:4].transpose(1, 2)
            hist_mask = neighbor_history_mask.transpose(1, 2)
            pred_mean = self.gst_predictor.predict_features(
                hist_pos, hist_vel, hist_mask
            )
            pred_displacements = pred_mean.view(B, N, 5, 2)
            future_pos = rel_pos.unsqueeze(2) + pred_displacements
            future_features = future_pos.reshape(B, N, 10)
        else:
            dt = 0.25
            steps = torch.arange(1, 6, device=device, dtype=dtype).view(1, 1, 5, 1)
            future_pos = rel_pos.unsqueeze(2) + vel.unsqueeze(2) * (steps * dt)
            future_features = future_pos.reshape(B, N, 10)

        spatial_edges = torch.cat([rel_pos, future_features], dim=-1)

        # Pad or slice to self.human_num
        if N < self.human_num:
            pad = spatial_edges.new_zeros(B, self.human_num - N, 12)
            spatial_edges = torch.cat([spatial_edges, pad], dim=1)
            mask_pad = neighbor_mask.new_zeros(B, self.human_num - N)
            mask = torch.cat([neighbor_mask.bool(), mask_pad.bool()], dim=1)
        else:
            spatial_edges = spatial_edges[:, : self.human_num]
            mask = neighbor_mask[:, : self.human_num].bool()

        detected = mask.sum(dim=-1, keepdim=True).int()
        # CrowdNav++ requires at least 1 human slot in attention to avoid empty masks
        no_humans = (detected == 0).squeeze(-1)
        if no_humans.any():
            mask = mask.clone()
            mask[no_humans, 0] = True
            detected = mask.sum(dim=-1, keepdim=True).int()

        return {
            "robot_node": robot_node,
            "temporal_edges": temporal_edges,
            "spatial_edges": spatial_edges,
            "visible_masks": mask.float(),
            "detected_human_num": detected,
        }

    def _forward_core(
        self,
        inputs: dict[str, Tensor],
        rnn_hxs: Tensor | dict[str, Tensor],
        masks: Tensor,
        ray_features: Tensor | None = None,
        infer: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Execute the forward pass: real humans through untouched attention,
        obstacles encoded separately, fused immediately before EndRNN.
        """
        B = inputs["robot_node"].shape[0] if infer else self.base.nenv
        nenv = B
        seq_length = 1 if infer else self.base.seq_length
        self.base.nenv = nenv

        robot_node = reshapeT(inputs["robot_node"], seq_length, nenv)
        temporal_edges = reshapeT(inputs["temporal_edges"], seq_length, nenv)
        spatial_edges = reshapeT(inputs["spatial_edges"], seq_length, nenv)
        detected_human_num = (
            inputs["detected_human_num"].squeeze(-1).to(device=robot_node.device).int()
        )

        # Handle tensor or dict rnn_hxs
        if isinstance(rnn_hxs, Tensor):
            node_rnn_hxs = rnn_hxs.view(1, nenv, 1, self.config.rnn_hidden_size)
        elif isinstance(rnn_hxs, dict):
            node_rnn_hxs = reshapeT(rnn_hxs["human_node_rnn"], 1, nenv)
        else:
            raise TypeError(f"Unsupported rnn_hxs type: {type(rnn_hxs)}")

        if masks.dim() == 1:
            masks = masks.view(seq_length, nenv, 1)
        else:
            masks = reshapeT(masks, seq_length, nenv)

        # 1. Robot states (untouched original linear layer)
        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.base.robot_linear(robot_states)

        # 2. REAL HUMANS ONLY through untouched attention pipeline
        if self.base.args.use_self_attn:
            spatial_attn_out = self.base.spatial_attn(
                spatial_edges, detected_human_num
            ).view(seq_length, nenv, self.human_num, -1)
        else:
            spatial_attn_out = spatial_edges
        output_spatial = self.base.spatial_linear(spatial_attn_out)

        # Original crowd context
        hidden_attn_weighted, _ = self.base.attn(
            robot_states, output_spatial, detected_human_num
        )

        # 3. OBSTACLE BRANCH (completely separate from human attention)
        if (
            self.config.obstacle_mode == ObstacleMode.POINT_TOKENS
            and ray_features is not None
            and self.obstacle_tokenizer is not None
            and self.obstacle_encoder is not None
        ):
            # Step 1: Convert ray hits into pseudo-human tokens
            obs_geom, obs_hit_mask = self.obstacle_tokenizer.tokenize(ray_features)
            obs_pos = obs_geom[..., 0:2]
            obs_future = obs_pos.unsqueeze(2).repeat(1, 1, 5, 1).reshape(nenv, -1, 10)
            obs_tokens = torch.cat(
                [obs_pos, obs_future], dim=-1
            )  # [nenv, num_rays, 12]

            # Step 3 & 4: Encode and aggregate obstacle tokens into fixed-size context
            obs_context = self.obstacle_encoder(
                obs_tokens, obs_hit_mask
            )  # [nenv, obstacle_dim]
            obs_context = obs_context.view(seq_length, nenv, 1, -1)

            # Step 5: Concatenate Original Crowd Context || Obstacle Context
            concat_context = torch.cat([hidden_attn_weighted, obs_context], dim=-1)

            # Step 6: Fuse and feed into EndRNN
            fused_context = self.obstacle_fusion(concat_context)
        elif (
            self.config.obstacle_mode == ObstacleMode.ENCODER
            and ray_features is not None
            and self.obstacle_encoder is not None
        ):
            raw_emb = self.obstacle_encoder(ray_features)
            # Pool across rays
            hit_mask = ray_features[..., 0] > 0.5
            mask = hit_mask.unsqueeze(-1).to(raw_emb.dtype)
            pooled = (raw_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            has_hits = hit_mask.any(dim=-1, keepdim=True).to(raw_emb.dtype)
            obs_context = (pooled * has_hits).view(seq_length, nenv, 1, -1)
            concat_context = torch.cat([hidden_attn_weighted, obs_context], dim=-1)
            fused_context = self.obstacle_fusion(concat_context)
        else:
            fused_context = hidden_attn_weighted

        # 4. Untouched EndRNN GRU
        outputs, h_nodes = self.base.humanNodeRNN(
            robot_states, fused_context, node_rnn_hxs, masks
        )

        # 5. Untouched Actor and Critic
        x = outputs[:, :, 0, :]
        hidden_critic = self.base.critic(x)
        hidden_actor = self.base.actor(x)

        if infer:
            value = self.base.critic_linear(hidden_critic).squeeze(0)
            actor_features = hidden_actor.squeeze(0)
            new_hidden = h_nodes.squeeze(0).squeeze(1)  # [nenv, rnn_hidden_size]
            return value, actor_features, new_hidden
        else:
            value = self.base.critic_linear(hidden_critic).view(-1, 1)
            actor_features = hidden_actor.view(-1, self.base.output_size)
            new_hidden = h_nodes.squeeze(0).squeeze(1)
            return value, actor_features, new_hidden

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
        """Run one recurrent tick for a batch of environments."""
        inputs = self._build_inputs(
            robot_features,
            neighbor_features,
            neighbor_mask,
            neighbor_history,
            neighbor_history_mask,
            ray_features=ray_features,
        )
        value, actor_features, new_hidden = self._forward_core(
            inputs,
            hidden_state,
            not_done_mask,
            ray_features=ray_features,
            infer=True,
        )
        distribution = self.dist(actor_features)
        return distribution, value, new_hidden

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
        """Convenience wrapper: run forward and sample action."""
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
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_prob(action)
        return action, action_log_probs, value, new_hidden

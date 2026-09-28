# navcore/policies/crowdnav_pp/obstacle_tokenizer.py
"""ObstacleTokenizer: turns ObstacleDetector ray hits into pseudo-human tokens.

Purpose:
    Every ray that hit something becomes one token laid out exactly like an
    ``ObservationEncoder`` neighbor feature -- ``(rel_px, rel_py, vx, vy,
    radius)`` -- with zero velocity and a fixed radius. The policy then
    treats obstacles as stationary pedestrians: they go through the same
    embedding MLP and attention layers as real humans, so whatever the
    network learns about "keep clear of a person standing there" transfers
    to walls and tables for free.

Responsibilities:
    - Un-normalize ``scan_to_features``' ``dx/max_range`` channels back to
      meters, so obstacle tokens live on the same scale as human features
      (``ObservationEncoder`` stores human relative positions in raw
      meters; a shared embedding MLP must not see two scales in one slot).
    - Emit a validity mask from the scan's ``hit_mask`` channel. A ray that
      hit nothing is padding, not "an obstacle at the robot".

Deliberately NOT responsible for:
    - The type flag, temporal-embedding padding and GST padding a token
      needs before it can sit next to a human token. Those widths belong
      to ``CrowdNavPPPolicyConfig``, so the policy appends them
      (``CrowdNavPPPolicy._pad_obstacle_tokens``).

Assumptions / coupling (real, flagged):
    - ``max_range`` must equal the ``ObstacleDetectorConfig.max_range`` the
      scan was cast with. A mismatch silently rescales every obstacle
      position. The training/eval entry points pass one value to both.
    - Channel layout is ``scan_to_features``' ``[hit_mask, dx_norm,
      dy_norm]`` (``RAY_FEATURE_DIM == 3``).
    - Ray index carries no identity: ray i hits different points as the
      robot moves. That is why these tokens have no temporal history.

Performance: one allocation-light pass of a few elementwise ops over
``[nenv, num_rays, 3]``. No parameters, no state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from navcore.entities.components.sensors.obstacle_detector import RAY_FEATURE_DIM

_HIT_CHANNEL = 0
_DX_CHANNEL = 1
_DY_CHANNEL = 2


@dataclass(slots=True, frozen=True)
class ObstacleTokenizerConfig:
    """Parameters for :class:`ObstacleTokenizer`.

    Attributes:
        max_range: The ``ObstacleDetectorConfig.max_range`` the scans were
            cast with; used to convert normalized hit offsets back to meters.
        hit_radius: Radius assigned to every hit token, in meters.
    """

    max_range: float
    hit_radius: float = 0.1

    def __post_init__(self) -> None:
        if self.max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {self.max_range!r}.")
        if self.hit_radius <= 0.0:
            raise ValueError(f"hit_radius must be positive, got {self.hit_radius!r}.")


class ObstacleTokenizer:
    """Stateless, parameter-free ray-hit -> pseudo-human token converter."""

    #: Width of the geometric token: (rel_px, rel_py, vx, vy, radius).
    TOKEN_DIM = 5

    def __init__(self, config: ObstacleTokenizerConfig) -> None:
        self.config = config

    def tokenize(self, ray_features: Tensor) -> tuple[Tensor, Tensor]:
        """Convert one batch of ray scans into pseudo-human tokens.

        Args:
            ray_features: ``[nenv, num_rays, RAY_FEATURE_DIM]`` as produced
                by ``scan_to_features`` (one row per ray).

        Returns:
            ``(tokens, mask)``. ``tokens`` is ``[nenv, num_rays, 5]`` laid
            out as ``(rel_px, rel_py, vx, vy, radius)``; rows for rays that
            hit nothing are all-zero, matching how ``ObservationEncoder``
            zero-fills padded neighbor slots. ``mask`` is ``[nenv,
            num_rays]`` boolean, True where the ray hit something.
            No side effects.

        Raises:
            ValueError: If the last dimension is not ``RAY_FEATURE_DIM``.
        """
        if ray_features.shape[-1] != RAY_FEATURE_DIM:
            raise ValueError(
                f"ray_features' last dim is {ray_features.shape[-1]}, expected "
                f"RAY_FEATURE_DIM={RAY_FEATURE_DIM}."
            )

        mask = ray_features[..., _HIT_CHANNEL] > 0.5
        relative = (
            ray_features[..., _DX_CHANNEL : _DY_CHANNEL + 1] * self.config.max_range
        )
        velocity = torch.zeros_like(relative)
        radius = torch.full_like(relative[..., :1], self.config.hit_radius)

        tokens = torch.cat((relative, velocity, radius), dim=-1)
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, mask

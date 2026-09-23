# navcore/policies/crowdnav_pp/obstacle_encoder.py
"""ObstacleEncoder: 1D CNN-based embedding of one ObstacleDetector ray scan.

SUPERSEDED, not deleted:
    This is the obstacle branch's *previous* design (a 1D CNN pooling a
    per-ray feature scan into a single embedding, concatenated into
    RobotHumanAttention as extra key/value tokens). It has been replaced
    end-to-end by the range-image branch: see
    ``navcore.entities.components.sensors.range_image.RangeImageBuilder``
    and ``navcore.policies.crowdnav_pp.range_image_encoder.
    RangeImageEncoder`` for the current design, and
    ``navcore.policies.crowdnav_pp.policy``'s module docstring for the
    full architectural rationale (this pooled design is exactly what
    ``navcore/probe_obstacle_encoder_mirror.py`` found destroys
    left/right directional information).

    ``CrowdNavPPPolicy`` no longer imports or constructs this class.
    This file is kept only because some standalone scripts may still
    reference it directly; do not wire it into new policy code.

Turns one tick's ObstacleScan (navcore.entities.components.sensors.
obstacle_detector) into a fixed-width embedding, using a 1D Convolutional
Neural Network.

A LiDAR-style ray scan forms a ring around the robot. To capture spatial
patterns properly without introducing a seam at the start/end of the array,
circular padding is applied to all convolutions. The local receptive field of
the CNN naturally detects gaps, corners, and obstacle boundaries from adjacent
rays, independent of the robot's heading.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from navcore.entities.components.sensors.obstacle_detector import (
    RAY_FEATURE_DIM,
    scan_to_features,
)

# Re-exported for backward compatibility -- callers that used to import
# `scan_to_features`/`RAY_FEATURE_DIM` from this module still can, but the
# canonical, single copy now lives in obstacle_detector.py (see that
# module's "Consolidation note"). Do not add a second implementation here.
__all__ = [
    "ObstacleEncoder",
    "ObstacleEncoderConfig",
    "scan_to_features",
    "RAY_FEATURE_DIM",
]


@dataclass(slots=True, frozen=True)
class ObstacleEncoderConfig:
    """Hyperparameters for :class:`ObstacleEncoder`.

    Attributes:
        ray_feature_dim: Width of one ray's feature vector. Defaults to
            `RAY_FEATURE_DIM` (6).
        embedding_dim: Dimensionality of the final output embedding.
        conv_channels: Number of channels for each of the 3 Conv1d layers.
        kernel_sizes: Kernel sizes for each of the 3 Conv1d layers.
    """

    ray_feature_dim: int = RAY_FEATURE_DIM
    embedding_dim: int = 128
    conv_channels: tuple[int, ...] = (64, 128, 128)
    kernel_sizes: tuple[int, ...] = (5, 5, 3)

    def __post_init__(self) -> None:
        """Validates configuration values."""
        if self.ray_feature_dim <= 0:
            raise ValueError(
                f"ray_feature_dim must be positive, got {self.ray_feature_dim!r}."
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim!r}."
            )
        if len(self.conv_channels) != 3:
            raise ValueError(
                f"conv_channels must have exactly 3 elements, got {self.conv_channels!r}."
            )
        if len(self.kernel_sizes) != 3:
            raise ValueError(
                f"kernel_sizes must have exactly 3 elements, got {self.kernel_sizes!r}."
            )
        for c in self.conv_channels:
            if c <= 0:
                raise ValueError(f"All conv_channels must be positive, got {c!r}.")
        for k in self.kernel_sizes:
            if k <= 0 or k % 2 == 0:
                raise ValueError(
                    f"All kernel_sizes must be positive odd integers, got {k!r}."
                )

    @property
    def output_dim(self) -> int:
        """Width of the embedding `ObstacleEncoder.forward` returns."""
        return self.embedding_dim


class ObstacleEncoder(nn.Module):
    """1D CNN over one ray scan's rays, producing one fixed-width embedding.

    See module docstring: superseded by ``RangeImageEncoder`` for
    ``CrowdNavPPPolicy``'s default architecture. Left implemented and
    importable for backward compatibility only.
    """

    def __init__(self, config: ObstacleEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=config.ray_feature_dim,
                out_channels=config.conv_channels[0],
                kernel_size=config.kernel_sizes[0],
                padding=config.kernel_sizes[0] // 2,
                padding_mode="circular",
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=config.conv_channels[0],
                out_channels=config.conv_channels[1],
                kernel_size=config.kernel_sizes[1],
                padding=config.kernel_sizes[1] // 2,
                padding_mode="circular",
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=config.conv_channels[1],
                out_channels=config.conv_channels[2],
                kernel_size=config.kernel_sizes[2],
                padding=config.kernel_sizes[2] // 2,
                padding_mode="circular",
            ),
            nn.ReLU(),
        )
        self.project = nn.Sequential(
            nn.Linear(config.conv_channels[2], config.embedding_dim),
            nn.ReLU(),
        )

    def forward(self, ray_features: Tensor) -> Tensor:
        """Encode one (batch of) ray scan(s) into a fixed-width embedding.

        Args:
            ray_features: `[..., num_rays, ray_feature_dim]` -- any
                leading batch shape. Only the last two dims are consumed;
                every leading dim is preserved.

        Returns:
            `[..., config.output_dim]` -- one obstacle embedding per
            leading-shape slot, ready to concatenate alongside other embeddings.

        Raises:
            ValueError: If `ray_features`'s last dimension doesn't match
                `config.ray_feature_dim`.
        """
        if ray_features.shape[-1] != self.config.ray_feature_dim:
            raise ValueError(
                f"ray_features' last dim is {ray_features.shape[-1]}, but "
                f"this encoder was configured for ray_feature_dim="
                f"{self.config.ray_feature_dim}."
            )

        *leading, num_rays, feature_dim = ray_features.shape
        batch = 1
        for dim in leading:
            batch *= dim

        flat_batch = ray_features.reshape(batch, num_rays, feature_dim)
        conv_input = flat_batch.transpose(1, 2)  # [batch, feat, num_rays]
        conv_out = self.net(conv_input)  # [batch, channels, num_rays]
        conv_out = conv_out.transpose(1, 2)  # [batch, num_rays, channels]
        embedding = self.project(conv_out)

        # Reshape back to the original leading batch dimensions
        return embedding.reshape(*leading, num_rays, self.config.output_dim)

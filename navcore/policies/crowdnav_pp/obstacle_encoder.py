# navcore/policies/crowdnav_pp/obstacle_encoder.py
"""ObstacleEncoder: 1D CNN-based embedding of one ObstacleDetector ray scan.

Turns one tick's ObstacleScan (navcore.sensor.obstacle_detector) into a
fixed-width embedding, using a 1D Convolutional Neural Network. The output is
meant to be concatenated into CrowdNavPPPolicy's existing robot/crowd embedding
space alongside RobotStateEncoder and TemporalEncoder.

A LiDAR-style ray scan forms a ring around the robot. To capture spatial
patterns properly without introducing a seam at the start/end of the array,
circular padding is applied to all convolutions. The local receptive field of
the CNN naturally detects gaps, corners, and obstacle boundaries from adjacent
rays, independent of the robot's heading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from navcore.sensor.obstacle_detector import ObstacleScan

#: Per-ray feature layout fed to the CNN:
#: [hit_mask, distance_norm, dx_norm, dy_norm, sin(ray_angle), cos(ray_angle)].
RAY_FEATURE_DIM = 6


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

    Extracts spatial features using circular padding to respect the ring
    topology of a LiDAR scan, pools them, and applies a linear projection.

    Attributes:
        config: This encoder's hyperparameters.
        net: The sequential CNN model.
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
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
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

        # Reshape to [batch, num_rays, feature_dim]
        flat_batch = ray_features.reshape(batch, num_rays, feature_dim)

        # Conv1d expects [batch, channels, length], so transpose the last two dims
        # resulting in [batch, feature_dim, num_rays]
        conv_input = flat_batch.transpose(1, 2)

        # Forward pass through CNN
        embedding = self.net(conv_input)

        # Reshape back to the original leading batch dimensions
        return embedding.reshape(*leading, self.config.output_dim)


def scan_to_features(scan: ObstacleScan, max_range: float) -> npt.NDArray[np.float32]:
    """Convert one `ObstacleScan` into the `(num_rays, RAY_FEATURE_DIM)`
    array `ObstacleEncoder` expects.

    Args:
        scan: One tick's ray-casting result (obstacles + boundary).
        max_range: The `ObstacleDetectorConfig.max_range` the scan was cast
            with, used to normalize distances and relative positions into
            a continuous scale.

    Returns:
        `(num_rays, RAY_FEATURE_DIM)` float32 array:
        `[hit_mask, distance / max_range, dx / max_range, dy / max_range,
        sin(theta), cos(theta)]` per ray, where theta is the ray angle.
    """
    num_rays = scan.hit_mask.shape[0]
    features = np.zeros((num_rays, RAY_FEATURE_DIM), dtype=np.float32)

    # 1. hit_mask (0 or 1)
    features[:, 0] = scan.hit_mask.astype(np.float32)

    # 2. distance_norm
    features[:, 1] = scan.distances / max_range

    # 3. dx_norm
    features[:, 2] = scan.relative_positions[:, 0] / max_range

    # 4. dy_norm
    features[:, 3] = scan.relative_positions[:, 1] / max_range

    # 5 & 6. Ray angles (uniformly distributed over 0 to 2*pi)
    indices = np.arange(num_rays, dtype=np.float32)
    theta = 2.0 * np.pi * indices / num_rays
    features[:, 4] = np.sin(theta)
    features[:, 5] = np.cos(theta)

    return features

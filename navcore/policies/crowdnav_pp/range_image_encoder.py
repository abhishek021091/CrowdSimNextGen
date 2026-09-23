"""RangeImageEncoder: residual CNN backbone over a binary range image,
producing a compact set of angularly-tagged obstacle tokens.

Consumes `navcore.entities.components.sensors.range_image.RangeImageBuilder`'s
output. This is the new obstacle branch's perception stage; see
`navcore.policies.crowdnav_pp.policy`'s module docstring for how its
output (obstacle tokens) is consumed by `RobotObstacleAttention` and fused
with the human branch.

Pipeline:
    range image [B, 1, H, W]
        -> stem conv
        -> N residual stages (each optionally halving H, W)
        -> feature map [B, C, H', W']
        -> height-collapse conv: strided conv spanning the full
           remaining height -> [B, D, 1, W']
        -> token-compress conv: strided conv tiling W' into
           num_obstacle_tokens equal-width groups -> [B, D, 1, num_tokens]
        -> LayerNorm + learned angular positional embedding
        -> obstacle tokens [B, num_obstacle_tokens, D]

Circular padding, angular dimension only:
    The range image's width axis is a laser ray fan wrapped in angle
    (see `RangeImageBuilder`) -- ray 0 and ray W-1 are angularly adjacent
    in the real world even though they sit at opposite edges of the
    image. Every conv in the backbone therefore pads its width dimension
    circularly and its height dimension with zeros (the height axis is
    genuinely bounded: row 0 is `max_range`, the far edge of the sensing
    square, and there is nothing "beyond" it to wrap into). This is
    implemented as an explicit `F.pad` + valid-mode `F.conv2d`, since
    `nn.Conv2d`'s own `padding_mode` applies uniformly to both spatial
    dims and cannot mix circular/zero per-axis.

No global pooling, ever:
    Average/max pooling to a single vector would discard exactly the
    left/right directional information the previous 1D-CNN obstacle
    branch's own diagnostic probe
    (`navcore/probe_obstacle_encoder_mirror.py`) found destroyed by
    pooling, even in an untrained encoder. Every spatial reduction here
    is a strided, *learned* convolution that keeps a spatial (angular)
    axis alive all the way to the token sequence -- "compression," never
    "pooling."

Exact token count, by construction, not by interpolation:
    `config.num_obstacle_tokens` must evenly divide the backbone's
    output width (`__post_init__` validates this and raises a clear
    error naming a compatible value otherwise). The token-compress conv
    then uses `kernel_size == stride == (output_width //
    num_obstacle_tokens)` along the width axis -- an exact, non-
    overlapping tiling, with no interpolation or padding needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class RangeImageEncoderConfig:
    """Hyperparameters for :class:`RangeImageEncoder`.

    Attributes:
        in_height: Range image height (num_range_bins). Must match
            `RangeImageBuilderConfig.num_range_bins`.
        in_width: Range image width (num_rays). Must match
            `RangeImageBuilderConfig.num_rays`.
        stem_channels: Output channels of the initial stem conv.
        stage_channels: Output channels of each residual stage, in
            order. The last entry is the backbone's feature-map width
            (128 by default, matching the architecture spec).
        blocks_per_stage: Number of residual blocks in each stage. Same
            length as `stage_channels`.
        downsample_after_stage: Whether each stage's *first* block uses
            stride 2 (halving H and W). Same length as `stage_channels`.
        token_embedding_dim: Width of each output obstacle token. Must
            equal `stage_channels[-1]` (the height-collapse conv reads
            directly from the backbone's channel width) and should
            match `RobotObstacleAttentionConfig.obstacle_embedding_dim`.
        num_obstacle_tokens: Number of tokens the angular axis is
            compressed to. Must evenly divide the backbone's output
            width -- see module docstring.
    """

    in_height: int = 128
    in_width: int = 180
    stem_channels: int = 32
    stage_channels: tuple[int, ...] = (32, 64, 128)
    blocks_per_stage: tuple[int, ...] = (2, 2, 2)
    downsample_after_stage: tuple[bool, ...] = (True, True, False)
    token_embedding_dim: int = 128
    num_obstacle_tokens: int = 15

    def __post_init__(self) -> None:
        if len(self.stage_channels) != len(self.blocks_per_stage):
            raise ValueError(
                "stage_channels and blocks_per_stage must have the same "
                f"length, got {len(self.stage_channels)} and "
                f"{len(self.blocks_per_stage)}."
            )
        if len(self.stage_channels) != len(self.downsample_after_stage):
            raise ValueError(
                "stage_channels and downsample_after_stage must have the "
                f"same length, got {len(self.stage_channels)} and "
                f"{len(self.downsample_after_stage)}."
            )
        if self.in_height <= 0 or self.in_width <= 0:
            raise ValueError("in_height and in_width must be positive.")
        if self.token_embedding_dim <= 0:
            raise ValueError("token_embedding_dim must be positive.")
        if self.num_obstacle_tokens <= 0:
            raise ValueError("num_obstacle_tokens must be positive.")
        if self.stage_channels[-1] != self.token_embedding_dim:
            raise ValueError(
                f"stage_channels[-1] ({self.stage_channels[-1]}) must equal "
                f"token_embedding_dim ({self.token_embedding_dim}) -- the "
                f"height-collapse conv reads the backbone's final channel "
                f"width directly."
            )

        backbone_width = self.in_width
        for downsample in self.downsample_after_stage:
            if downsample:
                backbone_width = backbone_width // 2
        if backbone_width % self.num_obstacle_tokens != 0:
            raise ValueError(
                f"Backbone output width ({backbone_width}, derived from "
                f"in_width={self.in_width} and downsample_after_stage="
                f"{self.downsample_after_stage}) does not evenly divide "
                f"num_obstacle_tokens={self.num_obstacle_tokens}. Choose a "
                f"num_obstacle_tokens that divides {backbone_width}, or "
                f"adjust in_width/downsample_after_stage."
            )

    @property
    def backbone_output_height(self) -> int:
        """Feature-map height right before the height-collapse conv."""
        height = self.in_height
        for downsample in self.downsample_after_stage:
            if downsample:
                height = height // 2
        return height

    @property
    def backbone_output_width(self) -> int:
        """Feature-map width right before the token-compress conv."""
        width = self.in_width
        for downsample in self.downsample_after_stage:
            if downsample:
                width = width // 2
        return width


class _AngularConv2d(nn.Module):
    """A single square-kernel Conv2d with circular width / zero height
    padding.

    A thin wrapper around raw conv weights (rather than `nn.Conv2d`
    itself) because `nn.Conv2d`'s own `padding_mode` cannot mix
    "circular" and "zeros" across the two spatial axes -- see module
    docstring.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size!r}.")
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, 0, 0), mode="circular")  # width: wraps
        x = F.pad(x, (0, 0, pad, pad), mode="constant", value=0.0)  # height: bounded
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=0)


class _ResidualBlock2D(nn.Module):
    """Two angular convs + BatchNorm + ReLU, with an identity/1x1 skip.

    The 1x1 skip projection (when channels or stride change) uses a
    plain `nn.Conv2d`, not `_AngularConv2d` -- a 1x1 kernel touches no
    neighboring column, so there is nothing to pad circularly.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _AngularConv2d(in_channels, out_channels, 3, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _AngularConv2d(out_channels, out_channels, 3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.skip is None else self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class RangeImageEncoder(nn.Module):
    """Residual CNN + angular token compression over a binary range image.

    Attributes:
        config: This encoder's hyperparameters.
        angular_position_embedding: One learned embedding per output
            token slot. Token slots correspond to fixed angular sectors
            every tick (the range image is always built in the same
            robot-centric, angle-ordered layout -- see
            `RangeImageBuilder`), so a per-slot learned embedding is
            sufficient; there is no need for a sinusoidal scheme keyed
            off a variable sequence length the way a transformer's
            token position usually is.
    """

    def __init__(self, config: RangeImageEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.stem = nn.Sequential(
            _AngularConv2d(1, config.stem_channels, 3, stride=1),
            nn.BatchNorm2d(config.stem_channels),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        in_channels = config.stem_channels
        for out_channels, n_blocks, downsample in zip(
            config.stage_channels,
            config.blocks_per_stage,
            config.downsample_after_stage,
        ):
            blocks = [
                _ResidualBlock2D(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride=2 if (downsample and i == 0) else 1,
                )
                for i in range(n_blocks)
            ]
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)

        backbone_channels = config.stage_channels[-1]
        out_height = config.backbone_output_height
        out_width = config.backbone_output_width
        tokens_per_group = out_width // config.num_obstacle_tokens

        # Collapse height to 1 with a single strided conv spanning the
        # full remaining height -- a strided convolution (learned
        # weights), not a pooling op.
        self.height_collapse = nn.Conv2d(
            backbone_channels,
            config.token_embedding_dim,
            kernel_size=(out_height, 1),
            stride=(out_height, 1),
        )

        # Collapse the remaining angular width down to num_obstacle_tokens
        # with a strided conv whose kernel exactly spans each token's
        # column group -- exact tiling by construction (see
        # RangeImageEncoderConfig.__post_init__'s divisibility check), so
        # no padding is needed here.
        self.token_compress = nn.Conv2d(
            config.token_embedding_dim,
            config.token_embedding_dim,
            kernel_size=(1, tokens_per_group),
            stride=(1, tokens_per_group),
        )

        self.token_norm = nn.LayerNorm(config.token_embedding_dim)
        self.angular_position_embedding = nn.Parameter(
            torch.zeros(config.num_obstacle_tokens, config.token_embedding_dim)
        )
        nn.init.normal_(self.angular_position_embedding, std=0.02)

    def forward(self, range_image: Tensor) -> Tensor:
        """Encode a batch of range images into obstacle tokens.

        Args:
            range_image: `[B, 1, H, W]`, binary (1.0 free / 0.0
                blocked), as produced by `RangeImageBuilder`.

        Returns:
            `[B, num_obstacle_tokens, token_embedding_dim]` obstacle
            tokens, each already carrying its angular positional
            encoding.

        Raises:
            ValueError: If `range_image`'s shape doesn't match
                `config.in_height`/`config.in_width`.
        """
        if range_image.dim() != 4 or range_image.shape[1] != 1:
            raise ValueError(
                f"range_image must be [B, 1, H, W], got shape "
                f"{tuple(range_image.shape)}."
            )
        if range_image.shape[2] != self.config.in_height:
            raise ValueError(
                f"range_image height {range_image.shape[2]} does not match "
                f"config.in_height={self.config.in_height}."
            )
        if range_image.shape[3] != self.config.in_width:
            raise ValueError(
                f"range_image width {range_image.shape[3]} does not match "
                f"config.in_width={self.config.in_width}."
            )

        x = self.stem(range_image)
        for stage in self.stages:
            x = stage(x)

        x = self.height_collapse(x)  # [B, D, 1, backbone_output_width]
        x = self.token_compress(x)  # [B, D, 1, num_obstacle_tokens]

        tokens = x.squeeze(2).transpose(1, 2)  # [B, num_obstacle_tokens, D]
        tokens = self.token_norm(tokens)
        tokens = tokens + self.angular_position_embedding.unsqueeze(0)
        return tokens

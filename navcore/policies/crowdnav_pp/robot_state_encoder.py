"""Robot state encoder for CrowdNav++'s recurrent interaction-graph policy.

Ported from the ``robot_linear`` step in ``selfAttn_merge_SRNN.__init__``/
``forward`` (rl/networks/selfAttn_srnn_temp_node.py of
github.com/Shuijing725/CrowdNav_Prediction_AttnGraph). See
``attention.py``'s module docstring for the general porting rationale.

Input width is a real fork, not an implementation detail:
    The original concatenates ``temporal_edges`` (robot's own vx, vy) and
    ``robot_node`` (absolute px, py, radius, gx, gy, v_pref, theta) into
    a 9-dim vector before this layer. navcore's
    ``ObservationEncoder._ROBOT_FEATURES`` is already a different,
    deliberately relative-frame 8-dim encoding (goal displacement, vx,
    vy, v_pref, radius, cos(theta), sin(theta)) -- see that module's own
    docstring for why relative-frame was chosen. This encoder therefore
    takes ``robot_feature_dim`` as an explicit constructor parameter
    (8 for navcore's current ``ObservationEncoder`` output) rather than
    hardcoding the original's 9, so it is correct for navcore's actual
    observation, not the paper's.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(slots=True, frozen=True)
class RobotStateEncoderConfig:
    """Hyperparameters for :class:`RobotStateEncoder`.

    Attributes:
        robot_feature_dim: Width of the incoming robot feature vector.
            8 for navcore's current ``ObservationEncoder`` (see module
            docstring for why this differs from the original paper's 9).
        embedding_dim: Width of the projected embedding. Must match
            whatever ``embedding_dim`` the rest of the network (attention
            layers, recurrent node update) is configured with, since this
            embedding is later concatenated/compared against
            human-derived embeddings of that same width.
    """

    robot_feature_dim: int
    embedding_dim: int

    def __post_init__(self) -> None:
        if self.robot_feature_dim <= 0:
            raise ValueError(
                f"robot_feature_dim must be positive, got {self.robot_feature_dim!r}."
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim!r}."
            )


class RobotStateEncoder(nn.Module):
    """Projects the robot's own feature vector into the network's embedding space.

    A single ``Linear`` + ``ReLU`` -- the original's ``robot_linear`` is
    exactly this, no more. Kept as its own small module (rather than
    inlined into a larger network class) so the input-width decision
    above is visible and independently testable, per the project's
    "small classes, explicit forks" conventions.
    """

    def __init__(self, config: RobotStateEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Sequential(
            nn.Linear(config.robot_feature_dim, config.embedding_dim),
            nn.ReLU(),
        )

    def forward(self, robot_features: Tensor) -> Tensor:
        """Return the robot's projected embedding.

        Args:
            robot_features: ``[..., robot_feature_dim]`` -- any leading
                shape is preserved; only the last dimension is consumed.
                Typically ``[seq_len, nenv, robot_feature_dim]`` to match
                ``HumanHumanAttention``/``RobotHumanAttention``'s
                ``[seq_len, nenv, ...]`` convention (the caller adds the
                singleton "1 robot" axis ``RobotHumanAttention`` expects
                via ``.unsqueeze(-2)``, since that axis is specific to
                that layer's contract, not to encoding itself).

        Returns:
            ``[..., embedding_dim]``, same leading shape as the input.

        Raises:
            ValueError: If the input's last dimension doesn't match
                ``config.robot_feature_dim``.
        """
        if robot_features.shape[-1] != self.config.robot_feature_dim:
            raise ValueError(
                f"robot_features' last dim is {robot_features.shape[-1]}, "
                f"but this encoder was configured for robot_feature_dim="
                f"{self.config.robot_feature_dim}."
            )
        return self.project(robot_features)

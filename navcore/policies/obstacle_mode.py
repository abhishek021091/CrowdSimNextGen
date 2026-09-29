"""Single definition of ObstacleMode, shared by every policy implementation.

Two separate ``ObstacleMode`` enums used to exist (crowdnav_pp and original/).
The trainer compares ``policy.config.obstacle_mode is ObstacleMode.X`` against
one of them, so with the other policy every ``is`` comparison silently failed.
"""

from enum import Enum


class ObstacleMode(Enum):
    """How static obstacles (ray-cast hits) reach the policy."""

    NONE = "none"
    ENCODER = "encoder"
    POINT_TOKENS = "point_tokens"

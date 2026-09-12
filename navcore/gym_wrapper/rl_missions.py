"""RL-specific missions and the baseline goal-reaching task.

Reward is dense progress-to-goal shaping (distance closed this tick)
plus a per-tick time penalty, a one-time goal bonus, and a collision
penalty. Progress shaping avoids the classic sparse-reward problem of
"only reward reaching the goal" -- with ORCA and pedestrians in the
mix, an untrained policy essentially never reaches the goal early in
training, so a sparse signal gives no gradient at all.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState
from navcore.entities.environment.environment import Environment
from navcore.missions.goal_reaching import GoalReachingMission

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


class RLWaypointMission:
    """Mission whose target is supplied by an RL waypoint action.

    ``CrowdSimEnv`` owns this mission in waypoint action mode.  Each
    decoded action replaces the target via :meth:`set_target`; the target is
    then consumed by ``Step`` for that tick's ORCA preferred velocity.  The
    robot's persistent goal is intentionally not changed, so task success
    and reward calculations continue to use its semantic destination.
    """

    def __init__(self, initial_target: Vector2) -> None:
        self._target = initial_target

    def set_target(self, target: Vector2) -> None:
        """Set the world-coordinate waypoint for the next planning tick."""
        self._target = target

    def get_target(self, agent: Agent, neighbors: Sequence[ObservableState]) -> Vector2:
        """Return the current waypoint.

        ``agent`` and ``neighbors`` are accepted to satisfy the common
        :class:`Mission` protocol.  The action-selected waypoint is already
        the complete target, so neither is consulted here.
        """
        return self._target

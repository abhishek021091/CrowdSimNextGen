"""GoalReachingMission: the baseline "go to my goal" mission.

Used by plain (non-group) pedestrians, and by robots running in
goal-reach mode (as opposed to sweep mode). Deliberately has no
knowledge of collision avoidance, global planning, or coverage --
avoidance for this mission is entirely the responsibility of whatever
``Policy`` the agent uses (ORCA, RL, ...). See the project's mission
architecture discussion for why this split exists: bundling avoidance
into every ``Mission`` would duplicate logic that ``Policy`` already
owns, and would give agents with no avoidance-capable ``Policy`` a false
sense of safety.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


class GoalReachingMission:
    """Always targets the agent's own ``goal``.

    Raises:
        RuntimeError: From :meth:`get_target`, if the agent's ``goal``
            has not been set yet.
    """

    def get_target(
        self, agent: Agent, neighbors: Sequence[ObservableState]
    ) -> Vector2:
        """Return the agent's goal position, ignoring ``neighbors``."""
        if agent.goal is None:
            raise RuntimeError(
                "GoalReachingMission requires agent.goal to be set before "
                "get_target() is called."
            )
        return Vector2(agent.goal.gx, agent.goal.gy)

"""StraightLinePolicy: a wiring placeholder, not a research policy.

Moves directly toward its target at ``agent.v_pref``, completely
ignoring ``neighbors``. It exists only so the ``Mission`` /
``VisibilityPolicy`` / ``NeighborQuery`` plumbing can be exercised
end-to-end before ORCA, PID, and RL policies are implemented.

Do not use this for experiments or benchmarking -- it performs no
collision avoidance whatsoever. It is a test double, not a baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState
from navcore.entities.components.velocity import Velocity

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


class StraightLinePolicy:
    """Heads straight for the target at the agent's preferred speed."""

    def compute_velocity(
        self,
        agent: Agent,
        target: Vector2,
        neighbors: Sequence[ObservableState],
    ) -> Velocity:
        """Return a velocity pointed at ``target``, ignoring ``neighbors``.

        Returns ``Velocity(0.0, 0.0)`` if the agent is already
        (numerically) at ``target``, to avoid normalizing a zero vector.
        """
        if agent.pose is None:
            raise RuntimeError("agent.pose must be set before computing velocity.")

        position = Vector2(agent.pose.px, agent.pose.py)
        offset = target - position
        if offset.magnitude() == 0.0:
            return Velocity(0.0, 0.0)

        direction = offset.normalize()
        heading = direction * agent.v_pref
        return Velocity(heading.x, heading.y)

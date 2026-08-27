"""GlobalPlanner: "where should the robot go next to finish sweeping".

Deliberately excluded from this interface: anything about pedestrians,
collisions, or handoff to avoidance. That split is ``SweepMission``'s
job (added separately) -- a ``GlobalPlanner`` only ever answers "given
what's already swept, what's the next coverage waypoint," and is called
at ``SweepMission``'s ~2m cadence, not every tick.

Three implementations exist side by side on purpose, as a research
comparison axis, not because one obsoletes the others:

    - ``EuclideanNearestPlanner`` -- cheapest, purely distance-based,
      ignores obstacles between robot and target.
    - ``GraphSearchPlanner`` -- BFS over the free-cell grid graph,
      respects connectivity, more expensive per call.
    - ``BoustrophedonPlanner`` -- a fixed lawnmower sweep pattern,
      stateful, not a "nearest" search at all.

All three return ``None`` to signal "no unswept free cells remain" --
callers (``SweepMission``) treat that as sweep-complete.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.planning.coverage_grid import CoverageGrid


@runtime_checkable
class GlobalPlanner(Protocol):
    """Produces the next coverage waypoint, given the current coverage state."""

    def next_waypoint(
        self, current_position: Vector2, grid: CoverageGrid
    ) -> Vector2 | None:
        """Return the next point the robot should sweep toward.

        Args:
            current_position: The robot's current world position.
            grid: The shared coverage state. Implementations must treat
                this as read-only -- marking cells swept is not this
                interface's responsibility (see ``SweepMission``).

        Returns:
            The next waypoint in world coordinates, or ``None`` if every
            free cell has already been swept.
        """
        ...

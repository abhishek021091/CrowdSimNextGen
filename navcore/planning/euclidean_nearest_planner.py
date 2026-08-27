"""EuclideanNearestPlanner: pick the closest unswept free cell by straight-line distance.

The cheapest of the three strategies -- O(unswept free cells) per call,
no graph traversal. Deliberately does not check whether a straight line
to the chosen cell actually crosses an obstacle; that's the point of
keeping this as a distinct, named strategy rather than quietly folding
obstacle-awareness into it -- it's a useful baseline precisely because
it can fail that way, and comparing it against ``GraphSearchPlanner``
is expected to be part of the research question, not a bug to hide.
"""

from __future__ import annotations

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.planning.coverage_grid import CoverageGrid


class EuclideanNearestPlanner:
    """Greedy nearest-unswept-cell selection, ignoring reachability."""

    def next_waypoint(
        self, current_position: Vector2, grid: CoverageGrid
    ) -> Vector2 | None:
        """Return the unswept free cell closest to ``current_position``.

        Returns:
            The nearest unswept free cell's world position, or ``None``
            if no unswept free cells remain.
        """
        best_position: Vector2 | None = None
        best_distance = float("inf")

        for row, col in grid.unswept_free_cells():
            candidate = grid.cell_to_world(row, col)
            distance = current_position.distance_to(candidate)
            if distance < best_distance:
                best_distance = distance
                best_position = candidate

        return best_position

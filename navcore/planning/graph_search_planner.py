"""GraphSearchPlanner: BFS over the free-cell grid graph for geodesic-nearest coverage.

Unlike ``EuclideanNearestPlanner``, this treats free cells as a graph
(4-connected by default) and finds the unswept free cell that's
cheapest to *actually reach* without crossing an obstacle. BFS on an
unweighted grid graph gives shortest-path-in-cell-count, which is a
reasonable proxy for geodesic distance at this cell resolution.

Design note -- why this returns a point partway along the path, not the
target cell itself:
    Jumping straight to the final unswept cell would mean the robot
    (and its local ``Policy``) has to somehow already know to route
    around every obstacle between here and there in one step -- exactly
    the failure mode this planner exists to avoid. Instead, it walks
    the BFS path and returns the point ``lookahead_distance`` along it
    (2m by default, per the project's sweep cadence), so each waypoint
    is a short, locally-reachable step. ``SweepMission`` calls this
    again once that step is reached, extending the path incrementally.

Performance note: BFS is O(free cells) worst case per call. This is
paid at the ~2m waypoint cadence, not every tick, so it's acceptable
even for a moderately large grid; if arenas grow much larger, the first
optimization is caching/reusing the previous BFS tree rather than
rebuilding from scratch each call.
"""

from __future__ import annotations

from collections import deque

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.planning.coverage_grid import CoverageGrid

#: 4-connected grid steps: up, down, left, right.
_FOUR_CONNECTED = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: 8-connected grid steps: the above plus the four diagonals.
_EIGHT_CONNECTED = _FOUR_CONNECTED + ((-1, -1), (-1, 1), (1, -1), (1, 1))


class GraphSearchPlanner:
    """BFS-based nearest-reachable-unswept-cell selection.

    Attributes:
        lookahead_distance: How far along the shortest path to project
            the returned waypoint, in world units.
        eight_connected: If ``True``, diagonal moves are allowed.
            Defaults to ``False`` (4-connected), which is cheaper and
            avoids "corner-cutting" past a diagonally-adjacent blocked
            cell.
    """

    def __init__(
        self, lookahead_distance: float = 2.0, eight_connected: bool = False
    ) -> None:
        if lookahead_distance <= 0.0:
            raise ValueError(
                f"lookahead_distance must be positive, got {lookahead_distance!r}."
            )
        self.lookahead_distance = lookahead_distance
        self._steps = _EIGHT_CONNECTED if eight_connected else _FOUR_CONNECTED

    def next_waypoint(
        self, current_position: Vector2, grid: CoverageGrid
    ) -> Vector2 | None:
        """Return a point ``lookahead_distance`` along the path to the nearest unswept cell.

        Returns:
            A waypoint along the shortest free-space path toward the
            nearest reachable unswept cell, or ``None`` if no unswept
            free cell is reachable from ``current_position``.
        """
        start = grid.world_to_cell(current_position)
        path = self._shortest_path_to_nearest_unswept(start, grid)
        if path is None:
            return None
        return self._point_along_path(path, grid)

    def _shortest_path_to_nearest_unswept(
        self, start: tuple[int, int], grid: CoverageGrid
    ) -> list[tuple[int, int]] | None:
        """Return the cell path from ``start`` to the nearest unswept free cell."""
        if not grid.is_free(*start):
            return None

        visited = {start}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        queue: deque[tuple[int, int]] = deque([start])

        while queue:
            current = queue.popleft()
            row, col = current

            if grid.is_free(row, col) and not grid.is_swept(row, col) and current != start:
                return self._reconstruct_path(start, current, parents)

            for d_row, d_col in self._steps:
                neighbor = (row + d_row, col + d_col)
                if neighbor in visited:
                    continue
                if not grid.in_bounds(*neighbor):
                    continue
                if not grid.is_free(*neighbor):
                    continue
                visited.add(neighbor)
                parents[neighbor] = current
                queue.append(neighbor)

        return None

    @staticmethod
    def _reconstruct_path(
        start: tuple[int, int],
        target: tuple[int, int],
        parents: dict[tuple[int, int], tuple[int, int]],
    ) -> list[tuple[int, int]]:
        path = [target]
        while path[-1] != start:
            path.append(parents[path[-1]])
        path.reverse()
        return path

    def _point_along_path(
        self, path: list[tuple[int, int]], grid: CoverageGrid
    ) -> Vector2:
        """Walk ``path`` and return the point at ``lookahead_distance``, or the end."""
        traveled = 0.0
        previous = grid.cell_to_world(*path[0])
        for row, col in path[1:]:
            current = grid.cell_to_world(row, col)
            traveled += previous.distance_to(current)
            if traveled >= self.lookahead_distance:
                return current
            previous = current
        return previous

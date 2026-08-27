"""BoustrophedonPlanner: a fixed straight-line lawnmower sweep, with extended goals.

Unlike ``EuclideanNearestPlanner`` and ``GraphSearchPlanner``, this
isn't a "nearest cell" search -- it's a deterministic sweep pattern.
The robot sweeps one grid row in a straight line until it hits an
obstacle or the arena boundary (the "extended goal": the farthest free,
unswept cell reachable in the current direction, not a fixed short
step), then steps to the next sweep line and reverses direction, the
same way a lawnmower covers a lawn.

Design note -- why this needs internal state:
    ``EuclideanNearestPlanner`` and ``GraphSearchPlanner`` are stateless
    per call -- they only need the current grid snapshot. Boustrophedon
    fundamentally isn't: "which line am I on, which direction am I
    going" is the entire strategy, not derivable from the grid alone.
    That state lives on this instance, not on ``CoverageGrid`` or
    ``SweepMission`` -- one ``BoustrophedonPlanner`` per robot per
    episode, reset (a fresh instance) at episode start.

Design note -- this is a *simple* sweep, not a full coverage-planning
algorithm:
    If an obstacle sits mid-row, this stops the current line at the
    obstacle and does not resume sweeping the remainder of that row
    from the far side -- that's what proper boustrophedon *cell
    decomposition* (splitting free space at obstacle silhouettes before
    sweeping) is for, and is a legitimate, more thorough alternative
    still on the roadmap as a separate ``GlobalPlanner`` implementation.
    This class is deliberately the cheap, naive baseline for comparison
    against it.

Important dependency: like the other two planners, this only makes
forward progress if something external -- ``SweepMission``'s tick loop,
not yet built -- marks cells swept as the robot physically travels
through them, not just at the final waypoint. Without that, this
planner will keep returning the same "farthest" point indefinitely,
since it never sees the ground it already covered.
"""

from __future__ import annotations

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.planning.coverage_grid import CoverageGrid


class BoustrophedonPlanner:
    """Deterministic row-by-row lawnmower sweep with alternating direction.

    Attributes:
        row_step: Number of grid rows to advance between sweep lines.
            Defaults to ``1`` (every row swept). Set higher to match a
            robot with a wider effective coverage/sensor footprint than
            one grid cell, so lines don't unnecessarily overlap.
    """

    def __init__(self, row_step: int = 1) -> None:
        if row_step < 1:
            raise ValueError(f"row_step must be at least 1, got {row_step!r}.")
        self.row_step = row_step
        self._current_row: int | None = None
        self._direction: int = 1  # +1: sweeping toward increasing column

    def next_waypoint(
        self, current_position: Vector2, grid: CoverageGrid
    ) -> Vector2 | None:
        """Return the farthest reachable point along the current sweep line.

        On the first call, the sweep line is initialized to the row
        containing ``current_position``. Subsequent calls continue from
        wherever the sweep last left off, advancing lines and reversing
        direction as each line is exhausted.

        Returns:
            The farthest free, unswept cell reachable in the current
            sweep direction, or ``None`` once every sweep line has been
            exhausted (the whole arena is covered or blocked).
        """
        row, col = grid.world_to_cell(current_position)
        if self._current_row is None:
            self._current_row = row

        while grid.in_bounds(self._current_row, 0):
            target_col = self._farthest_reachable_column(grid, self._current_row, col)
            if target_col is not None:
                return grid.cell_to_world(self._current_row, target_col)

            self._current_row += self.row_step
            self._direction *= -1
            col = 0 if self._direction == 1 else grid.n_cols - 1

        return None

    def _farthest_reachable_column(
        self, grid: CoverageGrid, row: int, start_col: int
    ) -> int | None:
        """Return the farthest free, unswept column reachable from ``start_col``.

        Scans from ``start_col`` toward the current sweep direction,
        stopping at the first blocked cell or the grid edge. Returns the
        farthest column encountered that is still unswept, or ``None``
        if nothing unswept is reachable before the stop point.
        """
        if not grid.in_bounds(row, start_col):
            return None

        columns = (
            range(start_col, grid.n_cols)
            if self._direction == 1
            else range(start_col, -1, -1)
        )

        farthest: int | None = None
        for col in columns:
            if not grid.is_free(row, col):
                break
            if not grid.is_swept(row, col):
                farthest = col

        return farthest

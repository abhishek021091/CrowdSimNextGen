"""CoverageGrid: the shared spatial record of "what's blocked" and "what's swept".

Built once per episode from the world's *static* obstacles. Two
independent things read it downstream, and neither owns it:

    - Classical ``GlobalPlanner`` implementations (occupancy-grid
      nearest-unswept-cell, and later cell-decomposition) query it for
      waypoint selection.
    - An RL observation adapter (added separately) crops an egocentric
      window around the robot's pose and hands it to a policy network
      as a tensor -- the grid's array-of-cells shape is exactly what
      makes it usable for both classical search *and* as a CNN input,
      without maintaining two separate representations.

Design note -- why this knows nothing about pedestrians:
    Coverage is a property of *static* free space only. Pedestrians are
    handled dynamically, tick by tick, by ``SweepMission``'s
    ``SWEEPING``/``AVOIDING`` state machine (added separately) --
    baking pedestrian positions into this grid would mean rebuilding
    (or partially invalidating) it every tick, which is both wasteful
    and would blur a boundary that's cheap to keep clean: this class
    answers "where is coverage still needed", nothing else.

Design note -- why ``occupied`` and ``swept`` are separate arrays:
    ``occupied`` is fixed at construction (static obstacles don't
    move mid-episode). ``swept`` mutates every tick as the robot
    covers new ground. Keeping them separate means resetting an episode
    only requires re-zeroing ``swept`` -- ``occupied`` is reusable
    as-is if the obstacle layout hasn't changed.

Performance note: rasterization is O(rows * cols * obstacles), paid
once at construction. All per-tick operations (``mark_swept``,
``is_free``, cell/world conversion) are O(1) array lookups.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np

from navcore.entities.components.geometry.containment import point_in_geometry
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.obstacles.obstacle import Obstacle


class CoverageGrid:
    """A uniform grid over a rectangular arena, tracking blocked vs. swept cells.

    The arena is assumed centered on the world origin, spanning
    ``[-width/2, width/2] x [-height/2, height/2]`` -- consistent with
    how ``env.toml``'s ``arenaSize`` and the existing ``Circle``/
    ``Rectangle`` bounds checks already treat the arena.

    Attributes:
        width: Arena width, in world units.
        height: Arena height, in world units.
        cell_size: Side length of each square cell, in world units.
        n_rows: Number of grid rows (along y).
        n_cols: Number of grid columns (along x).
    """

    def __init__(
        self,
        width: float,
        height: float,
        cell_size: float,
        obstacles: Sequence[Obstacle],
    ) -> None:
        """Build the grid and rasterize ``obstacles`` into it.

        Args:
            width: Arena width, in world units. Must be positive.
            height: Arena height, in world units. Must be positive.
            cell_size: Side length of each square cell. Must be
                positive and should evenly divide (or nearly divide)
                ``width``/``height`` for a clean fit; the grid rounds
                up if it doesn't.
            obstacles: Static obstacles to rasterize as blocked cells.
                Obstacles with ``traversable=True`` are skipped -- the
                robot (and coverage accounting) may pass through them,
                so they should not block sweep waypoints.

        Raises:
            ValueError: If ``width``, ``height``, or ``cell_size`` is
                not strictly positive.
        """
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"width and height must be positive, got width={width!r}, "
                f"height={height!r}."
            )
        if cell_size <= 0.0:
            raise ValueError(f"cell_size must be positive, got {cell_size!r}.")

        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.n_cols = math.ceil(width / cell_size)
        self.n_rows = math.ceil(height / cell_size)

        self._origin_x = -width / 2.0
        self._origin_y = -height / 2.0

        self.occupied = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        self.swept = np.zeros((self.n_rows, self.n_cols), dtype=bool)

        self._rasterize(obstacles)

    def _rasterize(self, obstacles: Sequence[Obstacle]) -> None:
        """Mark ``occupied`` cells whose center falls inside a blocking obstacle."""
        blocking = [o for o in obstacles if not o.traversable]
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                center = self.cell_to_world(row, col)
                for obstacle in blocking:
                    if point_in_geometry(center, obstacle.geometry):
                        self.occupied[row, col] = True
                        break

    # -- coordinate conversion -----------------------------------------------

    def world_to_cell(self, point: Vector2) -> tuple[int, int]:
        """Return the ``(row, col)`` of the cell containing ``point``.

        The result is clamped to the grid's bounds, so points slightly
        outside the arena (e.g. numerical overshoot at a boundary) still
        resolve to the nearest edge cell rather than raising.
        """
        col = int((point.x - self._origin_x) / self.cell_size)
        row = int((point.y - self._origin_y) / self.cell_size)
        col = max(0, min(col, self.n_cols - 1))
        row = max(0, min(row, self.n_rows - 1))
        return row, col

    def cell_to_world(self, row: int, col: int) -> Vector2:
        """Return the world-coordinate center of cell ``(row, col)``."""
        x = self._origin_x + (col + 0.5) * self.cell_size
        y = self._origin_y + (row + 0.5) * self.cell_size
        return Vector2(x, y)

    def in_bounds(self, row: int, col: int) -> bool:
        """Return whether ``(row, col)`` is a valid cell index."""
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    # -- queries ---------------------------------------------------------------

    def is_free(self, row: int, col: int) -> bool:
        """Return whether cell ``(row, col)`` is not blocked by a static obstacle."""
        return not self.occupied[row, col]

    def is_swept(self, row: int, col: int) -> bool:
        """Return whether cell ``(row, col)`` has been marked swept."""
        return bool(self.swept[row, col])

    def mark_swept(self, point: Vector2) -> None:
        """Mark the cell containing ``point`` as swept.

        Marking a blocked cell as swept is harmless (it's never counted
        in coverage statistics or offered as a waypoint) but is not
        expected to happen in normal operation.
        """
        row, col = self.world_to_cell(point)
        self.swept[row, col] = True

    def unswept_free_cells(self) -> Iterator[tuple[int, int]]:
        """Yield ``(row, col)`` for every free cell not yet swept.

        Used by ``GlobalPlanner`` implementations to search for the next
        coverage waypoint. Order is row-major and carries no priority --
        callers that care about proximity should search, not rely on
        iteration order.
        """
        free_unswept = (~self.occupied) & (~self.swept)
        rows, cols = np.nonzero(free_unswept)
        for row, col in zip(rows.tolist(), cols.tolist()):
            yield row, col

    def free_cell_count(self) -> int:
        """Return the total number of unblocked cells in the grid."""
        return int(np.count_nonzero(~self.occupied))

    def swept_free_cell_count(self) -> int:
        """Return the number of unblocked cells that have been swept."""
        return int(np.count_nonzero(self.swept & ~self.occupied))

    def coverage_fraction(self) -> float:
        """Return swept-free-cells / total-free-cells, in ``[0.0, 1.0]``.

        Returns:
            ``1.0`` if the arena has no free cells at all (vacuously
            fully covered), to avoid a division by zero. This is a
            metrics primitive -- ``MetricsCollector`` (added separately)
            is expected to sample this each tick or at episode end.
        """
        total = self.free_cell_count()
        if total == 0:
            return 1.0
        return self.swept_free_cell_count() / total

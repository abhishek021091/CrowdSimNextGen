"""SweepMission: coverage-sweeping mission for the robot.

Wraps a GlobalPlanner + CoverageGrid behind the Mission protocol so
Step/Policy can drive a sweeping robot exactly like any other mission
(see mission.py for why Mission only ever returns a point, never *how*).

Design note -- why this marks cells swept itself, on every get_target():
    Nothing else in the tick loop knows "sweeping" is happening. If
    cell-marking lived in Step, every caller would need to remember to
    call it at the right moment; doing it here keeps that invariant
    local to the one class that owns it.

Design note -- why replanning is arrival-driven, not per-tick:
    GlobalPlanner implementations are meant to be called at a coarse
    (~2m) cadence, not every tick (see global_planner.py). This mission
    keeps returning the same waypoint until the agent is within
    `arrival_tolerance` of it, then asks the planner for the next one
    exactly once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState
from navcore.planning.boustrophedon_planner import BoustrophedonPlanner
from navcore.planning.coverage_grid import CoverageGrid
from navcore.planning.global_planner import GlobalPlanner

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


class SweepMission:
    """Coverage-sweep mission: visit every free cell of a CoverageGrid.

    Falls back to the agent's own ``goal`` once the grid reports full
    coverage, so Step's goal-reached bookkeeping keeps working after
    the sweep finishes.

    Attributes:
        grid: The shared coverage record this mission sweeps and marks.
        planner: Waypoint strategy. Defaults to ``BoustrophedonPlanner``
            (deterministic, no per-call graph search -- swap in
            ``EuclideanNearestPlanner``/``GraphSearchPlanner`` for the
            other two research conditions without touching this class).
        arrival_tolerance: Distance, in world units, within which the
            current waypoint counts as reached. Should be >=
            ``grid.cell_size / 2`` so a target at a cell center is
            always numerically reachable.
    """

    def __init__(
        self,
        grid: CoverageGrid,
        planner: GlobalPlanner | None = None,
        arrival_tolerance: float = 0.3,
    ) -> None:
        self.grid = grid
        self.planner = planner if planner is not None else BoustrophedonPlanner()
        self.arrival_tolerance = arrival_tolerance
        self._current_target: Vector2 | None = None
        self.finished = False

    def get_target(self, agent: Agent, neighbors: Sequence[ObservableState]) -> Vector2:
        """Return the next sweep waypoint, marking coverage as we go.

        Args:
            agent: Must have ``agent.pose`` set. ``neighbors`` is
                unused -- local avoidance is the ``Policy``'s job, not
                this mission's (see module docstring pattern in
                ``goal_reaching.py``).

        Raises:
            RuntimeError: If ``agent.pose`` has not been set yet.
        """
        if agent.pose is None:
            raise RuntimeError("SweepMission requires agent.pose to be set.")

        position = Vector2(agent.pose.px, agent.pose.py)
        self.grid.mark_swept(position)

        reached = (
            self._current_target is not None
            and position.distance_to(self._current_target) <= self.arrival_tolerance
        )
        if self._current_target is None or reached:
            self._current_target = self.planner.next_waypoint(position, self.grid)

        if self._current_target is None:
            self.finished = True
            if agent.goal is not None:
                return Vector2(agent.goal.gx, agent.goal.gy)
            return position

        return self._current_target

    def coverage_fraction(self) -> float:
        """Return the underlying grid's swept/free fraction, for metrics."""
        return self.grid.coverage_fraction()

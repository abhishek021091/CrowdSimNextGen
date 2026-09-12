"""SweepingMission: lawnmower-style coverage of one convex cell.

Redesigned from whole-arena coverage to per-cell coverage, to consume
DecompositionResult/TraversalStep output (see environment_decomposition.py
and graph_traversal.py): the BCD+traversal pipeline decides *which* convex
cell to sweep, and this class's only job is to actually sweep that one
cell. It has no notion of "the arena" anymore -- env is used only for the
robot's radius/pose/goal and for avoid_crowd's sensor access, never for
arena_width/arena_height.

Why there is no stored entry point:
    An earlier version of this class took an explicit entry_point,
    computed once by the traversal planner as the midpoint of the edge
    shared with the previous cell. That's wrong for an open area with a
    crowd in it: nothing guarantees the robot's actual position, once it
    finishes transiting into the cell, matches that planned point --
    ORCA and any avoid_crowd detour along the way can land it somewhere
    else on the cell's boundary (or well inside it) instead.
    reach_closest_corner() therefore reads env.robot.pose live, at the
    moment sweeping actually starts, and picks the nearest corner/lane
    from wherever the robot really is. The traversal planner's own
    entry_point/exit_point bookkeeping (graph_traversal.py) is
    unaffected by this -- it still exists for transit-phase goal-setting
    between cells, this class just no longer consumes it.

Sweep direction is computed as the cell's minimal-width direction. For a
convex polygon, the minimum-width supporting direction is always
perpendicular to one of its edges (the same standard result
environment_decomposition._polygon_width relies on via
minimum_rotated_rectangle); trying every edge as a candidate and keeping
the one with the smallest perpendicular extent finds it exactly, using
only navcore's own Vector2 math -- no shapely dependency needed here,
keeping shapely quarantined to environment_decomposition.py as
originally scoped.

Per-lane clipping, not a fixed rectangle:
    A rectangle's sweep bounds are the same on every lane. An arbitrary
    convex cell's are not -- a triangular or pentagonal cell's width at
    one end of its length can differ from its width at the other. Every
    lane's [x_min, x_max] is therefore computed on demand
    (_lane_bounds) by intersecting a horizontal line, in the cell's own
    rotated local frame, against the cell's actual edges -- not assumed
    constant across the whole sweep the way a rectangle-arena version
    could assume.
</code_to_edit>
Area accounting caveat:
    total_area_swept() sums each completed lane's actual clipped width
    times lane spacing. This is still an approximation for
    non-rectangular cells (a lane's swept footprint is a trapezoid-ish
    strip, not exactly width * lane_step, wherever the cell's boundary
    isn't parallel to the sweep direction).

avoid_crowd() is unchanged from the whole-arena version: it manipulates
env.robot.goal / observes the robot's sensor and never referenced arena
bounds or entry_point in the first place.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.environment.environment import Environment
from navcore.graph_traversal.traversal import TraversalStep

_EPS = 1e-9


@dataclass
class GetData:
    pose_before_avoidance: Pose
    goal_before_avoidance: Goal


class SweepingMission:
    """Lawnmower-style coverage mission for one convex cell.

    On ``reach_closest_corner()``, drives the robot to whichever corner
    of the cell is closest to its *current* pose, then sweeps back and
    forth in lanes perpendicular to the cell's minimal-width direction
    until the far side of the cell is reached.

    All mission progress (``started``, ``sweeping``, ``collisions``,
    ``area_swept``, ``avoiding_obstacle``, ``sweep_finished``) lives on
    the instance -- construct a fresh ``SweepingMission`` per cell rather
    than reusing one across cells (see ``for_traversal_step``).
    """

    def __init__(
        self,
        env: Environment,
        cell_vertices: tuple[Vector2, ...],
        robot_sweep_step: float = 5,
        robot_sweep_lane_step: float | None = None,
        robot_sweep_margin: float | None = None,
    ):
        if len(cell_vertices) < 3:
            raise ValueError(
                "SweepingMission requires a polygon of at least 3 vertices, "
                f"got {len(cell_vertices)}."
            )

        self.env = env
        self.cell_vertices = cell_vertices
        self.robot_sweep_margin = (
            robot_sweep_margin
            if robot_sweep_margin is not None
            else self.env.robot.radius
        )
        self.robot_sweep_step = robot_sweep_step
        # Default lane spacing: one robot diameter, so consecutive lanes
        # don't overlap or leave gaps.
        self.robot_sweep_lane_step = (
            robot_sweep_lane_step
            if robot_sweep_lane_step is not None
            else self.env.robot.radius * 2
        )

        self.sweep_direction, self.lane_direction = self._minimal_width_axes(
            cell_vertices
        )

        self.sweep_dir: int = 1
        self.sweep_finished: bool = False

        self.started: bool = False
        self.sweeping: bool = False
        self.collisions: int = 0
        self.area_swept: float = 0.0
        self.avoiding_obstacle: bool = False
        self.current_safe_point: tuple[float, float] | None = None

        self._lane_local_y: float = 0.0
        self._lane_x_min: float = 0.0
        self._lane_x_max: float = 0.0
        self._lane_start_local_x: float = 0.0
        self._shift_positive: bool = True
        self._completed_lane_length_sum: float = 0.0

    @classmethod
    def for_traversal_step(
        cls, step: TraversalStep, env: Environment, **kwargs
    ) -> SweepingMission:
        """Build a mission that sweeps ``step``'s cell.

        Convenience constructor tying this class directly to the
        decomposition/traversal pipeline's output -- callers driving a
        full-coverage run should construct one ``SweepingMission`` per
        ``TraversalStep`` with ``requires_sweep=True`` (transit-only
        steps don't need a mission at all). ``step.entry_point`` is
        deliberately not passed through -- see module docstring for why
        this class reads the robot's live pose instead.
        """
        return cls(env=env, cell_vertices=step.cell.vertices, **kwargs)

    # -- local <-> world frame -------------------------------------------

    def _to_local(self, point: Vector2) -> tuple[float, float]:
        """Project a world-frame point into (along-sweep, along-lane) coordinates.

        This is a pure rotation about the world origin (dot products
        against unit basis vectors), not a rotation-plus-translation --
        correct here because ``sweep_direction``/``lane_direction`` are
        an orthonormal basis and every point involved (cell vertices,
        goal, pose) is already expressed in the same world frame.
        """
        return point.dot(self.sweep_direction), point.dot(self.lane_direction)

    def _to_world(self, local_x: float, local_y: float) -> Vector2:
        """Inverse of ``_to_local``."""
        return self.sweep_direction * local_x + self.lane_direction * local_y

    @staticmethod
    def _minimal_width_axes(vertices: tuple[Vector2, ...]) -> tuple[Vector2, Vector2]:
        """Return ``(sweep_direction, lane_direction)``, an orthonormal basis
        aligned with the cell's minimal-width orientation.

        Tries every edge as a candidate sweep direction and keeps
        whichever gives the smallest perpendicular extent -- see module
        docstring for why this is exact, not a heuristic, for convex
        polygons.
        """
        n = len(vertices)
        best_width = float("inf")
        best_direction = Vector2(1.0, 0.0)
        best_perp = Vector2(0.0, 1.0)

        for i in range(n):
            edge = vertices[(i + 1) % n] - vertices[i]
            if edge.magnitude() < _EPS:
                continue
            direction = edge.normalize()
            perp = Vector2(-direction.y, direction.x)

            projections = [v.dot(perp) for v in vertices]
            width = max(projections) - min(projections)
            if width < best_width:
                best_width = width
                best_direction = direction
                best_perp = perp

        return best_direction, best_perp

    # -- per-lane geometry -------------------------------------------------

    def _local_vertices(self) -> list[tuple[float, float]]:
        return [self._to_local(v) for v in self.cell_vertices]

    def _y_extent(self) -> tuple[float, float]:
        """Return the cell's full (min, max) extent along the lane axis."""
        y_values = [ly for _, ly in self._local_vertices()]
        return min(y_values), max(y_values)

    def _lane_bounds(self, local_y: float) -> tuple[float, float] | None:
        """Return the margin-shrunk ``(x_min, x_max)`` where the cell's
        boundary crosses ``local_y``, or ``None`` if the cell doesn't
        reach ``local_y`` (after margin shrink) at all -- e.g. a lane
        placed right at a triangular cell's narrowing tip.

        Convex-polygon scanline intersection: for each edge, if
        ``local_y`` falls within the edge's y-span, linearly interpolate
        the crossing x. A convex polygon crosses any horizontal line in
        at most one contiguous span, so the min/max of the collected
        crossing x's is exactly that span.
        """
        local_vertices = self._local_vertices()
        n = len(local_vertices)
        xs: list[float] = []

        for i in range(n):
            x1, y1 = local_vertices[i]
            x2, y2 = local_vertices[(i + 1) % n]

            if abs(y1 - y2) < _EPS:
                if abs(y1 - local_y) < _EPS:
                    xs.extend((x1, x2))
                continue

            lo, hi = (y1, y2) if y1 < y2 else (y2, y1)
            if lo - _EPS <= local_y <= hi + _EPS:
                t = (local_y - y1) / (y2 - y1)
                xs.append(x1 + t * (x2 - x1))

        if not xs:
            return None

        x_min = min(xs) + self.robot_sweep_margin
        x_max = max(xs) - self.robot_sweep_margin
        if x_min > x_max:
            return None
        return x_min, x_max

    # -- mission lifecycle ---------------------------------------------------

    def reach_closest_corner(self) -> None:
        """Set the robot's goal to the cell corner closest to its current pose.

        Reads ``env.robot.pose`` live rather than a stored entry point --
        see module docstring. "Closest corner" means: whichever end of
        the cell's lane-axis extent is nearer the robot's current
        position decides the starting lane, and whichever end of that
        lane's clipped x-range is nearer decides the starting x.

        Raises:
            RuntimeError: If the robot has no pose yet, or if the cell is
                degenerate enough that no lane bounds can be found near
                its lane-axis extent (should not happen for any cell that
                survived environment_decomposition's convexity/area
                checks).
        """
        if self.env.robot.pose is None:
            raise RuntimeError(
                "SweepingMission.reach_closest_corner() requires the robot's "
                "pose to be set."
            )
        current_position = Vector2(self.env.robot.pose.px, self.env.robot.pose.py)
        current_x, current_y = self._to_local(current_position)
        y_min, y_max = self._y_extent()

        self._shift_positive = abs(current_y - y_min) <= abs(current_y - y_max)
        start_y = (
            y_min + self.robot_sweep_margin
            if self._shift_positive
            else y_max - self.robot_sweep_margin
        )

        lane_bounds = self._lane_bounds(start_y)
        if lane_bounds is None:
            raise RuntimeError(
                f"SweepingMission: no valid lane found near y={start_y!r}; "
                "cell may be too small for the current robot radius/margin."
            )
        self._lane_x_min, self._lane_x_max = lane_bounds

        start_x = (
            self._lane_x_min
            if abs(current_x - self._lane_x_min) <= abs(current_x - self._lane_x_max)
            else self._lane_x_max
        )
        self.sweep_dir = 1 if start_x == self._lane_x_min else -1

        self._lane_local_y = start_y
        self._lane_start_local_x = start_x

        world_point = self._to_world(start_x, start_y)
        self.env.robot.set_goal_position(Goal(world_point.x, world_point.y))
        self.started = True

    def update_sweep(self) -> None:
        """Advance the robot's sweep goal by one step (lawnmower pattern).

        "Primary axis" and "cross axis" are the cell's own computed
        sweep/lane directions rather than a hardcoded x-or-y choice, and
        lane bounds are recomputed per lane instead of fixed -- this is
        what lets one implementation handle every convex cell shape
        without a per-axis special case (the same class of bug the
        original whole-arena version's axis-agnostic rewrite was fixing).
        """
        goal = self.env.robot.goal
        assert goal is not None
        current_x, current_y = self._to_local(Vector2(goal.gx, goal.gy))

        next_x, turned = self._step_primary(current_x)
        next_y = current_y

        if turned:
            self._completed_lane_length_sum += self._lane_x_max - self._lane_x_min

            candidate_y = current_y + (
                self.robot_sweep_lane_step
                if self._shift_positive
                else -self.robot_sweep_lane_step
            )

            y_min, y_max = self._y_extent()
            if self._shift_positive and candidate_y > y_max - self.robot_sweep_margin:
                candidate_y = y_max - self.robot_sweep_margin
                self.sweep_finished = True
            elif (
                not self._shift_positive
                and candidate_y < y_min + self.robot_sweep_margin
            ):
                candidate_y = y_min + self.robot_sweep_margin
                self.sweep_finished = True

            lane_bounds = self._lane_bounds(candidate_y)
            if lane_bounds is None:
                # The cell has narrowed to nothing at this lane (e.g. a
                # triangular cell's tip) -- there is nothing left to sweep.
                self.sweep_finished = True
                lane_bounds = (next_x, next_x)

            self._lane_x_min, self._lane_x_max = lane_bounds
            next_x = min(max(next_x, self._lane_x_min), self._lane_x_max)
            self._lane_start_local_x = (
                self._lane_x_min if self.sweep_dir == 1 else self._lane_x_max
            )
            next_y = candidate_y

        self._lane_local_y = next_y
        world_point = self._to_world(next_x, next_y)
        goal.gx, goal.gy = world_point.x, world_point.y

    def _step_primary(self, primary_pos: float) -> tuple[float, bool]:
        """Advance the sweep-axis coordinate by one step, or turn.

        Returns:
            ``(next_pos, turned)`` -- ``turned`` is ``True`` exactly when
            this call hit the current lane's bound and flipped
            ``sweep_dir``, signalling the caller to also shift lanes.
        """
        next_pos = primary_pos + self.sweep_dir * self.robot_sweep_step

        if self.sweep_dir > 0:
            if next_pos > self._lane_x_max:
                if primary_pos < self._lane_x_max - _EPS:
                    return self._lane_x_max, False
                self.sweep_dir = -1
                return primary_pos, True
        else:
            if next_pos < self._lane_x_min:
                if primary_pos > self._lane_x_min + _EPS:
                    return self._lane_x_min, False
                self.sweep_dir = 1
                return primary_pos, True

        return next_pos, False

    def total_area_swept(self) -> float:
        """Return the area swept so far, in square meters.

        Derived from completed lanes' actual clipped widths plus real
        progress into the current lane -- never from distance to a goal
        the robot hasn't reached yet (crediting area for ground the robot
        is only about to cover would overcount the moment a lane is
        aborted). See module docstring for the non-rectangular-cell
        approximation caveat.
        """
        if not self.started:
            return 0.0
        pose = self.env.robot.pose
        current_x = self._lane_start_local_x
        if pose is not None:
            current_x, _ = self._to_local(Vector2(pose.px, pose.py))

        partial_lane_length = abs(current_x - self._lane_start_local_x)
        lane_width = self.robot_sweep_lane_step

        self.area_swept = lane_width * (
            self._completed_lane_length_sum + partial_lane_length
        )
        return self.area_swept

    def avoid_crowd(self, predictor, safe_point_finder) -> None:
        """Choose one avoidance goal; the caller remains responsible for ticks.

        Call this once per outer-loop iteration while an intrusion is active.
        It intentionally contains no ``Step`` or rendering dependency.
        """
        if not self.avoiding_obstacle:
            assert self.env.robot.pose is not None and self.env.robot.goal is not None
            self.avoiding_obstacle = True
            self.get_data = GetData(
                deepcopy(self.env.robot.pose), deepcopy(self.env.robot.goal)
            )
            self._returning_to_sweep = False

        origin = self.get_data.pose_before_avoidance
        original_goal = self.get_data.goal_before_avoidance
        assert self.env.robot.pose is not None and self.env.robot.sensor is not None
        predictor.obs = self.env.robot.sensor.observe(self.env, robot_visible=False)
        distance = (
            (self.env.robot.pose.px - origin.px) ** 2
            + (self.env.robot.pose.py - origin.py) ** 2
        ) ** 0.5
        observable = distance < self.env.robot.sensor.range
        safe = observable and not predictor.checkIntrusionSAT(origin, original_goal)

        if safe or not observable:
            self._returning_to_sweep = True
            self.current_safe_point = None
            self.env.robot.set_goal_position(Goal(origin.px, origin.py))
        elif not self._returning_to_sweep:
            safe_point = safe_point_finder.find_safe_point(self.env.robot.pose)
            self.current_safe_point = safe_point
            if safe_point is None:
                self.env.robot.set_velocity(0.0, 0.0)
            else:
                self.env.robot.set_goal_position(Goal(*safe_point))

        if self._returning_to_sweep and distance < 0.2:
            self.env.robot.set_goal_position(deepcopy(original_goal))
            self.avoiding_obstacle = False
            self._returning_to_sweep = False
            self.current_safe_point = None

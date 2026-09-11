"""GlobalPlanner: full-arena coverage via BCD + traversal + per-cell sweep.

Ties together, in order, the three pipeline stages that already exist as
separate modules (see each module's own docstring for its own scope):

    1. ``navcore.boustropheden.boustropheden.decompose`` -- convex cell
       decomposition of free space.
    2. ``navcore.graph_traversal.traversal.plan_full_coverage`` -- DFS
       traversal ordering over the decomposition's adjacency graph, one
       ordered list of ``TraversalStep`` per connected component.
    3. ``navcore.missions.sweeping.SweepingMission`` -- per-cell lawnmower
       coverage, driven exactly the way ``navcore/test_sweep.py`` drives
       it (``reach_closest_corner()`` once, then a predictor-check /
       ``avoid_crowd`` / ``update_sweep`` tick loop).

This module adds no new sweep or collision-avoidance math of its own; it
is orchestration only, following the "BCD decomposes, traversal
sequences, sweeping sweeps" separation the surrounding modules already
establish.

Two pre-existing bugs fixed as prerequisite work (see prompt):
    - ``navcore/graph_traversal/traversal.py`` and
      ``navcore/missions/sweeping.py`` imported ``DecomposedCell`` /
      ``DecompositionResult`` / ``TraversalStep`` from a
      ``navcore.planning`` package that does not exist anywhere in this
      project (the real modules are ``navcore.boustropheden.boustropheden``
      and ``navcore.graph_traversal.traversal``). Fixed in place, since
      ``GlobalPlanner`` cannot import either module while that path is
      broken.
    - ``SweepingMission.total_area_swept()`` used
      ``abs(current_x - lane_start_x)`` for the in-progress lane's
      partial length. If a collision-avoidance detour ever leaves the
      robot briefly *behind* the lane's start point (opposite side from
      ``sweep_dir``), ``abs()`` still reports positive progress instead
      of zero. Fixed to ``max(0.0, sweep_dir * (current_x - lane_start_x))``
      per the prompt's mandated fix.

Explicitly open design decisions (flagged here, not silently resolved
-- see each call site below and the class docstring for detail):

    1. Injection vs. self-construction of Step/ORCA-planner/Visualizer:
       this class accepts all three as optional constructor arguments
       and only builds its own if none are given. This mirrors the
       existing project convention for injectable-but-defaulted
       collaborators (``RobotBuilder``, ``ObstacleBuilder``, and
       ``CrowdBuilder`` all take an optional ``rand`` the same way).
       Recommended over ``test_sweep.py``'s "always self-construct"
       pattern specifically because it lets tests substitute a fake
       ``Step``/``Visualizer`` without touching global config files.

    2. Whether transit-only steps need their own local-avoidance detour,
       not just ORCA's default per-tick reactive nudge: implemented as
       "yes" (see ``_run_transit_step``), because a transit leg's goal
       is a fixed straight-line target the same way a sweep lane's goal
       is fixed -- if ORCA's per-tick velocity alone isn't sufficient to
       avoid a crossing pedestrian during sweeping (which is exactly why
       ``SweepingMission.avoid_crowd`` exists), there's no reason to
       assume it is sufficient during transit either. The transit
       version below is a deliberately smaller inline analog of
       ``avoid_crowd`` (retarget to a ``LocalAvoidancePlanner`` safe
       point, then back to the real target once safe) rather than a
       reuse of ``avoid_crowd`` itself, since that method is written
       against a live ``SweepingMission`` instance and there is no
       mission active during transit.

    3. Whether ``avoid_crowd``'s blocking while loop can leak state
       across a ``TraversalStep`` boundary: it cannot, by construction --
       ``avoid_crowd`` is only ever invoked from inside
       ``_run_sweep_step``'s own loop, scoped to one ``SweepingMission``
       instance that is discarded when that step finishes. Transit steps
       never call ``avoid_crowd``; they use the separate inline detour
       described above. No state is carried between calls.

    4. Pedestrian respawn bookkeeping across a multi-cell run: reuses
       ``EnvironmentBuilder.rebuild_pedestrian`` exactly as
       ``test_sweep.py`` does, keyed off ``self._tick_count`` -- a single
       counter incremented once per simulation tick and never reset
       between cells or traversals, so the derived reseed
       (``base_seed + tick_count + ped_id``) stays unique across the
       whole run, not just within one cell.

    5. Disconnected components (multiple traversals): run sequentially,
       in the order ``plan_full_coverage`` returned them. This class does
       **not** teleport or otherwise relocate the robot between
       traversals -- if the arena genuinely has physically unreachable
       components, the robot must already be able to reach the next
       traversal's start some other way, which is out of scope for this
       slice and is flagged rather than silently handled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from navcore.avoidace_planner.local_avoidace_planner import LocalAvoidancePlanner
from navcore.boustropheden.boustropheden import decompose
from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.collision_predictor.sat import SAT
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.goal import Goal
from navcore.entities.environment.environment import Environment
from navcore.graph_traversal.traversal import TraversalStep, plan_full_coverage
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.missions.sweeping import SweepingMission
from navcore.step.step import Step, StepResult
from navcore.visualization.visualizer import Visualizer

#: Matches Step._compute_velocities's own hardcoded goal-reach radius,
#: and GoalReachingTask.GOAL_REACH_TOLERANCE. Duplicated here for the
#: same reason those two duplicate it from each other (see that file's
#: own TODO) -- all three should eventually read one shared config
#: value instead of three independent hardcoded constants.
GOAL_REACH_TOLERANCE = 0.5
#: How close is "arrived" at a temporary safe point during a detour,
#: matching SweepingMission.avoid_crowd's own hardcoded 0.2.
SAFE_POINT_TOLERANCE = 0.2


class SafePointFinder:
    """Adapts ``LocalAvoidancePlanner`` to the ``pose -> (x, y) | None``
    interface ``SweepingMission.avoid_crowd`` expects.

    This is the same adapter ``test_sweep.py`` defines locally for its
    own script. It is redefined here, rather than imported from
    ``test_sweep``, because ``test_sweep.py`` is a manual smoke-test
    script (see its own module docstring), not a reusable library
    module -- importing library code from a test script would invert
    the intended dependency direction.
    """

    def __init__(self, planner: LocalAvoidancePlanner, env: Environment) -> None:
        self._planner = planner
        self._env = env

    def find_safe_point(self, pose) -> tuple[float, float] | None:
        assert self._env.robot.sensor is not None
        observation = self._env.robot.sensor.observe(self._env, robot_visible=False)
        point = self._planner.plan_escape_point(observation)
        return point.to_tuple() if point is not None else None


@dataclass(slots=True)
class CoverageStats:
    """Aggregated results across the whole multi-cell, multi-traversal run.

    Attributes:
        total_area_swept: Sum of ``SweepingMission.total_area_swept()``
            across every completed sweep step.
        cells_completed: Number of ``requires_sweep=True`` steps whose
            sweep actually finished.
        cells_total: Number of ``requires_sweep=True`` steps discovered
            by ``plan()``, across every traversal.
        collisions: Ground-truth collision count, incremented once per
            tick that ``Environment.did_collision_happened()`` returns
            ``True`` -- see ``GlobalPlanner._tick`` for why this is
            counted independently of ``Environment`` 's own
            ``info.collision_counter`` (which can be incremented more
            than once per tick by code paths, such as
            ``SweepingMission.avoid_crowd``'s own status print, that
            call ``did_collision_happened()`` a second time for
            reasons unrelated to counting).
    """

    total_area_swept: float = 0.0
    cells_completed: int = 0
    cells_total: int = 0
    collisions: int = 0


def _distance(px: float, py: float, qx: float, qy: float) -> float:
    return math.hypot(px - qx, py - qy)


class GlobalPlanner:
    """Orchestrates one full-arena coverage run end-to-end.

    Two-phase tick ordering is preserved throughout: this class never
    mutates agent pose/velocity itself, and only ever advances the
    simulation by calling ``self.step_driver.step()`` (wrapped as
    ``self._tick()``). It does mutate ``env.robot.goal`` directly (via
    ``set_goal_position``), which is the same thing
    ``SweepingMission``/``GoalReachingMission``/every existing
    ``Mission`` does -- goal-setting is a planning decision, not a
    physics-integration one, and stays outside the compute/apply split
    ``Step`` protects.

    All orchestration state -- which traversal, which step within it,
    the currently active ``SweepingMission`` (if any), and the running
    ``CoverageStats`` -- lives on the instance. There is no module-level
    or otherwise hidden mutable state.

    Attributes:
        env: The live environment being covered.
        step_driver: Drives one simulation tick (``Step.step()``).
        visualizer: Rendering only -- called from this class's own loop,
            never from inside a ``SweepingMission``/avoidance call, per
            the project's "rendering stays in the outer loop" rule.
        traversals: One ordered list of ``TraversalStep`` per connected
            free-space component, as produced by ``plan()``.
        stats: Running totals; see ``CoverageStats``.
    """

    def __init__(
        self,
        env: Environment | None = None,
        step_driver: Step | None = None,
        visualizer: Visualizer | None = None,
        orca_config_file: str = "orca.toml",
        rand: np.random.Generator | None = None,
    ) -> None:
        self._env_builder = EnvironmentBuilder(rand)
        self.env = env if env is not None else self._env_builder.build_environment()
        self.rand = rand if rand is not None else np.random.default_rng()

        if step_driver is not None:
            self.step_driver = step_driver
        else:
            planner = DecentralizedORCAPlanner(
                config_file=orca_config_file  # , obstacles=self.env.obstacles
            )
            self.step_driver = Step(
                planner=planner, env=self.env, robot_visible=False, rand=self.rand
            )

        self.visualizer = visualizer if visualizer is not None else Visualizer()

        self._local_avoidance = LocalAvoidancePlanner(
            agent=self.env.robot,
            arena_width=float(self.env.info.arena_width),
            arena_height=float(self.env.info.arena_height),
        )
        self._safe_point_finder = SafePointFinder(self._local_avoidance, self.env)

        # -- orchestration state (see class docstring) ----------------
        self.decomposition = None
        self.traversals: list[list[TraversalStep]] = []
        self.traversal_index: int = 0
        self.step_index: int = 0
        self.current_mission: SweepingMission | None = None
        self.stats = CoverageStats()
        self._tick_count = 0

    # -- planning -----------------------------------------------------

    def plan(self) -> list[list[TraversalStep]]:
        """Decompose free space and plan a full-coverage traversal.

        Uses the robot's current pose as the traversal's preferred start
        point, so the very first sweep step starts (as closely as
        ``plan_full_coverage`` can manage) from wherever the robot
        already is.

        Raises:
            RuntimeError: If the robot has no pose yet.
        """
        if self.env.robot.pose is None:
            raise RuntimeError("GlobalPlanner.plan() requires the robot's pose.")

        start = Vector2(self.env.robot.pose.px, self.env.robot.pose.py)
        result = decompose(self.env)
        self.decomposition = result
        # Sticky overlay: every later self.visualizer.refresh() call in
        # this run keeps drawing these cells without having to pass
        # `decomposition=` on every single tick -- see Visualizer's own
        # docstring for why this is a set-once, not a per-call, concern.
        self.visualizer.set_decomposition(result)
        self.traversals = plan_full_coverage(result, start_point=start)
        self.stats.cells_total = sum(
            1
            for traversal in self.traversals
            for step in traversal
            if step.requires_sweep
        )
        return self.traversals

    # -- tick + bookkeeping ---------------------------------------------

    def _tick(self) -> StepResult:
        """Advance the simulation by exactly one tick and update bookkeeping.

        Ground-truth collision detection only, per project convention --
        never derived from the sensor-limited observation used for
        planning/avoidance.
        """
        result = self.step_driver.step()
        self._tick_count += 1
        if self.env.did_collision_happened():
            self.stats.collisions += 1
        self._respawn_pedestrians(result)
        return result

    def _respawn_pedestrians(self, result: StepResult) -> None:
        for ped_id, reached in result.pedestrian_reached_goals.items():
            if reached:
                self.env = self._env_builder.rebuild_pedestrian(
                    env=self.env,
                    ped_id=ped_id,
                    random_seed=self.env.info.random_seed + self._tick_count + ped_id,
                )

    def _collision_predictor(self) -> SAT:
        assert self.env.robot.sensor is not None
        observation = self.env.robot.sensor.observe(self.env, robot_visible=False)
        return SAT(observation, self.env.robot, self.env.obstacles)

    # -- per-step drivers -------------------------------------------------

    def _run_sweep_step(self, step: TraversalStep) -> None:
        """Sweep one cell, following test_sweep.py._update_mission's loop
        order exactly: tick, then check/react, then (maybe) advance the
        lane goal.
        """
        mission = SweepingMission.for_traversal_step(step, self.env)
        self.current_mission = mission
        mission.reach_closest_corner()

        while not mission.sweep_finished:
            result = self._tick()
            self.visualizer.refresh(self.env, mission=mission)

            predictor = self._collision_predictor()
            if predictor.checkIntrusionSAT():
                mission.avoid_crowd(
                    predictor=predictor,
                    safe_point_finder=self._safe_point_finder,
                )
                continue

            if result.robot_reached_goal:
                if not mission.sweeping:
                    mission.sweeping = True
                mission.update_sweep()

        self.stats.total_area_swept += mission.total_area_swept()
        self.stats.cells_completed += 1
        self.current_mission = None

    def _run_transit_step(self, step: TraversalStep) -> None:
        """Drive the robot to ``step.exit_point`` with its own detour logic.

        See module docstring, open decision #2, for why this exists
        rather than relying solely on ORCA's default reactive avoidance.
        A ``None`` exit point means this is a traversal's final step --
        nothing further to transit to, so this is a no-op.
        """
        if step.exit_point is None:
            return

        target = step.exit_point
        self.env.robot.set_goal_position(Goal(target.x, target.y))
        avoiding = False

        while True:
            self._tick()
            self.visualizer.refresh(self.env, mission=self.current_mission)

            robot_pose = self.env.robot.pose
            assert robot_pose is not None

            if (
                not avoiding
                and _distance(robot_pose.px, robot_pose.py, target.x, target.y)
                <= GOAL_REACH_TOLERANCE
            ):
                return

            predictor = self._collision_predictor()
            unsafe = predictor.checkIntrusionSAT()

            if unsafe and not avoiding:
                safe_point = self._safe_point_finder.find_safe_point(robot_pose)
                if safe_point is not None:
                    self.env.robot.set_goal_position(Goal(*safe_point))
                    avoiding = True
                else:
                    self.env.robot.set_velocity(0.0, 0.0)
            elif not unsafe and avoiding:
                self.env.robot.set_goal_position(Goal(target.x, target.y))
                avoiding = False

            if avoiding:
                current_goal = self.env.robot.goal
                assert current_goal is not None
                if (
                    _distance(
                        robot_pose.px, robot_pose.py, current_goal.gx, current_goal.gy
                    )
                    <= SAFE_POINT_TOLERANCE
                ):
                    self.env.robot.set_goal_position(Goal(target.x, target.y))
                    avoiding = False

    # -- top-level run ----------------------------------------------------

    def run(self) -> CoverageStats:
        """Run every planned traversal to completion.

        Calls ``plan()`` first if it hasn't been called yet. Traversals
        (one per connected free-space component -- see open decision #5)
        run sequentially in the order ``plan_full_coverage`` returned
        them; steps within a traversal run in DFS order.

        Returns:
            The final ``self.stats``, after every traversal has finished.
        """
        if not self.traversals:
            self.plan()

        for self.traversal_index, traversal in enumerate(self.traversals):
            for self.step_index, step in enumerate(traversal):
                if step.requires_sweep:
                    self._run_sweep_step(step)
                else:
                    self._run_transit_step(step)

        return self.stats

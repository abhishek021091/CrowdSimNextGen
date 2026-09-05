"""SweepingMission: lawnmower-style coverage mission for the robot."""

from __future__ import annotations

import numpy as np

from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.environment.environment import Environment


class SweepingMission:
    """Lawnmower-style coverage mission for the robot.

    Drives the robot from its current position to the nearest arena
    corner, then sweeps back and forth in parallel lanes (perpendicular
    to ``sweep_axes``) until the far wall is reached.

    All mission progress (``started``, ``sweeping``, ``collisions``,
    ``area_swept``, ``avoiding_obstacle``, ``sweep_finished``) lives on
    the instance, not on a shared class -- see prior revision notes for
    why that matters.
    """

    def __init__(
        self,
        env: Environment,
        robot_sweep_axes: str = "random",
        robot_sweep_step: float = 5,
        robot_sweep_lane_step: float | None = None,
        random_seed: int | None = None,
        robot_sweep_margin: float | None = None,
    ):
        self.env = env
        self.robot_sweep_axes = robot_sweep_axes
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
        seed = random_seed if random_seed is not None else self.env.info.random_seed
        self.rand = np.random.default_rng(seed=seed)

        self.sweep_dir: int = 1
        self.sweep_axes: int = 0
        self.sweep_start: tuple[float, float] = (0.0, 0.0)
        self.sweep_stop: tuple[float, float] | None = None
        self.sweep_finished: bool = False

        self.started: bool = False
        self.sweeping: bool = False
        self.collisions: int = 0
        self.area_swept: float = 0.0
        self.avoiding_obstacle: bool = False
        self._lanes_completed: int = 0
        self._lane_start_primary_pos: float = 0.0

    def reach_closest_corner(self) -> None:
        # Move the robot to the corner of the environment
        assert self.env.robot.pose is not None
        px = self.env.robot.pose.px
        py = self.env.robot.pose.py
        half_width = float(self.env.info.arena_width) / 2
        half_height = float(self.env.info.arena_height) / 2
        gx = np.sign(px) * (half_width - self.robot_sweep_margin)
        gy = np.sign(py) * (half_height - self.robot_sweep_margin)
        sweep_start_pose = Goal(gx, gy)
        self.sweep_start_pose = sweep_start_pose

        if self.robot_sweep_axes == "random":
            sweep_axes = int(self.rand.integers(0, 2))
        else:
            assert self.robot_sweep_axes in (0, 1)
            sweep_axes = self.robot_sweep_axes
        self.sweep_axes = sweep_axes

        if sweep_axes == 0:
            self.sweep_dir = -1 if gx > 0 else 1
            total_lanes_required = float(self.env.info.arena_width) / (
                self.env.robot.radius * 2
            )
            self.sweep_stop = (gx, -gy) if total_lanes_required % 2 == 0 else (-gx, -gy)
        else:
            self.sweep_dir = -1 if gy > 0 else 1
            total_lanes_required = float(self.env.info.arena_height) / (
                self.env.robot.radius * 2
            )
            self.sweep_stop = (-gx, gy) if total_lanes_required % 2 == 0 else (-gx, -gy)

        self.env.robot.set_goal_position(sweep_start_pose)
        self._lane_start_primary_pos = (
            sweep_start_pose.gx if sweep_axes == 0 else sweep_start_pose.gy
        )
        self.started = True

    def update_sweep(self) -> None:
        """Advance the robot's sweep goal by one step (lawnmower pattern).

        Call this once per tick after ``reach_closest_corner`` has been
        called once to establish ``sweep_start``/``sweep_stop``/``sweep_dir``.
        Mutates ``self.env.robot.goal`` in place; sets ``self.sweep_finished``
        to True once the far wall is reached on the cross axis.

        Design note -- why this is axis-agnostic instead of two branches:
            A previous version duplicated this logic once per axis. The
            y-axis copy's lane-shift direction was hardcoded instead of
            depending on the sweep's starting quadrant (unlike the
            x-axis copy, which correctly varied it via
            ``sweep_start[1] > 0``), so sweeps along the y-axis shifted
            lanes the wrong way and finished early, well short of full
            coverage -- the same class of bug as the known
            ``BoustrophedonPlanner`` coverage issue. Expressing the step
            once, parameterized by which coordinate is "primary" (the
            one advancing every tick) vs. "cross" (the one that shifts
            by a lane on each turn), makes that kind of axis-specific
            drift structurally impossible: there is only one
            implementation for both axes to share.
        """
        half_width = float(self.env.info.arena_width) / 2
        half_height = float(self.env.info.arena_height) / 2
        margin = self.robot_sweep_margin
        lane_step = self.robot_sweep_lane_step

        goal = self.env.robot.goal

        if self.sweep_axes == 0:
            assert goal is not None
            primary_pos, cross_pos = goal.gx, goal.gy
            primary_bound, cross_bound = half_width, half_height
            # Which quadrant the sweep started in decides which way
            # lanes shift on each turn.
            shift_positive = self.sweep_start[1] > 0
        else:
            assert goal is not None
            primary_pos, cross_pos = goal.gy, goal.gx
            primary_bound, cross_bound = half_height, half_width
            shift_positive = self.sweep_start[0] > 0

        primary_pos = self._step_primary(
            primary_pos, primary_bound, margin, shift_positive, lane_step
        )
        if self._turned_this_call:
            cross_pos += -lane_step if shift_positive else lane_step

        if cross_pos > cross_bound - margin:
            cross_pos = cross_bound - margin
            self.sweep_finished = True
        elif cross_pos < -cross_bound + margin:
            cross_pos = -cross_bound + margin
            self.sweep_finished = True

        if self.sweep_axes == 0:
            goal.gx, goal.gy = primary_pos, cross_pos
        else:
            goal.gy, goal.gx = primary_pos, cross_pos

    def _step_primary(
        self,
        primary_pos: float,
        bound: float,
        margin: float,
        shift_positive: bool,
        lane_step: float,
    ) -> float:
        """Advance the primary-axis coordinate by one step, or turn.

        Sets ``self._turned_this_call`` so the caller knows whether to
        also shift the cross-axis coordinate this tick.
        """
        self._turned_this_call = False
        next_pos = primary_pos + self.sweep_dir * self.robot_sweep_step

        if next_pos > bound - margin:
            if primary_pos != bound - margin:
                next_pos = bound - margin
            else:
                next_pos = primary_pos
                self.sweep_dir = -1
                self._turned_this_call = True
        elif next_pos < -bound + margin:
            if primary_pos != -bound + margin:
                next_pos = -bound + margin
            else:
                next_pos = primary_pos
                self.sweep_dir = 1
                self._turned_this_call = True

        if self._turned_this_call:
            self._lanes_completed += 1
            self._lane_start_primary_pos = next_pos

        return next_pos

    def total_area_swept(self) -> float:
        """Return the area swept so far, in square meters.

        Derived from completed lanes plus real progress into the current
        lane -- never from distance to a goal the robot hasn't reached
        yet. A tick-by-tick accumulator keyed on pose-to-goal distance
        would credit area for ground the robot is only *about* to cover,
        which overcounts the moment a lane is aborted (avoidance detour,
        collision, early stop). This is deterministic and can't drift.
        """
        arena_length = (
            float(self.env.info.arena_height)
            if self.sweep_axes == 0
            else float(self.env.info.arena_width)
        )
        lane_width = self.robot_sweep_lane_step

        pose = self.env.robot.pose
        current_primary_pos: float = self._lane_start_primary_pos
        if pose is not None:
            current_primary_pos = pose.px if self.sweep_axes == 0 else pose.py

        partial_lane_length: float = abs(
            current_primary_pos - self._lane_start_primary_pos
        )
        self.area_swept: float = (
            self._lanes_completed * lane_width * arena_length
            + lane_width * partial_lane_length
        )
        return self.area_swept

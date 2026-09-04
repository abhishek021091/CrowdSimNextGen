import numpy as np
from navcore.entities.components.goal import Goal
from navcore.entities.environment.environment import Environment
from dataclasses import dataclass


@dataclass
class Sweep:
    started: bool = False
    area_swept: float = 0.0
    collisions: int = 0
    sweeping: bool = False
    avoiding_obstacle: bool = False


class SweepingMission:
    def __init__(
        self,
        env: Environment,
        robot_sweep_axes: str = "random",
        robot_sweep_margin: float = 0.0,
        robot_sweep_step: float = 0.1,
        robot_sweep_lane_step: float | None = None,
        random_seed: int = 42,
    ):
        self.env = env
        self.robot_sweep_axes = robot_sweep_axes
        self.robot_sweep_margin = robot_sweep_margin
        self.robot_sweep_step = robot_sweep_step
        random_seed = self.env.info.random_seed
        # Default lane spacing: one robot diameter, so consecutive lanes
        # don't overlap or leave gaps.
        self.robot_sweep_lane_step = (
            robot_sweep_lane_step
            if robot_sweep_lane_step is not None
            else self.env.robot.radius * 2
        )
        self.rand = np.random.default_rng(seed=random_seed)

        self.sweep_dir: int = 1
        self.sweep_axes: int = 0
        self.sweep_start: tuple[float, float] = (0.0, 0.0)
        self.sweep_stop: Goal | None = None
        self.sweep_finished: bool = False

    def reach_closest_corner(self):
        # Move the robot to the corner of the environment
        px = self.env.robot.pose.px
        py = self.env.robot.pose.py
        gx = np.sign(px) * (self.env.info.arena_width / 2 - self.robot_sweep_margin)
        gy = np.sign(py) * (self.env.info.arena_height / 2 - self.robot_sweep_margin)
        sweep_start = (gx, gy)
        self.sweep_start = (gx, gy)

        if self.robot_sweep_axes == "random":
            sweep_axes = self.rand.integers(0, 2)
        else:
            sweep_axes = self.robot_sweep_axes
        self.sweep_axes = sweep_axes

        if sweep_axes == 0:
            self.sweep_dir = -1 if gx > 0 else 1
            total_lanes_required = (self.env.info.arena_width) / (
                self.env.robot.radius * 2
            )
            if total_lanes_required % 2 == 0:
                self.sweep_stop = (gx, -gy)
            else:
                self.sweep_stop = (-gx, -gy)
        else:
            self.sweep_dir = -1 if gy > 0 else 1
            total_lanes_required = (self.env.info.arena_height) / (
                self.env.robot.radius * 2
            )
            if total_lanes_required % 2 == 0:
                self.sweep_stop = (-gx, gy)
            else:
                self.sweep_stop = (-gx, -gy)

        self.env.robot.set_goal_position(sweep_start)
        Sweep.started = True
        Sweep.sweeping = True

    def update_sweep(self):
        """Advance the robot's sweep goal by one step (lawnmower pattern).

        Call this once per tick after ``reach_closest_corner`` has been
        called once to establish ``sweep_start``/``sweep_stop``/``sweep_dir``.
        Mutates ``self.env.robot.goal`` in place; sets ``self.sweep_finished``
        to True once the far wall is reached on the perpendicular axis.
        """
        width = self.env.configs["arenaSize"]["width"]
        height = self.env.configs["arenaSize"]["height"]
        margin = self.robot_sweep_margin
        lane_step = self.robot_sweep_lane_step

        goal = self.env.robot.goal
        gx, gy = goal.gx, goal.gy

        if self.sweep_axes == 0:  # x-axis sweep
            gx = goal.gx + self.sweep_dir * self.robot_sweep_step

            if gx > width - margin:  # right wall
                if goal.gx != width - margin:
                    gx = width - margin
                else:
                    gx = goal.gx
                    gy += -lane_step if self.sweep_start[1] > 0 else lane_step
                    self.sweep_dir = -1
            elif gx < -width + margin:  # left wall
                if goal.gx != -width + margin:
                    gx = -width + margin
                else:
                    gx = goal.gx
                    gy += -lane_step if self.sweep_start[1] > 0 else lane_step
                    self.sweep_dir = 1

            if gy > height - margin:
                gy = height - margin
                self.sweep_finished = True
            elif gy < -height + margin:
                gy = -height + margin
                self.sweep_finished = True

        else:  # y-axis sweep
            gy = goal.gy + self.sweep_dir * self.robot_sweep_step

            if gy > height - margin:  # top wall
                if goal.gy != height - margin:
                    gy = height - margin
                else:
                    gy = goal.gy
                    gx += -lane_step if np.sign(gx) == 1 else lane_step
                    self.sweep_dir = -1
            elif gy < -height + margin:  # bottom wall
                if goal.gy != -height + margin:
                    gy = -height + margin
                else:
                    gy = goal.gy
                    gx += -lane_step if np.sign(gx) == 1 else lane_step
                    self.sweep_dir = 1

            if gx > width - margin:
                gx = width - margin
                self.sweep_finished = True
            elif gx < -width + margin:
                gx = -width + margin
                self.sweep_finished = True

        self.env.robot.goal.gx = gx
        self.env.robot.goal.gy = gy

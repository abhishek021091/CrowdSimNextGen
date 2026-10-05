# tests/utils/runner.py
"""Runs one scenario with the existing Step / ORCA stack and collects metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from navcore.entities.agents.robot import Robot
from navcore.entities.environment.environment import Environment
from navcore.entities.obstacles.geometry_conversion import obstacle_to_shapely_polygon
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.step.step import Step

from tests.utils.controllers import ORCA_CONFIG_FILE, RobotController
from tests.utils.metrics import (
    DeadlockDetector,
    EpisodeMetrics,
    PathFrame,
    Thresholds,
    Trajectory,
    compute_episode_metrics,
)
from tests.utils.scenario_builder import Scenario, build_environment


@dataclass(slots=True)
class EpisodeResult:
    scenario: Scenario
    controller_name: str
    thresholds: Thresholds
    frame: PathFrame
    trajectory: Trajectory
    metrics: EpisodeMetrics
    v_max: float

    def describe(self) -> str:
        return (
            f"scenario={self.scenario.name} controller={self.controller_name}\n"
            f"{self.metrics.summary()}"
        )


def _left_map(env: Environment) -> bool:
    robot = env.robot
    assert robot.pose is not None
    half_w = float(env.info.arena_width) / 2.0
    half_h = float(env.info.arena_height) / 2.0
    return (
        abs(robot.pose.px) + robot.radius > half_w
        or abs(robot.pose.py) + robot.radius > half_h
    )


def _at_goal(env: Environment) -> bool:
    robot = env.robot
    assert robot.pose is not None and robot.goal is not None
    return (
        math.hypot(robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py)
        <= robot.radius + env.info.goal_reach_tolerance
    )


def run_episode(
    scenario: Scenario,
    controller: RobotController,
    thresholds: Thresholds | None = None,
    robot_visible: bool = False,
) -> EpisodeResult:
    """Simulate ``scenario`` until goal / collision / out-of-bounds /
    deadlock / timeout.

    ``robot_visible=False`` matches ``CrowdSimEnvConfig``'s default: pedestrians
    do not react to the robot, so avoidance is entirely the robot's job.
    """
    thresholds = thresholds or Thresholds()
    env = build_environment(scenario)
    controller.reset(env)

    step_driver = Step(
        env=env,
        robot_visible=robot_visible,
        robot_planner=controller.make_robot_planner(env),
        crowd_planner=DecentralizedORCAPlanner(config_file=ORCA_CONFIG_FILE),
        rand=np.random.default_rng(scenario.seed + 10_000),
    )

    dt = Step.dt
    trajectory = Trajectory(dt=dt)
    trajectory.record(env)
    deadlock = DeadlockDetector(
        window_steps=int(round(thresholds.deadlock_window_s / dt)),
        min_displacement=thresholds.deadlock_min_displacement,
    )
    deadlock.update(trajectory.robot_xy[0])

    polygons = [
        obstacle_to_shapely_polygon(obstacle)
        for key, obstacle in env.obstacles.items()
        if key != "boundary"
    ]
    max_steps = int(round(scenario.time_limit_s / dt))
    termination = "timeout"

    for _ in range(max_steps):
        velocity = controller.act(env)
        step_driver.step(robot_velocity_override=velocity)
        trajectory.record(env)

        if env.did_collision_happened():
            termination = "collision"
            break
        if _left_map(env):
            termination = "out_of_bounds"
            break
        if _at_goal(env):
            termination = "goal"
            break
        if deadlock.update(trajectory.robot_xy[-1]):
            termination = "deadlock"
            break

    frame = PathFrame.from_points(scenario.robot_start, scenario.robot_goal)
    metrics = compute_episode_metrics(
        trajectory,
        frame,
        polygons,
        termination,
        thresholds.oscillation_velocity_threshold,
    )
    return EpisodeResult(
        scenario=scenario,
        controller_name=controller.name,
        thresholds=thresholds,
        frame=frame,
        trajectory=trajectory,
        metrics=metrics,
        v_max=float(Robot.config["kinematics"]["v_pref"]),
    )

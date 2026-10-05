# tests/utils/scenario_builder.py
"""Scenario description and construction on top of the existing builders.

``EnvironmentBuilder`` produces a fully wired ``Environment``; the scenario's
robot pose/goal, pedestrians, groups and obstacles are then placed into it.
Pedestrians are real ``Pedestrian`` agents and are driven by the project's
``DecentralizedORCAPlanner`` through ``Step`` (see ``runner.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.components.velocity import Velocity
from navcore.entities.environment.environment import Environment
from navcore.entities.groups.group import Group
from navcore.entities.obstacles.geometry_conversion import obstacle_to_shapely_polygon
from navcore.entities.obstacles.table import Table

Point2 = tuple[float, float]

ROBOT_START: Point2 = (0.0, -8.0)
ROBOT_GOAL: Point2 = (0.0, 8.0)

#: Free half-width (beyond the robot radius) required around the straight
#: start->goal segment. No obstacle may intrude into this corridor.
STRAIGHT_PATH_CLEARANCE = 0.75
#: Minimum surface-to-surface gap between the robot and any human at spawn.
SPAWN_CLEARANCE = 1.0


class ScenarioError(ValueError):
    """Raised when a scenario violates the suite's construction rules."""


@dataclass(frozen=True)
class HumanSpec:
    start: Point2
    goal: Point2
    speed: float = 1.0
    radius: float = 0.3


@dataclass(frozen=True)
class ObstacleSpec:
    center: Point2
    width: float
    height: float


@dataclass(frozen=True)
class GroupSpec:
    member_indices: tuple[int, ...]
    leader_index: int


def default_obstacles() -> tuple[ObstacleSpec, ...]:
    """Realistic clutter, kept well away from the robot corridor (x ~ 0)
    and from every pedestrian route used by the catalog."""
    return (
        ObstacleSpec((-5.5, -12.0), 1.5, 2.0),
        ObstacleSpec((5.5, -12.0), 1.5, 2.0),
        ObstacleSpec((-5.5, 12.0), 1.5, 2.0),
        ObstacleSpec((5.5, 12.0), 1.5, 2.0),
        ObstacleSpec((-5.0, -17.0), 2.0, 1.5),
        ObstacleSpec((5.0, 17.0), 2.0, 1.5),
    )


@dataclass(frozen=True)
class Scenario:
    name: str
    humans: tuple[HumanSpec, ...]
    robot_start: Point2 = ROBOT_START
    robot_goal: Point2 = ROBOT_GOAL
    groups: tuple[GroupSpec, ...] = ()
    obstacles: tuple[ObstacleSpec, ...] = field(default_factory=default_obstacles)
    seed: int = 0
    time_limit_s: float = 60.0


def build_environment(scenario: Scenario) -> Environment:
    rng = np.random.default_rng(scenario.seed)
    builder = EnvironmentBuilder(rand=rng, include_static_obstacles=False)
    env = builder.build_environment()
    env.info.random_seed = scenario.seed

    _place_robot(env, scenario)
    env.crowd = _build_crowd(scenario, rng)
    env.groups = _build_groups(scenario, env.crowd)
    env.obstacles = _build_obstacles(scenario)
    validate_environment(env, scenario)
    return env


def _place_robot(env: Environment, scenario: Scenario) -> None:
    sx, sy = scenario.robot_start
    gx, gy = scenario.robot_goal
    robot = env.robot
    robot.set_state(
        Pose(sx, sy, math.atan2(gy - sy, gx - sx)),
        Goal(gx, gy),
        robot.v_pref,
        robot.radius,
        Velocity(0.0, 0.0),
    )


def _build_crowd(scenario: Scenario, rng: np.random.Generator) -> dict[int, Pedestrian]:
    crowd: dict[int, Pedestrian] = {}
    for index, spec in enumerate(scenario.humans):
        heading = math.atan2(spec.goal[1] - spec.start[1], spec.goal[0] - spec.start[0])
        pedestrian = Pedestrian(rng)
        pedestrian.set_id(index)
        pedestrian.set_state(
            Pose(spec.start[0], spec.start[1], heading),
            Goal(spec.goal[0], spec.goal[1]),
            spec.speed,
            spec.radius,
            Velocity(spec.speed * math.cos(heading), spec.speed * math.sin(heading)),
        )
        crowd[index] = pedestrian
    return crowd


def _build_groups(scenario: Scenario, crowd: dict[int, Pedestrian]) -> dict[int, Group]:
    groups: dict[int, Group] = {}
    for group_id, spec in enumerate(scenario.groups):
        leader = crowd[spec.leader_index]
        assert leader.goal is not None
        group = Group(
            id=group_id,
            member_ids=tuple(spec.member_indices),
            goal=Goal(leader.goal.gx, leader.goal.gy),
            leader_id=spec.leader_index,
        )
        for member_id in spec.member_indices:
            crowd[member_id].group = group
            crowd[member_id].group_id = group_id
        groups[group_id] = group
    return groups


def _build_obstacles(scenario: Scenario) -> dict:
    obstacles = {}
    for index, spec in enumerate(scenario.obstacles):
        table = Table.rectangular(
            id=f"table_{index}",
            center=Vector2(*spec.center),
            width=spec.width,
            height=spec.height,
            name=f"Table {index}",
        )
        obstacles[table.id] = table.to_obstacle()
    return obstacles


def validate_environment(env: Environment, scenario: Scenario) -> None:
    """Enforce the suite's construction rules.

    * The straight robot start->goal corridor is obstacle free.
    * Nothing spawns inside an obstacle or outside the arena.
    * No human spawns overlapping another human or crowding the robot.
    """
    robot = env.robot
    corridor = LineString([scenario.robot_start, scenario.robot_goal]).buffer(
        robot.radius + STRAIGHT_PATH_CLEARANCE
    )
    half_w = float(env.info.arena_width) / 2.0
    half_h = float(env.info.arena_height) / 2.0

    for key, obstacle in env.obstacles.items():
        polygon = obstacle_to_shapely_polygon(obstacle)
        if corridor.intersects(polygon):
            raise ScenarioError(
                f"[{scenario.name}] obstacle {key} intrudes into the straight "
                f"robot-goal corridor."
            )
        for index, human in env.crowd.items():
            assert human.pose is not None and human.goal is not None
            for label, (px, py) in (
                ("start", (human.pose.px, human.pose.py)),
                ("goal", (human.goal.gx, human.goal.gy)),
            ):
                if polygon.distance(Point(px, py)) < human.radius:
                    raise ScenarioError(
                        f"[{scenario.name}] human {index} {label} overlaps {key}."
                    )

    for index, human in env.crowd.items():
        assert human.pose is not None and human.goal is not None
        for px, py in (
            (human.pose.px, human.pose.py),
            (human.goal.gx, human.goal.gy),
        ):
            if abs(px) + human.radius > half_w or abs(py) + human.radius > half_h:
                raise ScenarioError(f"[{scenario.name}] human {index} outside arena.")
        assert robot.pose is not None
        gap = (
            math.hypot(human.pose.px - robot.pose.px, human.pose.py - robot.pose.py)
            - human.radius
            - robot.radius
        )
        if gap < SPAWN_CLEARANCE:
            raise ScenarioError(
                f"[{scenario.name}] human {index} spawns {gap:.2f} m from the robot."
            )
        for other_index, other in env.crowd.items():
            if other_index <= index:
                continue
            assert other.pose is not None
            if (
                math.hypot(human.pose.px - other.pose.px, human.pose.py - other.pose.py)
                < human.radius + other.radius
            ):
                raise ScenarioError(
                    f"[{scenario.name}] humans {index} and {other_index} overlap."
                )

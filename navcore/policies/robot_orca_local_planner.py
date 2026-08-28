from __future__ import annotations

from math import cos, hypot, pi, sin
from pathlib import Path
from typing import Any

import rvo2
import tomllib

import navcore.configs
from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.polygon import Polygon
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.components.velocity import Velocity
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.environment.environment import Environment


class BaseORCAPlanner:
    CONFIG_DIR = Path(navcore.configs.__file__).parent

    with open(CONFIG_DIR / "env.toml", "rb") as f:
        env_config: dict[str, Any] = tomllib.load(f)

    TIME_STEP = env_config["policy"]["time_step"]

    def __init__(self, config_file: str) -> None:

        with open(self.CONFIG_DIR / config_file, "rb") as f:
            config = tomllib.load(f)

        self.sim = rvo2.PyRVOSimulator(
            self.TIME_STEP,
            config["orca"]["neighbor_dist"],
            config["orca"]["max_neighbors"],
            config["orca"]["time_horizon"],
            config["orca"]["time_horizon_obst"],
            config["orca"]["agent_radius"],
            config["orca"]["max_speed"],
        )

        self.robot_id: int | None = None
        self.pedestrian_ids: dict[int, int] = {}

        self.initialized = False

        def initialize(self, env: Environment) -> None:

        robot = env.robot
        assert robot.pose is not None

        self.robot_id = self.sim.addAgent(
            (robot.pose.px, robot.pose.py)
        )

        self.pedestrian_ids.clear()

        for ped in env.crowd.values():
            assert ped.pose is not None

            self.pedestrian_ids[ped.id] = self.sim.addAgent(
                (ped.pose.px, ped.pose.py)
            )

        for obstacle in env.obstacles.values():
            self._add_obstacle(obstacle)

        self.sim.processObstacles()

        self.initialized = True

        def _sync_agents(self, env: Environment) -> None:

        assert self.robot_id is not None

        robot = env.robot
        assert robot.pose is not None

        self.sim.setAgentPosition(
            self.robot_id,
            (robot.pose.px, robot.pose.py),
        )

        if robot.velocity is None:
            self.sim.setAgentVelocity(self.robot_id, (0.0, 0.0))
        else:
            self.sim.setAgentVelocity(
                self.robot_id,
                (robot.velocity.vx, robot.velocity.vy),
            )

        for ped in env.crowd.values():

            assert ped.pose is not None

            self.sim.setAgentPosition(
                self.pedestrian_ids[ped.id],
                (ped.pose.px, ped.pose.py),
            )

            if ped.velocity is None:
                self.sim.setAgentVelocity(
                    self.pedestrian_ids[ped.id],
                    (0.0, 0.0),
                )
            else:
                self.sim.setAgentVelocity(
                    self.pedestrian_ids[ped.id],
                    (ped.velocity.vx, ped.velocity.vy),
                )

        @staticmethod
    def _preferred_velocity(px, py, gx, gy, speed):

        dx = gx - px
        dy = gy - py

        distance = hypot(dx, dy)

        if distance <= 1e-6:
            return (0.0, 0.0)

        return (
            speed * dx / distance,
            speed * dy / distance,
        )

    def _add_obstacle(self, obstacle: Obstacle) -> None:
    geometry = obstacle.geometry

    if isinstance(geometry, Polygon):
        vertices = [
            (vertex.x, vertex.y)
            for vertex in geometry.vertices
        ]

    elif isinstance(geometry, Rectangle):
        vertices = [
            (vertex.x, vertex.y)
            for vertex in geometry.vertices()
        ]

    elif isinstance(geometry, Circle):
        # Approximate the circle with a polygon.
        NUM_SEGMENTS = 16

        vertices = []

        for i in range(NUM_SEGMENTS):
            theta = 2.0 * pi * i / NUM_SEGMENTS

            vertices.append(
                (
                    geometry.center.x + geometry.radius * cos(theta),
                    geometry.center.y + geometry.radius * sin(theta),
                )
            )

    else:
        raise TypeError(
            f"Unsupported obstacle geometry: {type(geometry).__name__}"
        )

    self.sim.addObstacle(vertices)
        def _sync_agents(self, env: Environment) -> None:

        assert self.robot_id is not None

        robot = env.robot
        assert robot.pose is not None

        self.sim.setAgentPosition(
            self.robot_id,
            (robot.pose.px, robot.pose.py),
        )

        if robot.velocity is None:
            self.sim.setAgentVelocity(self.robot_id, (0.0, 0.0))
        else:
            self.sim.setAgentVelocity(
                self.robot_id,
                (robot.velocity.vx, robot.velocity.vy),
            )

        for ped in env.crowd.values():

            assert ped.pose is not None

            self.sim.setAgentPosition(
                self.pedestrian_ids[ped.id],
                (ped.pose.px, ped.pose.py),
            )

            if ped.velocity is None:
                self.sim.setAgentVelocity(
                    self.pedestrian_ids[ped.id],
                    (0.0, 0.0),
                )
            else:
                self.sim.setAgentVelocity(
                    self.pedestrian_ids[ped.id],
                    (ped.velocity.vx, ped.velocity.vy),
                )

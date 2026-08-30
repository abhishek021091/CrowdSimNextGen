from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
from numpy import atan2

import navcore.configs
from navcore.entities.agents.agent import Agent
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.components.velocity import Velocity
from navcore.environment.environment import Environment
from navcore.middleware.orca_middleware import VelocityPlanner


@dataclass(slots=True)
class StepResult:
    robot_velocity: Velocity
    crowd_velocities: dict[int, Velocity]


class Step:
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)

    ROBOT_KEY = -1
    dt = env_config["policy"]["time_step"]

    def __init__(
        self,
        agent: Robot | Pedestrian,
        planner: VelocityPlanner,
        env: Environment,
        robot_visible: bool,
    ) -> None:
        self.agent = agent
        self.env = env
        self.planner: Any = planner
        self.robot_visible = robot_visible

    def step(self) -> None:
        self._validate()
        self._compute_velocities()
        self._advance_agent(self.env.robot)
        for ped in self.env.crowd.values():
            self._advance_agent(ped)

    def _validate(self) -> None:
        if (
            self.env.robot.pose is None
            or self.env.robot.velocity is None
            or self.env.robot.goal is None
        ):
            raise RuntimeError("Robot must have pose, velocity, and goal initialized.")

        for ped in self.env.crowd.values():
            if ped.pose is None or ped.velocity is None or ped.goal is None:
                raise RuntimeError(
                    f"Pedestrian {ped.id} must have pose, velocity, and goal initialized."
                )

    def _compute_velocities(
        self,
    ) -> tuple[Velocity, dict[int, Velocity]]:

        assert self.agent.sensor is not None
        observations = self.agent.sensor.observe(
            self.env, robot_visible=self.robot_visible
        )
        if isinstance(self.agent, Robot):
            observations[self.ROBOT_KEY] = self.agent.get_observable_state()
        else:
            observations[self.agent.id] = self.agent.get_observable_state()

        self_velocity, other_velocities = self.planner.compute_velocities(
            self.agent.get_observable_state(),
            observations,
        )

        return self_velocity, other_velocities

    def set_velocities(
        self,
        robot_velocity: Velocity,
        crowd_velocities: dict[int, Velocity],
    ) -> None:
        self.env.robot.set_velocity(robot_velocity.vx, robot_velocity.vy)
        for ped_id, vel in crowd_velocities.items():
            if ped_id == -1:
                continue  # Skip the robot key
            if ped_id not in self.env.crowd:
                raise KeyError(f"Pedestrian ID {ped_id} not found in the environment.")
            self.env.crowd[ped_id].set_velocity(vel.vx, vel.vy)

    def _advance_agent(self, agent: Agent) -> None:
        if agent.pose is None:
            raise RuntimeError("Agent pose is missing.")
        if agent.velocity is None:
            raise RuntimeError("Agent velocity is missing.")

        # Assumes pose uses px/py and velocity uses vx/vy
        agent.pose.px += agent.velocity.vx * self.dt
        agent.pose.py += agent.velocity.vy * self.dt

        # Optional heading update if your pose has an angle field
        speed_sq = (
            agent.velocity.vx * agent.velocity.vx
            + agent.velocity.vy * agent.velocity.vy
        )
        if speed_sq > 1e-12:
            heading = atan2(agent.velocity.vy, agent.velocity.vx)
            if hasattr(agent.pose, "theta"):
                agent.pose.theta = heading

    def _apply_robot_velocity(
        self, robot: Robot, velocities: dict[int, Velocity]
    ) -> Velocity:
        if self.ROBOT_KEY not in velocities:
            raise KeyError("Planner did not return a velocity for the robot.")
        robot.velocity = velocities[self.ROBOT_KEY]
        return robot.velocity

    def _apply_crowd_velocities(
        self, crowd: list[Pedestrian], velocities: dict[int, Velocity]
    ) -> dict[int, Velocity]:
        crowd_velocities: dict[int, Velocity] = {}

        for ped in crowd:
            assert ped.id is not None
            if ped.id not in velocities:
                raise KeyError(
                    f"Planner did not return a velocity for pedestrian '{ped.id}'."
                )
            ped.velocity = velocities[ped.id]
            crowd_velocities[ped.id] = ped.velocity

        return crowd_velocities

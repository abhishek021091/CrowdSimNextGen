from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    with open(env_path, "rb") as f:
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
        self.robot = env.robot
        self.crowd = env.crowd
        self.planner = planner
        self.robot_visible = robot_visible

    def step(self) -> StepResult:
        self._validate()
        result = self._compute_velocities()
        self.set_velocities(result.robot_velocity, result.crowd_velocities)

        self._advance_agent(self.env.robot)
        for ped in self.env.crowd.values():
            self._advance_agent(ped)

        return result

    def _validate(self) -> None:
        if (
            self.env.robot.pose is None
            or self.env.robot.velocity is None
            or self.env.robot.goal is None
        ):
            raise RuntimeError("Robot must have pose, velocity, and goal initialized.")
        if self.env.robot.sensor is None:
            raise RuntimeError("Robot sensor must be initialized.")

        for ped in self.env.crowd.values():
            if ped.pose is None or ped.velocity is None or ped.goal is None:
                raise RuntimeError(
                    f"Pedestrian {ped.id} must have pose, velocity, and goal initialized."
                )
            if ped.sensor is None:
                raise RuntimeError(f"Pedestrian {ped.id} sensor must be initialized.")

    def _compute_velocities(self) -> StepResult:
        robot_velocity = self._compute_robot_velocity()
        crowd_velocities = self._compute_crowd_velocities()
        return StepResult(
            robot_velocity=robot_velocity, crowd_velocities=crowd_velocities
        )

    def _compute_robot_velocity(self) -> Velocity:
        assert self.env.robot.sensor is not None
        robot_obs = self.env.robot.sensor.observe(
            self.env, robot_visible=self.robot_visible
        )
        robot_obs[self.ROBOT_KEY] = self.env.robot.get_observable_state()

        robot_velocity, _ = self.planner.compute_velocities(
            self.ROBOT_KEY,
            self.env.robot.get_full_state(),
            robot_obs,
        )
        return robot_velocity

    def _compute_crowd_velocities(self) -> dict[int, Velocity]:
        crowd_velocities: dict[int, Velocity] = {}

        for ped_id, ped in self.env.crowd.items():
            assert ped.sensor is not None
            ped_obs = ped.sensor.observe(self.env, robot_visible=self.robot_visible)
            ped_obs[ped_id] = ped.get_observable_state()

            ped_velocity, _ = self.planner.compute_velocities(
                ped_id,
                ped.get_full_state(),
                ped_obs,
            )
            crowd_velocities[ped_id] = ped_velocity

        return crowd_velocities

    def set_velocities(
        self,
        robot_velocity: Velocity,
        crowd_velocities: dict[int, Velocity],
    ) -> None:
        self._apply_robot_velocity(self.env.robot, {self.ROBOT_KEY: robot_velocity})
        self._apply_crowd_velocities(list(self.env.crowd.values()), crowd_velocities)

    def _advance_agent(self, agent: Agent) -> None:
        if agent.pose is None:
            raise RuntimeError("Agent pose is missing.")
        if agent.velocity is None:
            raise RuntimeError("Agent velocity is missing.")

        agent.pose.px += agent.velocity.vx * self.dt
        agent.pose.py += agent.velocity.vy * self.dt

        speed_sq = (
            agent.velocity.vx * agent.velocity.vx
            + agent.velocity.vy * agent.velocity.vy
        )
        if speed_sq > 1e-12 and hasattr(agent.pose, "theta"):
            agent.pose.theta = atan2(agent.velocity.vy, agent.velocity.vx)

    def _apply_robot_velocity(
        self,
        robot: Robot,
        velocities: dict[int, Velocity],
    ) -> Velocity:
        if self.ROBOT_KEY not in velocities:
            raise KeyError("Planner did not return a velocity for the robot.")
        robot.velocity = velocities[self.ROBOT_KEY]
        return robot.velocity

    def _apply_crowd_velocities(
        self,
        crowd: list[Pedestrian],
        velocities: dict[int, Velocity],
    ) -> dict[int, Velocity]:
        crowd_velocities: dict[int, Velocity] = {}

        for ped in crowd:
            if ped.id not in velocities:
                raise KeyError(
                    f"Planner did not return a velocity for pedestrian '{ped.id}'."
                )
            ped.velocity = velocities[ped.id]
            crowd_velocities[ped.id] = ped.velocity

        return crowd_velocities

from __future__ import annotations

from abc import ABC, abstractmethod
from math import hypot

from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.components.state import ObservableState
from navcore.environment.environment import Environment


class Sensor(ABC):
    """Base class for all robot sensors.

    A sensor converts the ground-truth environment into the robot's
    observable state.
    """

    @abstractmethod
    def observe(
        self, environment: Environment, robot_visible: bool
    ) -> dict[int, ObservableState]:
        """Return the robot's observable state."""
        ...


class RangeSensor(Sensor):
    """Circular range sensor.

    Detects every pedestrian whose center lies within the sensing radius.
    Obstacles and occlusions are ignored.
    """

    def __init__(self, agent: Pedestrian | Robot) -> None:
        self.agent: Pedestrian | Robot = agent
        self.range = agent.config["sensors"]["range"]

    def observe(
        self, environment: Environment, robot_visible: bool
    ) -> dict[int, ObservableState]:
        observer: Pedestrian | Robot

        if isinstance(self.agent, Pedestrian):
            observer = environment.crowd[self.agent.id]
        else:
            observer = environment.robot

        assert observer.pose is not None
        assert observer.goal is not None

        observation: dict[int, ObservableState] = {}

        for i, pedestrian in enumerate(environment.crowd.values()):
            assert pedestrian.pose is not None

            distance = hypot(
                pedestrian.pose.px - observer.pose.px,
                pedestrian.pose.py - observer.pose.py,
            )

            if distance > self.range:
                continue

            observation[i] = pedestrian.get_observable_state()

        if isinstance(self.agent, Pedestrian) and robot_visible == True:
            observation[-1] = environment.robot.get_observable_state()
        return observation

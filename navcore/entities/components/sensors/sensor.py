from __future__ import annotations

from abc import ABC, abstractmethod
from math import hypot
from typing import TYPE_CHECKING

import numpy as np

from navcore.entities.components.state import ObservableState

if TYPE_CHECKING:
    from navcore.entities.agents.pedestrians import Pedestrian
    from navcore.entities.agents.robot import Robot
    from navcore.environment.environment import Environment


class Sensor(ABC):
    @abstractmethod
    def observe(
        self, environment: Environment, robot_visible: bool
    ) -> dict[int, ObservableState]:
        raise NotImplementedError


class RangeSensor(Sensor):
    def __init__(self, agent: Pedestrian | Robot) -> None:
        self.agent = agent
        self.range = agent.config["sensors"]["sensor_range"]
        self.fov = agent.config["sensors"]["sensor_fov"] * np.pi

    def observe(
        self, environment: Environment, robot_visible: bool
    ) -> dict[int, ObservableState]:
        from navcore.entities.agents.pedestrians import Pedestrian

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
            if distance <= self.range:
                observation[i] = pedestrian.get_observable_state()

        if isinstance(self.agent, Pedestrian) and robot_visible:
            observation[-1] = environment.robot.get_observable_state()

        return observation

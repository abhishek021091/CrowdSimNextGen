from __future__ import annotations

from dataclasses import dataclass

from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.groups.group import Group
from navcore.entities.obstacles import Obstacle


@dataclass(slots=True)
class Scenario:
    """Complete description of one simulation episode."""

    robot: Robot
    crowd: dict[int, Pedestrian]
    groups: dict[int, Group]
    obstacles: dict[int, Obstacle]

from dataclasses import dataclass

from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.groups.group import Group
from navcore.entities.obstacles.obstacle import Obstacle


@dataclass
class Environment:
    obstacles: dict[str, Obstacle]
    crowd: dict[int, Pedestrian]
    groups: dict[int, Group]
    robot: Robot

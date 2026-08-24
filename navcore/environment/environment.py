from dataclasses import dataclass

from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.obstacles.obstacle import Obstacle


@dataclass
class Environment:
    obstacles: list[Obstacle]
    crowd: list[Pedestrian]
    robot: Robot

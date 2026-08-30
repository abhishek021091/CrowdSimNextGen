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

    def robot_state(self) -> Robot:
        return self.robot

    def crowd_state(self) -> dict[int, Pedestrian]:
        return self.crowd

    def obstacle_state(self) -> dict[str, Obstacle]:
        return self.obstacles

    def group_state(self) -> dict[int, Group]:
        return self.groups

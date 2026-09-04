import numpy as np

from navcore.entities.agents.robot import Robot
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.rectangle import Rectangle


class CollisionChecker:
    def __init__(
        self, robot: Robot, crowd: dict[int, Pedestrian], obstacles: dict[int, Obstacle]
    ):
        self.robot = robot
        self.crowd = crowd
        self.obstacles = obstacles

    def check_collision(self) -> bool:
        """Check if the robot is in collision with any pedestrian."""
        robot_pose = self.robot.pose
        for ped in self.crowd.values():
            ped_pose = ped.pose
            assert robot_pose is not None, "Robot pose is None"
            assert ped_pose is not None, "Pedestrian pose is None"
            distance = np.linalg.norm(
                [robot_pose.px - ped_pose.px, robot_pose.py - ped_pose.py]
            )
            if distance < (self.robot.radius + ped.radius):
                return True
        for obs in self.obstacles.values():
            obs_pose = obs.geometry.center
            assert robot_pose is not None, "Robot pose is None"
            if isinstance(obs.geometry, Circle):
                distance = np.linalg.norm(
                    [robot_pose.px - obs_pose.x, robot_pose.py - obs_pose.y]
                )
                if distance < (self.robot.radius + obs.geometry.radius):
                    return True
            elif isinstance(obs.geometry, Rectangle):
                # Check if the robot is within the rectangle bounds
                half_width = obs.geometry.width / 2
                half_height = obs.geometry.height / 2
                if (
                    obs_pose.x - half_width <= robot_pose.px <= obs_pose.x + half_width
                    and obs_pose.y - half_height
                    <= robot_pose.py
                    <= obs_pose.y + half_height
                ):
                    return True
        return False

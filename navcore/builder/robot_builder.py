import numpy as np

from navcore.configs import env
from navcore.entities.agents.robot import Robot


class RobotBuilder:
    def __init__(self):
        pass

    def build_robot(self):
        robot = Robot()
        robot.set_state(
            *self.generate_pose(),
            *self.generate_goal(),
            robot.v_pref,
            robot.radius,
        )

    def generate_pose(self):
        theta = self.rng.uniform(0, 2 * np.pi)
        px = self.rng.uniform(-env["arena"]["width"] / 2, env["arena"]["width"] / 2)
        py = self.rng.uniform(-env["arena"]["height"] / 2, env["arena"]["height"] / 2)
        return px, py, theta

    def generate_goal(self):
        gx = self.rng.uniform(-env["arena"]["width"] / 2, env["arena"]["width"] / 2)
        gy = self.rng.uniform(-env["arena"]["height"] / 2, env["arena"]["height"] / 2)
        return gx, gy

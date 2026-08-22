import numpy as np

from navcore.configs import env, pedestrian
from navcore.entities.agents.pedestrians import Pedestrian


class CrowdBuilder:
    def __init__(self):
        self.pedestrian_num = env[pedestrian]["num_pedestrians"]

    def build_crowd(self):
        for i in range(self.pedestrian_num):
            pedestrian = Pedestrian()
            pedestrian.set_id(i)
            pedestrian.set_state(
                *self.generate_pose(),
                *self.generate_goal(),
                pedestrian.v_pref,
                pedestrian.radius,
            )

    def generate_pose(self):
        theta = self.rng.uniform(0, 2 * np.pi)
        sides = [
            (-env["arena"]["width"] / 2, env["arena"]["width"] / 2),
            (-env["arena"]["height"] / 2, env["arena"]["height"] / 2),
        ]
        px = self.rng.uniform(*sides[0]) + np.random.choice([-0.5, 0.5])
        py = self.rng.uniform(*sides[1]) + np.random.choice([-0.5, 0.5])
        return px, py, theta

    def generate_goal(self):
        gx = self.rng.uniform(-env["arena"]["width"] / 2, env["arena"]["width"] / 2)
        gy = self.rng.uniform(-env["arena"]["height"] / 2, env["arena"]["height"] / 2)
        return gx, gy

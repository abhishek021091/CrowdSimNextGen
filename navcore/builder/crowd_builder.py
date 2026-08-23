from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose


class CrowdBuilder:
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    config_path = Path(navcore.configs.__file__).parent / "pedestrian.toml"
    with open(Path(config_path), "rb") as f:
        config = tomllib.load(f)
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)
    rand = np.random.default_rng(seed=env_config["random"]["seed"])

    def __init__(self):
        self.pedestrian_num = self.env_config["pedestrian"]["num_pedestrians"]

    def build_crowd(self):
        for i in range(self.pedestrian_num):
            pedestrian = Pedestrian()
            pedestrian.set_id(i)
            pedestrian.set_state(
                self.generate_pose(),
                self.generate_goal(),
                pedestrian.v_pref,
                pedestrian.radius,
            )

    def generate_pose(self) -> Pose:
        theta = self.rand.uniform(0, 2 * np.pi)

        sides: list[tuple[float, float]] = [
            (self.env_config["arena"]["width"] / 2, 0),
            (-self.env_config["arena"]["width"] / 2, 0),
            (0, self.env_config["arena"]["height"] / 2),
            (0, -self.env_config["arena"]["height"] / 2),
        ]

        side = np.random.choice(sides)

        px = side[0] + np.cos(theta) + np.random.choice([-0.5, 0.5])
        py = side[1] + np.sin(theta) + np.random.choice([-0.5, 0.5])
        return Pose(px, py, theta)

    def generate_goal(self) -> Goal:
        theta = self.rand.uniform(0, 2 * np.pi)
        sides: list[tuple[float, float]] = [
            (self.env_config["arena"]["width"] / 2, 0),
            (-self.env_config["arena"]["width"] / 2, 0),
            (0, self.env_config["arena"]["height"] / 2),
            (0, -self.env_config["arena"]["height"] / 2),
        ]

        side = np.random.choice(sides)

        gx = side[0] + np.cos(theta) + np.random.choice([-0.5, 0.5])
        gy = side[1] + np.sin(theta) + np.random.choice([-0.5, 0.5])
        return Goal(gx, gy)

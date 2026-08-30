from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.builder.crowd_builder import CrowdBuilder
from navcore.builder.obstacle_builder import ObstacleBuilder
from navcore.builder.robot_builder import RobotBuilder
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.environment.environment import Environment


class EnvironmentBuilder:
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)
    rand = np.random.default_rng(seed=env_config["random"]["seed"])

    def __init__(self):
        self.crowd_builder = CrowdBuilder()
        self.robot_builder = RobotBuilder(self.rand)
        self.obstacle_builder = ObstacleBuilder(self.rand)

    def build_environment(self) -> Environment:

        # self.obstacle_builder.build_boundary()
        self.obstacle_builder.build_table()
        self.crowd_builder.build_crowd()
        self.crowd_builder.build_groups()
        self.robot_builder.build_robot()
        assert self.robot_builder.robot is not None, "Robot has not been built."
        return Environment(
            self.obstacle_builder.obstacles,
            self.crowd_builder.crowd,
            self.crowd_builder.groups,
            self.robot_builder.robot,
        )

    def build_crowd(self) -> dict[int, "Pedestrian"]:
        crowd = self.crowd_builder.build_crowd()
        self.crowd_builder.build_groups()
        return crowd

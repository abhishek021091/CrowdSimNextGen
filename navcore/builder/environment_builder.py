from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.builder.crowd_builder import CrowdBuilder
from navcore.builder.obstacle_builder import ObstacleBuilder
from navcore.builder.robot_builder import RobotBuilder
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.environment.environment import Environment, EnvironmentInfo


class EnvironmentBuilder:
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)
    rand = np.random.default_rng(seed=env_config["random"]["seed"])

    def __init__(self):
        self.info = EnvironmentInfo()
        self.crowd_builder = CrowdBuilder(self.rand)
        self.robot_builder = RobotBuilder(self.rand)
        self.obstacle_builder = ObstacleBuilder(self.rand)

    def build_environment(self) -> Environment:

        # self.obstacle_builder.build_boundary()
        self.obstacle_builder.build_table()
        self.crowd_builder.build_crowd()
        self.crowd_builder.build_groups()
        self.robot_builder.build_robot()
        return Environment(
            self.info,
            self.obstacle_builder.obstacles,
            self.crowd_builder.crowd,
            self.crowd_builder.groups,
            self.robot_builder.robot,
        )

    def rebuild_crowd(self, env: Environment, random_seed: int) -> Environment:
        rand = np.random.default_rng(seed=random_seed)
        crowd_builder = CrowdBuilder(rand)
        rebuilded_crowd = crowd_builder.build_crowd()
        rebuilded_group = crowd_builder.build_groups()
        self.crowd_builder.build_groups()
        assert rebuilded_crowd is not None

        env.crowd = rebuilded_crowd
        if rebuilded_group:
            env.groups = rebuilded_group  # Add the newly built groups
        return env  # Return the newly built crowd

    def reset(self, random_seed: int) -> Environment:
        rand = np.random.default_rng(seed=random_seed)
        crowd_builder = CrowdBuilder(rand)
        robot_builder = RobotBuilder(rand)
        obstacle_builder = ObstacleBuilder(rand)
        obstacle_builder.build_table()
        crowd_builder.build_crowd()
        crowd_builder.build_groups()
        robot_builder.build_robot()
        return Environment(
            self.info,
            obstacle_builder.obstacles,
            crowd_builder.crowd,
            crowd_builder.groups,
            robot_builder.robot,
        )

    def rebuild_pedestrian(
        self, env: Environment, ped_id: int, random_seed: int
    ) -> Environment:
        rand = np.random.default_rng(seed=random_seed)
        crowd_builder = CrowdBuilder(rand)
        new_pedestrian = crowd_builder.build_single_pedestrian(ped_id)
        if new_pedestrian is not None:
            env.crowd[ped_id] = new_pedestrian
        return env

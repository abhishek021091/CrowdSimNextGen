from dataclasses import dataclass
from pathlib import Path

import tomllib

import navcore.configs
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.groups.group import Group
from navcore.entities.obstacles.obstacle import Obstacle


@dataclass
class EnvironmentInfo:
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)

    arena_width: str = env_config["arenaSize"]["width"]
    arena_height: str = env_config["arenaSize"]["height"]
    random_seed: int = env_config["random"]["seed"]


@dataclass
class Environment:
    info: EnvironmentInfo
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

    def get_info(self) -> EnvironmentInfo:
        return self.info

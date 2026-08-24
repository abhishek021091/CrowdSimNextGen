from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.entities.agents.robot import Robot
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose


class RobotBuilder:
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)
    rand = np.random.default_rng(seed=env_config["random"]["seed"])

    def __init__(self):
        self.robot: Robot | None = None

    def build_robot(self):
        robot = Robot()
        robot.set_state(
            self.generate_pose(),
            self.generate_goal(),
            robot.v_pref,
            robot.radius,
        )
        self.robot = robot

    def generate_pose(self) -> Pose:
        theta = self.rand.uniform(0, 2 * np.pi)
        px = self.rand.uniform(
            -self.env_config["arenaSize"]["width"] / 2,
            self.env_config["arenaSize"]["width"] / 2,
        )
        py = self.rand.uniform(
            -self.env_config["arenaSize"]["height"] / 2,
            self.env_config["arenaSize"]["height"] / 2,
        )
        return Pose(px, py, theta)

    def generate_goal(self) -> Goal:
        gx = self.rand.uniform(
            -self.env_config["arenaSize"]["width"] / 2,
            self.env_config["arenaSize"]["width"] / 2,
        )
        gy = self.rand.uniform(
            -self.env_config["arenaSize"]["height"] / 2,
            self.env_config["arenaSize"]["height"] / 2,
        )
        return Goal(gx, gy)

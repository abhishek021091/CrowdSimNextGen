"""RobotBuilder: places the robot's episode start pose and goal.

Takes its random generator as a constructor argument rather than
seeding one internally. Previously this class seeded a class-level
generator once at import time, shared across every instance -- which
meant repeated construction within one process (e.g. one episode reset
after another) kept advancing the *same* stream regardless of what
seed the caller thought they were using, breaking end-to-end
reproducibility from a single top-level seed. Injecting ``rand``
mirrors the fix already applied to ``CrowdSpawner``.
"""

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

    def __init__(self, rand: np.random.Generator) -> None:
        self.rand = rand
        self.robot: Robot | None = None

    def build_robot(self) -> None:
        robot = Robot()
        robot.set_state(
            self.generate_pose(), self.generate_goal(), robot.v_pref, robot.radius
        )
        self.robot = robot

    def generate_pose(self) -> Pose:
        theta: float = self.rand.uniform(0, 2 * np.pi)
        width: float = self.env_config["arenaSize"]["width"]
        height: float = self.env_config["arenaSize"]["height"]
        px: float = self.rand.uniform(-width / 2, width / 2)
        py: float = self.rand.uniform(-height / 2, height / 2)
        return Pose(px, py, theta)

    def generate_goal(self) -> Goal:
        width: float = self.env_config["arenaSize"]["width"]
        height: float = self.env_config["arenaSize"]["height"]
        gx: float = self.rand.uniform(-width / 2, width / 2)
        gy: float = self.rand.uniform(-height / 2, height / 2)
        return Goal(gx, gy)

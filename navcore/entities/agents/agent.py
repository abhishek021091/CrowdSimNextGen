from pathlib import Path
from typing import Any

import tomllib

import navcore.configs
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.components.state import FullState, ObservableState
from navcore.entities.components.velocity import Velocity


class Agent:
    def __init__(self, config: dict[str, Any], agent_type: str):
        self.agent: str = agent_type
        self.config: dict[str, Any] = config

        self.v_pref: float = self.config["kinematics"]["v_pref"]
        self.radius: float = self.config["physical"]["radius"]

        self.pose: Pose | None = None
        self.goal: Goal | None = None
        self.velocity: Velocity | None = None

        assert navcore.configs.__file__ is not None
        env_path = Path(navcore.configs.__file__).parent / "env.toml"
        with open(Path(env_path), "rb") as f:
            self.env_config = tomllib.load(f)

        self.time_step = self.env_config["policy"]["time_step"]

    def set_state(
        self,
        pose: Pose,
        goal: Goal,
        v_pref: float,
        radius: float,
    ):
        self.pose = pose
        self.goal = goal
        self.radius = radius
        self.v_pref = v_pref

    def get_observable_state(self) -> ObservableState:
        if self.pose is None or self.velocity is None:
            raise RuntimeError("Pose or velocity has not been initialized.")

        return ObservableState(self.pose, self.velocity, self.radius)

    def get_observable_state_list(self) -> ObservableState:
        if self.pose is None or self.velocity is None:
            raise RuntimeError("Pose or velocity has not been initialized.")
        return ObservableState(
            pose=self.pose, velocity=self.velocity, radius=self.radius
        )

    def get_full_state(self) -> FullState:
        if self.pose is None or self.velocity is None or self.goal is None:
            raise RuntimeError("Pose or velocity has not been initialized.")
        return FullState(self.pose, self.goal, self.velocity, self.radius, self.v_pref)

    def get_position(self) -> Pose:
        if self.pose is None:
            raise RuntimeError("Pose has not been initialized.")
        return self.pose

    def set_position(self, px: float, py: float):
        if self.pose is None:
            raise RuntimeError("Pose has not been initialized.")
        self.pose = Pose(px, py, self.pose.theta)

    def get_goal_position(self) -> Goal:
        if self.goal is None:
            raise RuntimeError("Goal has not been initialized.")
        return self.goal

    def set_goal_position(self, position: tuple[float, float]):
        self.goal = Goal(position[0], position[1])

    def get_velocity(self) -> Velocity:
        if self.velocity is None:
            raise RuntimeError("Velocity has not been initialized.")
        return self.velocity

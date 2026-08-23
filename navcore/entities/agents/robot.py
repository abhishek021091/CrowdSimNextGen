from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.entities.agents.agent import Agent


class Robot(Agent):
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    config_path = Path(navcore.configs.__file__).parent / "robot.toml"
    with open(Path(config_path), "rb") as f:
        config = tomllib.load(f)
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)

    def __init__(self):
        super().__init__(self.config, "Robot")
        self.sensor_range = self.config["sensors"]["sensor_range"]
        self.sensor_fov = np.pi * self.config["sensors"]["sensor_fov"]
        self.kinematics = self.config["kinematics"]
        self.policy = self.config["policy"]
        self.observable = self.config["physical"]["observable"]
        self.v_pref = self.config["kinematics"]["v_pref"]
        self.radius = self.config["physical"]["radius"]
        self.sweep = self.config["sweep"]
        if self.sweep:
            self.sweep_axis = self.config["sweep_axis"]
        self.observable = self.config["physical"]["observable"]

    def print_info(self) -> None:
        if self.pose is None or self.velocity is None or self.goal is None:
            raise RuntimeError("Pose or velocity has not been initialized.")

        print(
            "Robot: px=%s, py=%s, gx=%s, gy=%s, "
            "vx=%s, vy=%s, theta=%s, radius=%s, v_pref=%s",
            self.pose.px,
            self.pose.py,
            self.goal.gx,
            self.goal.gy,
            self.velocity.vx,
            self.velocity.vy,
            self.pose.theta,
            self.radius,
            self.v_pref,
        )

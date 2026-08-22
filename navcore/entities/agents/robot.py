from pathlib import Path

import numpy as np
import tomllib

from navcore import configs
from navcore.entities.agents.agent import Agent


class Robot(Agent):
    def __init__(self):
        with open(Path(configs / "robot.toml"), "rb") as f:
            self.config = tomllib.load(f)
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

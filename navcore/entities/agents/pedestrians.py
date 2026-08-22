import logging
from pathlib import Path

import tomllib

from navcore import configs
from navcore.entities.agents.agent import Agent

logger = logging.getLogger(__name__)


class Pedestrian(Agent):
    def __init__(self):
        with open(Path(configs / "pedestrian.toml"), "rb") as f:
            self.config = tomllib.load(f)
        super().__init__(self.config, "Pedestrian")
        self.rand = self.random.rand(seed=self.env_config["random"]["seed"])
        self.id = None
        self.observed_id = -1
        self.kinematics = self.config["kinematics"]["chassis"]
        self.policy = self.config["policy"]
        if self.env_config["randomize_pedestrian_radius"]:
            self.radius *= self.rng.uniform(
                self.config["physical"]["radius"] * 0.8,
                self.config["physical"]["radius"] * 1.2,
            )
        if self.env_config["randomize_pedestrian_v_pref"]:
            self.v_pref *= self.rng.uniform(
                self.config["kinematics"]["v_pref"] * 0.8,
                self.config["kinematics"]["v_pref"] * 1.2,
            )
        else:
            self.radius = self.config["physical"]["radius"]
            self.v_pref = self.config["kinematics"]["v_pref"]

    def set_id(self, id):
        self.id = id

    def print_info(self):
        logger.info(
            f"Pedestrian {self.id}: px={self.px}, py={self.py}, gx={self.gx}, gy={self.gy}, vx={self.vx}, vy={self.vy}, theta={self.theta}, radius={self.radius}, v_pref={self.v_pref}"
        )

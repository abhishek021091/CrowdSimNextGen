import logging
from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.entities.agents.agent import Agent

logger = logging.getLogger(__name__)


class Pedestrian(Agent):
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    config_path = Path(navcore.configs.__file__).parent / "pedestrians.toml"
    with open(Path(config_path), "rb") as f:
        config = tomllib.load(f)
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)
    rand = np.random.default_rng(seed=env_config["random"]["seed"])

    def __init__(self) -> None:
        super().__init__(self.config, "Pedestrian")
        self.id = None
        self.observed_id = -1
        self.kinematics = self.config["kinematics"]["chassis"]
        self.policy = self.config["policy"]

        if self.config["Randomization"]["randomize_pedestrian_radius"]:
            self.radius *= self.rand.uniform(
                self.config["physical"]["radius"] * 0.8,
                self.config["physical"]["radius"] * 1.2,
            )
        if self.config["Randomization"]["randomize_pedestrian_v_pref"]:
            self.v_pref *= self.rand.uniform(
                self.config["kinematics"]["v_pref"] * 0.8,
                self.config["kinematics"]["v_pref"] * 1.2,
            )
        else:
            self.radius = self.config["physical"]["radius"]
            self.v_pref = self.config["kinematics"]["v_pref"]

    def set_id(self, id: int) -> None:
        self.id = id

    def print_info(self) -> None:
        if self.pose is None or self.velocity is None or self.goal is None:
            raise RuntimeError("Pose or velocity has not been initialized.")

        logger.info(
            "Pedestrian %s: px=%s, py=%s, gx=%s, gy=%s, "
            "vx=%s, vy=%s, theta=%s, radius=%s, v_pref=%s",
            self.id,
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

    def __repr__(self) -> str:
        return (
            f"Pedestrian("
            f"pose={self.pose}, "
            f"goal={self.goal}, "
            f"velocity={self.velocity}, "
            f"radius={self.radius}, "
            f"v_pref={self.v_pref})"
        )

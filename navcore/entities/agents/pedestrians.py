import logging
from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.entities.agents.agent import Agent
from navcore.entities.groups.group import Group

logger = logging.getLogger(__name__)


class Pedestrian(Agent):
    """A crowd member.

    ``rand`` is injected at construction, not seeded internally -- the
    same fix already applied to ``CrowdSpawner``/``RobotBuilder``/
    ``ObstacleBuilder``. A previous version of this class seeded its
    own class-level generator once at import time, shared across every
    instance; radius/v_pref randomization therefore advanced regardless
    of which seed a caller thought it was using, silently breaking
    end-to-end reproducibility for any run with randomization enabled.
    """

    assert navcore.configs.__file__ is not None
    config_path = Path(navcore.configs.__file__).parent / "pedestrians.toml"
    with open(Path(config_path), "rb") as f:
        config = tomllib.load(f)

    def __init__(self, rand: np.random.Generator) -> None:
        super().__init__(self.config, "Pedestrian")
        self.rand = rand
        self.observed_id = -1
        self.group: Group | None = None
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
            "Pedestrian %s: px=%s, py=%s, gx=%s, gy=%s, vx=%s, vy=%s, theta=%s, radius=%s, v_pref=%s",
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
            f"Pedestrian(id ={self.id}, pose={self.pose}, goal={self.goal}, "
            f"velocity={self.velocity}, radius={self.radius}, v_pref={self.v_pref})"
        )

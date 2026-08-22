from pathlib import Path

import tomllib

from navcore import configs
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.components.state import FullState, ObservableState


class Agent:
    def __init__(self, config, agent_type):
        self.agent = agent_type
        self.config = config

        with open(Path(configs / "env.toml"), "rb") as f:
            self.env_config = tomllib.load(f)

        self.v_pref = self.config["v_pref"]
        self.radius = self.config["radius"]
        self.pose = None
        self.velocity = None
        self.time_step = self.env_config["time_step"]
        self.policy.time_step = self.env_config["time_step"]

    def set_state(self, px, py, theta, gx, gy, radius, v_pref):
        self.pose = Pose(px, py, theta)
        self.goal = Goal(gx, gy)
        self.radius = radius
        self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(
            self.pose.px, self.pose.py, self.velocity.vx, self.velocity.vy, self.radius
        )

    def get_observable_state_list(self):
        return [
            self.pose.px,
            self.pose.py,
            self.velocity.vx,
            self.velocity.vy,
            self.radius,
        ]

    def get_full_state(self):
        return FullState(
            px=self.pose.px,
            py=self.pose.py,
            vx=self.velocity.vx,
            vy=self.velocity.vy,
            radius=self.radius,
            gx=self.goal.gx,
            gy=self.goal.gy,
            preferred_speed=self.v_pref,
            theta=self.pose.theta,
        )

    def get_position(self):
        return self.pose.px, self.pose.py

    def set_position(self, px, py):
        self.pose = Pose(px, py, self.pose.theta)

    def get_goal_position(self):
        return self.goal.gx, self.goal.gy

    def set_goal_position(self, position):
        self.goal = Goal(*position)

    def get_velocity(self):
        return self.velocity.vx, self.velocity.vy

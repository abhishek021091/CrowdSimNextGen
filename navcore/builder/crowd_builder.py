from pathlib import Path

import numpy as np
import tomllib

import navcore.configs
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.groups.group import Group
from navcore.missions.group_goal_reaching import GroupGoalReachingMission


class CrowdBuilder:
    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    config_path = Path(navcore.configs.__file__).parent / "pedestrians.toml"
    with open(Path(config_path), "rb") as f:
        config = tomllib.load(f)
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)
    rand = np.random.default_rng(seed=env_config["random"]["seed"])

    def __init__(self):
        self.pedestrian_num = self.env_config["pedestrians"]["num_pedestrians"]
        self.groups: dict[int, Group] = {}
        self.crowd: dict[int, Pedestrian] = {}

    def build_crowd(self):
        for i in range(self.pedestrian_num):
            pedestrian = Pedestrian(self.rand)
            pedestrian.set_id(i)
            pedestrian.set_state(
                self.generate_pose(),
                self.generate_goal(),
                pedestrian.v_pref,
                pedestrian.radius,
            )
            self.crowd[i] = pedestrian

    def build_groups(self) -> None:
        group_size = self.env_config["pedestrians"]["group_size"]
        num_groups = self.env_config["pedestrians"]["num_groups"]

        crowd_list = list(self.crowd.values())
        # Build a lookup once
        agent_lookup = {ped.id: ped for ped in crowd_list}.__getitem__

        for i in range(num_groups):
            group_members = crowd_list[i * group_size : (i + 1) * group_size]

            group = Group(
                id=i,
                member_ids=tuple(member.id for member in group_members),
                leader_id=group_members[0].id,
                goal=group_members[0].get_goal_position(),
            )

            for member in group_members:
                member.group = group
                member = GroupGoalReachingMission(
                    agent_id=member.id,
                    group=group,
                    agent_lookup=agent_lookup,
                )
                member.set_pos()
                self.groups[i] = group

    def generate_pose(self) -> Pose:
        theta = self.rand.uniform(0, 2 * np.pi)

        sides: list[tuple[float, float]] = [
            (self.env_config["arenaSize"]["width"] / 2, 0),
            (-self.env_config["arenaSize"]["width"] / 2, 0),
            (0, self.env_config["arenaSize"]["height"] / 2),
            (0, -self.env_config["arenaSize"]["height"] / 2),
        ]

        side = sides[self.rand.integers(len(sides))]

        px = side[0] + np.cos(theta) + np.random.choice([-0.5, 0.5])
        py = side[1] + np.sin(theta) + np.random.choice([-0.5, 0.5])
        return Pose(px, py, theta)

    def generate_goal(self) -> Goal:
        theta = self.rand.uniform(0, 2 * np.pi)
        sides: list[tuple[float, float]] = [
            (self.env_config["arenaSize"]["width"] / 2, 0),
            (-self.env_config["arenaSize"]["width"] / 2, 0),
            (0, self.env_config["arenaSize"]["height"] / 2),
            (0, -self.env_config["arenaSize"]["height"] / 2),
        ]

        side = sides[self.rand.integers(len(sides))]

        gx = side[0] + np.cos(theta) + np.random.choice([-0.5, 0.5])
        gy = side[1] + np.sin(theta) + np.random.choice([-0.5, 0.5])
        return Goal(gx, gy)

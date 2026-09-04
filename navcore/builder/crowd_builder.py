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

    def __init__(self, rand: np.random.Generator | None = None):
        self.rand = (
            rand
            if rand is not None
            else np.random.default_rng(seed=self.env_config["random"]["seed"])
        )
        self.pedestrian_num = self.env_config["pedestrians"]["num_pedestrians"]
        self.groups: dict[int, Group] = {}
        self.crowd: dict[int, Pedestrian] = {}

    def build_crowd(self):
        for i in range(self.pedestrian_num):
            pedestrian = Pedestrian(self.rand)
            pedestrian.set_id(i)
            self.pose = self.generate_pose()
            self.goal = self.generate_goal()
            pedestrian.set_state(
                self.pose,
                self.goal,
                pedestrian.v_pref,
                pedestrian.radius,
            )
            self.crowd[i] = pedestrian
        return self.crowd

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
                member.set_position()
                member.set_goal()
                self.groups[i] = group

    def generate_pose(self) -> Pose:
        theta = self.rand.uniform(0, 2 * np.pi)
        width = self.env_config["arenaSize"]["width"]
        height = self.env_config["arenaSize"]["height"]

        edge = int(self.rand.integers(4))
        if edge == 0:  # north
            px, py = (
                self.rand.uniform((-width - 2) / 2, (width + 2) / 2),
                (height + 2) / 2,
            )
        elif edge == 1:  # south
            px, py = (
                self.rand.uniform((-width - 2) / 2, (width + 2) / 2),
                (-height - 2) / 2,
            )
        elif edge == 2:  # east
            px, py = (
                (width + 2) / 2,
                self.rand.uniform((-height - 2) / 2, (height + 2) / 2),
            )
        else:  # west
            px, py = (
                (-width - 2) / 2,
                self.rand.uniform((-height - 2) / 2, (height + 2) / 2),
            )

        return Pose(px, py, theta)

    def generate_goal(self, method: str = "opposite") -> Goal:
        if method == "opposite":
            goal: Goal = np.random.choice(
                [
                    Goal(-self.pose.px, -self.pose.py),
                    Goal(-self.pose.px, self.pose.py),
                    Goal(self.pose.px, -self.pose.py),
                ]
            )
            return goal
        else:
            theta = self.rand.uniform(0, 2 * np.pi)
            sides: list[tuple[float, float]] = [
                (
                    -self.env_config["arenaSize"]["width"] / 2,
                    -self.env_config["arenaSize"]["height"] / 2,
                ),
                (
                    -self.env_config["arenaSize"]["width"] / 2,
                    self.env_config["arenaSize"]["height"] / 2,
                ),
                (
                    self.env_config["arenaSize"]["width"] / 2,
                    self.env_config["arenaSize"]["height"] / 2,
                ),
                (
                    self.env_config["arenaSize"]["width"] / 2,
                    -self.env_config["arenaSize"]["height"] / 2,
                ),
            ]

            side = sides[self.rand.integers(len(sides))]

            gx = side[0] + np.cos(theta) + np.random.choice([0.0, 0.5])
            gy = side[1] + np.sin(theta) + np.random.choice([0.0, 0.5])
            return Goal(gx, gy)

    def build_single_pedestrian(self, ped_id: int) -> Pedestrian | None:
        if ped_id in self.crowd:
            return None  # Pedestrian with this ID already exists

        pedestrian = Pedestrian(self.rand)
        pedestrian.set_id(ped_id)
        self.pose = self.generate_pose()
        self.goal = self.generate_goal()
        pedestrian.set_state(
            self.pose,
            self.goal,
            pedestrian.v_pref,
            pedestrian.radius,
        )
        self.crowd[ped_id] = pedestrian
        return pedestrian

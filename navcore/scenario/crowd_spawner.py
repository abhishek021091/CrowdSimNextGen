"""CrowdSpawner: keeps a mall/hospital/house population alive for an event.

Unlike ``CrowdBuilder`` (which builds a fixed crowd once, at the start
of a run), ``CrowdSpawner`` treats the pedestrian population as a
living, self-maintaining thing: individuals and groups arrive, walk to
their goal (or time out), and leave -- continuously, for as long as the
robot's episode runs. What arrives and in what mix (solo vs. group, how
many, how variable) is driven entirely by the active ``PopulationStage``
of an injected ``ScenarioConfig``; the spawner itself has no opinion
about "quiet morning" vs. "lunch rush", it just reads whichever stage
is active for the current episode-relative time and steers the
population toward that stage's target.

The robot is deliberately not this class's concern. Pedestrians here
are environmental conditions the robot has to navigate, not individual
subjects of interest -- their arrival/departure is population
bookkeeping, not an episode-level outcome. See ``navcore.simulation``
for where episode outcomes (which *are* robot-centric) are decided.

All identifiers issued by this class are strings, to stay consistent
with ``Group.member_ids`` and ``GroupGoalReachingMission``'s
``AgentLookup``, both of which are keyed by ``str``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.components.velocity import Velocity
from navcore.entities.groups.group import Group
from navcore.missions.goal_reaching import GoalReachingMission
from navcore.missions.group_goal_reaching import GroupGoalReachingMission
from navcore.scenario.scenario_config import PopulationStage, ScenarioConfig

if TYPE_CHECKING:
    from navcore.missions.mission import Mission


@dataclass(slots=True, frozen=True)
class SpawnedAgent:
    """A newly spawned pedestrian, paired with the mission driving it.

    Callers are expected to register both under the same agent id in
    whatever registry they use to resolve (agent -> mission) each tick.
    """

    pedestrian: Pedestrian
    mission: "Mission"


@dataclass(slots=True)
class PopulationDelta:
    """Net population changes produced by one ``maintain_population()`` call."""

    spawned: list[SpawnedAgent] = field(default_factory=list)
    spawned_groups: list[Group] = field(default_factory=list)
    despawned_ids: list[str] = field(default_factory=list)


class CrowdSpawner:
    """Maintains a self-regenerating pedestrian population for one episode.

    Attributes:
        env_config: The environment config dict (arena size), used to
            place spawns/goals along the arena's edges, mirroring
            ``CrowdBuilder``'s existing edge-spawn placement strategy.
        scenario: The population recipe for the current episode.
        rand: The random generator this spawner draws from. Injected,
            not owned -- callers control seeding and can keep a single
            reproducible stream advancing across an entire training
            run rather than resetting it every episode.
    """

    _JITTER = (
        0.5  # meters, edge-spawn placement noise (matches prior CrowdBuilder behavior)
    )
    _GROUP_CLUSTER_RADIUS = 0.8  # meters, how tightly a group clusters at spawn

    def __init__(
        self, env_config: dict, scenario: ScenarioConfig, rand: np.random.Generator
    ) -> None:
        self.env_config = env_config
        self.scenario = scenario
        self.rand = rand

        self._goal_reaching_mission = GoalReachingMission()
        self._pedestrians: dict[str, Pedestrian] = {}
        self._missions: dict[str, "Mission"] = {}
        self._member_group: dict[str, str] = {}
        self._ages: dict[str, float] = {}
        self._active_stage: PopulationStage | None = None
        self._current_target: int = 0
        self._next_id = 0

    def reset(self) -> None:
        """Clear all tracked population state at the start of a new episode."""
        self._pedestrians.clear()
        self._missions.clear()
        self._member_group.clear()
        self._ages.clear()
        self._active_stage = None
        self._current_target = 0

    def spawn_initial_population(self) -> PopulationDelta:
        """Return the population that should exist at ``episode_time=0``.

        Equivalent to :meth:`maintain_population` with zero elapsed
        time -- there is no separate "initial" code path to keep in
        sync with the ongoing one.
        """
        return self.maintain_population(episode_time=0.0, dt=0.0)

    def maintain_population(self, episode_time: float, dt: float) -> PopulationDelta:
        """Advance population state by ``dt`` and return the net changes.

        Args:
            episode_time: Elapsed time since the current episode began,
                in seconds. Used to look up the active
                ``PopulationStage``.
            dt: Time elapsed since the previous call, in seconds. Used
                to age tracked pedestrians for the lifetime safety net.

        Returns:
            A ``PopulationDelta`` describing exactly what was spawned
            and despawned this call. Callers are responsible for
            folding this into their own agent/mission/backend/render
            registries -- this class only tracks what's alive, not how
            each pedestrian actually moves.
        """
        delta = PopulationDelta()
        stage = self.scenario.stage_at(episode_time)
        if stage is not self._active_stage:
            self._active_stage = stage
            self._current_target = self._sample_target(stage)

        self._age_and_expire(dt, delta)
        self._despawn_arrivals(delta)
        self._spawn_to_target(stage, delta)
        return delta

    # -- internal: lifecycle ---------------------------------------------------

    def _age_and_expire(self, dt: float, delta: PopulationDelta) -> None:
        """Age every tracked pedestrian; force-despawn if past its lifetime."""
        max_lifetime = self.scenario.pedestrian_max_lifetime
        for pid in list(self._ages):
            self._ages[pid] += dt
        if max_lifetime is None:
            return
        expired = [pid for pid, age in self._ages.items() if age >= max_lifetime]
        for pid in expired:
            self._despawn_one(pid, delta)

    def _despawn_arrivals(self, delta: PopulationDelta) -> None:
        """Despawn agents that reached their target.

        A solo pedestrian is despawned once it is within its own
        radius of its current mission target. A group is despawned
        only once *every* member has arrived -- groups leave together,
        the same way they arrived together.
        """
        arrived_group_ids: set[str] = set()
        for group_id, member_ids in self._members_by_group().items():
            if all(self._has_arrived(mid) for mid in member_ids):
                arrived_group_ids.add(group_id)

        arrived_solo = [
            pid
            for pid, group_id in (
                (pid, self._member_group.get(pid)) for pid in self._pedestrians
            )
            if group_id is None and self._has_arrived(pid)
        ]

        for pid in arrived_solo:
            self._despawn_one(pid, delta)
        for group_id in arrived_group_ids:
            members = [
                pid for pid, gid in self._member_group.items() if gid == group_id
            ]
            for pid in members:
                self._despawn_one(pid, delta)

    def _has_arrived(self, pedestrian_id: str) -> bool:
        pedestrian = self._pedestrians[pedestrian_id]
        mission = self._missions[pedestrian_id]
        assert pedestrian.pose is not None
        target = mission.get_target(pedestrian, [])
        position = Vector2(pedestrian.pose.px, pedestrian.pose.py)
        return position.distance_to(target) <= pedestrian.radius

    def _despawn_one(self, pedestrian_id: str, delta: PopulationDelta) -> None:
        if pedestrian_id not in self._pedestrians:
            return
        del self._pedestrians[pedestrian_id]
        self._missions.pop(pedestrian_id, None)
        self._member_group.pop(pedestrian_id, None)
        self._ages.pop(pedestrian_id, None)
        delta.despawned_ids.append(pedestrian_id)

    def _members_by_group(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for pid, group_id in self._member_group.items():
            groups.setdefault(group_id, []).append(pid)
        return groups

    # -- internal: spawning -----------------------------------------------------

    def _sample_target(self, stage: PopulationStage) -> int:
        if stage.spawn_mode == "fixed":
            return stage.num_people
        sampled = self.rand.normal(stage.num_people, stage.std)
        return max(0, round(sampled))

    def _spawn_to_target(self, stage: PopulationStage, delta: PopulationDelta) -> None:
        if (
            stage.spawn_mode == "random"
            and len(self._pedestrians) >= self._current_target
        ):
            self._current_target = self._sample_target(stage)

        deficit = self._current_target - len(self._pedestrians)
        while deficit > 0:
            if self.rand.random() < stage.group_ratio:
                size = int(
                    self.rand.integers(stage.group_size_min, stage.group_size_max + 1)
                )
                self._spawn_group(size, delta)
                deficit -= size
            else:
                self._spawn_solo(delta)
                deficit -= 1

    def _spawn_solo(self, delta: PopulationDelta) -> None:
        pedestrian = Pedestrian(self.rand)
        pid = self._issue_id()
        pedestrian.set_id(pid)
        pedestrian.set_state(
            self._generate_pose(),
            self._generate_goal(),
            pedestrian.v_pref,
            pedestrian.radius,
        )
        pedestrian.velocity = Velocity(0.0, 0.0)
        self._pedestrians[pid] = pedestrian
        self._missions[pid] = self._goal_reaching_mission
        self._ages[pid] = 0.0
        delta.spawned.append(SpawnedAgent(pedestrian, self._goal_reaching_mission))

    def _spawn_group(self, size: int, delta: PopulationDelta) -> None:
        group_id = f"group_{self._issue_id()}"
        shared_goal = self._generate_goal()
        leader_pose = self._generate_pose()

        member_ids: list[str] = []
        pedestrians: list[Pedestrian] = []
        for i in range(size):
            pedestrian = Pedestrian(self.rand)
            pid = self._issue_id()
            pedestrian.set_id(pid)
            if i == 0:
                pose = leader_pose
            else:
                jitter_x = self.rand.uniform(
                    -self._GROUP_CLUSTER_RADIUS, self._GROUP_CLUSTER_RADIUS
                )
                jitter_y = self.rand.uniform(
                    -self._GROUP_CLUSTER_RADIUS, self._GROUP_CLUSTER_RADIUS
                )
                pose = Pose(
                    leader_pose.px + jitter_x,
                    leader_pose.py + jitter_y,
                    leader_pose.theta,
                )
            pedestrian.set_state(
                pose, shared_goal, pedestrian.v_pref, pedestrian.radius
            )
            pedestrian.velocity = Velocity(0.0, 0.0)
            self._pedestrians[pid] = pedestrian
            self._ages[pid] = 0.0
            self._member_group[pid] = group_id
            member_ids.append(pid)
            pedestrians.append(pedestrian)

        group = Group(
            id=group_id,
            member_ids=tuple(member_ids),
            goal=Vector2(shared_goal.gx, shared_goal.gy),
            leader_id=member_ids[0],
        )
        delta.spawned_groups.append(group)

        for i, (pid, pedestrian) in enumerate(zip(member_ids, pedestrians)):
            if i == 0:
                mission: "Mission" = GroupGoalReachingMission(
                    agent_id=pid,
                    group=group,
                    agent_lookup=self._pedestrians.__getitem__,
                )
            else:
                offset = Vector2(-0.5 * i, 0.0)
                mission = GroupGoalReachingMission(
                    agent_id=pid,
                    group=group,
                    agent_lookup=self._pedestrians.__getitem__,
                    formation_offset=offset,
                )
            self._missions[pid] = mission
            delta.spawned.append(SpawnedAgent(pedestrian, mission))

    def _issue_id(self) -> str:
        pid = f"ped_{self._next_id}"
        self._next_id += 1
        return pid

    # -- internal: placement (mirrors CrowdBuilder's edge-spawn strategy) -------

    def _generate_pose(self) -> Pose:
        theta = self.rand.uniform(0, 2 * math.pi)
        side = self._random_edge_point()
        px = side[0] + math.cos(theta) + self.rand.choice([-self._JITTER, self._JITTER])
        py = side[1] + math.sin(theta) + self.rand.choice([-self._JITTER, self._JITTER])
        return Pose(px, py, theta)

    def _generate_goal(self) -> Goal:
        theta = self.rand.uniform(0, 2 * math.pi)
        side = self._random_edge_point()
        gx = side[0] + math.cos(theta) + self.rand.choice([-self._JITTER, self._JITTER])
        gy = side[1] + math.sin(theta) + self.rand.choice([-self._JITTER, self._JITTER])
        return Goal(gx, gy)

    def _random_edge_point(self) -> tuple[float, float]:
        width = self.env_config["arenaSize"]["width"]
        height = self.env_config["arenaSize"]["height"]
        sides: list[tuple[float, float]] = [
            (width / 2, 0.0),
            (-width / 2, 0.0),
            (0.0, height / 2),
            (0.0, -height / 2),
        ]
        index = int(self.rand.integers(len(sides)))
        return sides[index]

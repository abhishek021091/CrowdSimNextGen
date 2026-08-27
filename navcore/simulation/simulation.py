"""Simulation: owns the tick loop, and every episode-level decision.

``Simulation`` is the only place episode outcomes are decided -- and it
decides them for the *robot* only. A pedestrian reaching its goal is
population bookkeeping (see ``CrowdSpawner``); it never ends an
episode. This split exists because the robot is the subject of the
research -- pedestrians are environmental conditions it has to
contend with, not participants in the outcome being measured.

Each call to :meth:`Simulation.reset` rebuilds the static scene
(obstacles + robot start/goal, via the injected ``scene_factory``) and
restarts the population from stage 0 of ``scenario`` -- using whatever
random draws the injected, externally-owned ``rand`` generator
produces next, so consecutive episodes differ from each other while
the entire run stays reproducible from one seed.

Per-tick ordering is deliberately split into distinct phases (compute
every agent's command against last tick's state -> apply commands ->
advance every distinct backend once -> read results back) rather than
updating agents one at a time. Interleaving compute-and-apply per
agent would make results depend on iteration order -- an "earlier
agents get a first-mover advantage" bug that is easy to introduce and
hard to notice.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from navcore.entities.agents.robot import Robot
from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.info import (
    Collision,
    Danger,
    Nothing,
    OutRoad,
    ReachGoal,
    Timeout,
)
from navcore.entities.components.pose import Pose
from navcore.entities.components.velocity import Velocity
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.environment.neighbor_query import NeighborQuery
from navcore.execution.execution_backend import ExecutionBackend
from navcore.execution.kinematic_backend import NavCoreKinematicBackend
from navcore.rendering.render_frame import RenderFrame
from navcore.rendering.renderer import Renderer
from navcore.scenario.crowd_spawner import CrowdSpawner
from navcore.scenario.scenario_config import ScenarioConfig

if TYPE_CHECKING:
    from navcore.missions.mission import Mission
    from navcore.policies.policy import Policy

#: Placeholder pending a proper per-pedestrian sensor config (pedestrians.toml
#: has no [sensors] section today, unlike robot.toml). Documented here rather
#: than left as a bare literal elsewhere in this module.
_PEDESTRIAN_PERCEPTION_RADIUS = 5.0

EpisodeOutcome = ReachGoal | Collision | Timeout | OutRoad | Danger | Nothing

#: Outcomes that end the current episode and trigger an automatic reset.
_TERMINAL_OUTCOMES = (ReachGoal, Collision, Timeout, OutRoad)

SceneFactory = Callable[[np.random.Generator], tuple[tuple[Obstacle, ...], Robot]]
RobotBackendFactory = Callable[[Pose], ExecutionBackend]


class Simulation:
    """Runs one robot's episode against a continuously-regenerating crowd.

    Attributes:
        dt: Fixed simulation timestep, in seconds. ``Simulation`` uses
            a fixed-step clock throughout; a wall-clock-driven mode
            (required once a real-hardware ``ExecutionBackend`` is
            attached, since real time can't be paused and stepped by
            an exact amount) is a deliberately separate concern, not
            implemented here.
        max_episode_time: Episode length, in seconds, after which the
            robot's outcome is ``Timeout`` if nothing else has ended
            the episode first.
        danger_margin: Robot-to-pedestrian clearance, in meters, below
            which (and above zero, i.e. not yet a collision) the
            robot's outcome for that tick is ``Danger``.
    """

    def __init__(
        self,
        scene_factory: SceneFactory,
        scenario: ScenarioConfig,
        env_config: dict,
        rand: np.random.Generator,
        robot_mission: "Mission",
        robot_policy: "Policy",
        pedestrian_policy: "Policy",
        neighbor_query: NeighborQuery,
        dt: float,
        max_episode_time: float,
        renderer: Renderer | None = None,
        robot_backend_factory: RobotBackendFactory | None = None,
        danger_margin: float = 0.2,
    ) -> None:
        self._scene_factory = scene_factory
        self._scenario = scenario
        self._env_config = env_config
        self._rand = rand
        self._robot_mission = robot_mission
        self._robot_policy = robot_policy
        self._pedestrian_policy = pedestrian_policy
        self._neighbor_query = neighbor_query
        self.dt = dt
        self.max_episode_time = max_episode_time
        self._renderer = renderer
        self._robot_backend_factory: RobotBackendFactory = (
            robot_backend_factory or NavCoreKinematicBackend
        )
        self.danger_margin = danger_margin

        self._crowd_spawner = CrowdSpawner(env_config, scenario, rand)

        self._obstacles: tuple[Obstacle, ...] = ()
        self._robot: Robot | None = None
        self._robot_backend: ExecutionBackend | None = None
        self._pedestrians: dict[str, Pedestrian] = {}
        self._pedestrian_missions: dict[str, "Mission"] = {}
        self._pedestrian_backends: dict[str, ExecutionBackend] = {}

        self._episode_time = 0.0
        self._step_count = 0
        self._episode_index = -1

        self.reset()

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @property
    def episode_time(self) -> float:
        return self._episode_time

    @property
    def robot(self) -> Robot:
        assert self._robot is not None
        return self._robot

    @property
    def pedestrians(self) -> dict[str, Pedestrian]:
        return self._pedestrians

    def reset(self) -> None:
        """Start a fresh episode: new scene, population reset to stage 0."""
        self._obstacles, self._robot = self._scene_factory(self._rand)
        assert self._robot.pose is not None
        self._robot.velocity = Velocity(0.0, 0.0)
        self._robot_backend = self._robot_backend_factory(self._robot.pose)

        self._crowd_spawner.reset()
        self._pedestrians.clear()
        self._pedestrian_missions.clear()
        for backend in self._pedestrian_backends.values():
            backend.close()
        self._pedestrian_backends.clear()
        self._apply_population_delta(self._crowd_spawner.spawn_initial_population())

        self._episode_time = 0.0
        self._step_count = 0
        self._episode_index += 1

        if self._renderer is not None:
            self._renderer.reset(self._episode_index)

    def close(self) -> None:
        """Release the renderer and every active backend."""
        if self._robot_backend is not None:
            self._robot_backend.close()
        for backend in self._pedestrian_backends.values():
            backend.close()
        if self._renderer is not None:
            self._renderer.close()

    def step(self) -> EpisodeOutcome:
        """Advance the simulation by one tick.

        Returns:
            The robot's outcome for this tick -- one of ``ReachGoal``,
            ``Collision``, ``Timeout``, ``OutRoad``, ``Danger``, or
            ``Nothing`` (see ``entities.components.info``). If the
            outcome is one of the four terminal ones, :meth:`reset`
            has already been called before this method returns.
        """
        assert self._robot is not None and self._robot_backend is not None
        all_agents: list = [self._robot, *self._pedestrians.values()]

        # Phase 1: compute every agent's command against last tick's state.
        robot_neighbors = self._neighbor_query.neighbors_of(
            self._robot, all_agents, radius=self._robot.sensor_range
        )
        robot_target = self._robot_mission.get_target(self._robot, robot_neighbors)
        robot_velocity = self._robot_policy.compute_velocity(
            self._robot, robot_target, robot_neighbors
        )

        pedestrian_velocities: dict[str, Velocity] = {}
        for pid, pedestrian in self._pedestrians.items():
            neighbors = self._neighbor_query.neighbors_of(
                pedestrian, all_agents, radius=_PEDESTRIAN_PERCEPTION_RADIUS
            )
            target = self._pedestrian_missions[pid].get_target(pedestrian, neighbors)
            pedestrian_velocities[pid] = self._pedestrian_policy.compute_velocity(
                pedestrian, target, neighbors
            )

        # Phase 2: apply every command, then advance every distinct backend once.
        self._robot_backend.apply_command(robot_velocity)
        for pid, velocity in pedestrian_velocities.items():
            self._pedestrian_backends[pid].apply_command(velocity)

        distinct_backends = {self._robot_backend, *self._pedestrian_backends.values()}
        for backend in distinct_backends:
            backend.advance(self.dt)

        # Phase 3: read results back.
        self._robot.pose = self._robot_backend.read_pose()
        self._robot.velocity = self._robot_backend.read_velocity()
        for pid, pedestrian in self._pedestrians.items():
            backend = self._pedestrian_backends[pid]
            pedestrian.pose = backend.read_pose()
            pedestrian.velocity = backend.read_velocity()

        self._episode_time += self.dt
        self._step_count += 1

        # Phase 4: population maintenance. Never ends the episode.
        delta = self._crowd_spawner.maintain_population(self._episode_time, self.dt)
        self._apply_population_delta(delta)

        # Phase 5: robot-centric episode outcome.
        outcome = self._robot_outcome()

        # Phase 6: render, then handle episode boundary.
        if self._renderer is not None:
            self._renderer.render(self._build_frame())

        if isinstance(outcome, _TERMINAL_OUTCOMES):
            self.reset()

        return outcome

    # -- internal ----------------------------------------------------------------

    def _apply_population_delta(self, delta) -> None:
        for spawned in delta.spawned:
            pid = spawned.pedestrian.id
            assert pid is not None and spawned.pedestrian.pose is not None
            self._pedestrians[pid] = spawned.pedestrian
            self._pedestrian_missions[pid] = spawned.mission
            self._pedestrian_backends[pid] = NavCoreKinematicBackend(
                spawned.pedestrian.pose
            )
        for pid in delta.despawned_ids:
            self._pedestrians.pop(pid, None)
            self._pedestrian_missions.pop(pid, None)
            backend = self._pedestrian_backends.pop(pid, None)
            if backend is not None:
                backend.close()

    def _robot_outcome(self) -> EpisodeOutcome:
        assert self._robot is not None
        assert self._robot.pose is not None and self._robot.goal is not None
        robot_pos = Vector2(self._robot.pose.px, self._robot.pose.py)
        goal_pos = Vector2(self._robot.goal.gx, self._robot.goal.gy)

        if robot_pos.distance_to(goal_pos) <= self._robot.radius:
            return ReachGoal()

        min_gap = math.inf
        for pedestrian in self._pedestrians.values():
            assert pedestrian.pose is not None
            ped_pos = Vector2(pedestrian.pose.px, pedestrian.pose.py)
            gap = (
                robot_pos.distance_to(ped_pos) - self._robot.radius - pedestrian.radius
            )
            min_gap = min(min_gap, gap)

        if min_gap <= 0.0:
            return Collision()
        if min_gap <= self.danger_margin:
            return Danger(min_gap)

        width = self._env_config["arenaSize"]["width"]
        height = self._env_config["arenaSize"]["height"]
        if (
            abs(self._robot.pose.px) > width / 2
            or abs(self._robot.pose.py) > height / 2
        ):
            return OutRoad()

        if self._episode_time >= self.max_episode_time:
            return Timeout()

        return Nothing()

    def _build_frame(self) -> RenderFrame:
        assert self._robot is not None and self._robot.pose is not None
        agents = [self._robot, *self._pedestrians.values()]
        positions = np.array(
            [[a.pose.px, a.pose.py] for a in agents if a.pose is not None]
        )
        radii = np.array([a.radius for a in agents])
        kinds = tuple(a.agent for a in agents)
        return RenderFrame(
            sim_time=self._episode_time,
            step=self._step_count,
            episode=self._episode_index,
            agent_positions=positions,
            agent_radii=radii,
            agent_kinds=kinds,
            obstacles=self._obstacles,
        )


def default_scene_factory(
    rand: np.random.Generator,
) -> tuple[tuple[Obstacle, ...], Robot]:
    """Convenience default: existing ``ObstacleBuilder`` + ``RobotBuilder``.

    Deliberately skips ``CrowdBuilder`` -- the continuously-regenerating
    population is ``CrowdSpawner``'s job, not the static one-shot crowd
    ``CrowdBuilder`` produces. Kept as a plain function (not a class)
    since it has no state of its own beyond what the two builders
    already own.

    ``rand`` is the same generator ``Simulation`` itself was
    constructed with -- both builders now draw from it directly rather
    than seeding their own, so a full run (obstacles, robot, and
    population together) is reproducible end-to-end from one seed.
    """
    from navcore.builder.obstacle_builder import ObstacleBuilder
    from navcore.builder.robot_builder import RobotBuilder

    obstacle_builder = ObstacleBuilder(rand)
    obstacle_builder.build_boundary()
    obstacle_builder.build_table()

    robot_builder = RobotBuilder(rand)
    robot_builder.build_robot()
    assert robot_builder.robot is not None

    return tuple(obstacle_builder.obstacles), robot_builder.robot

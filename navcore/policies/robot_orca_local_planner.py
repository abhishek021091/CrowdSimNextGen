"""BaseORCAPlanner: RVO2-backed crowd simulation for the ORCA policy.

Wraps ``rvo2.PyRVOSimulator`` so it can consume a navcore ``Environment``
directly, without callers needing to know RVO2's own agent-id bookkeeping.
This class is deliberately *not* an ``ExecutionBackend`` -- it computes
velocities for a batch of agents in one RVO2-native step, whereas
``ExecutionBackend`` implementations advance the world given velocities
already chosen elsewhere. Where this fits: a ``Policy`` implementation
(e.g. an ``ORCAPolicy``) or the ``Simulation`` orchestrator's compute
phase would call ``compute_velocities`` once per tick and hand the
results to whichever ``ExecutionBackend`` is active, the same way any
other velocity source would.

Design note -- why this returns velocities instead of applying them:
    ``Simulation`` uses two-phase tick ordering (compute-then-apply) to
    avoid first-mover bias between agents. If this class wrote positions
    or velocities directly onto ``Agent`` instances, it would perform
    its own implicit "apply" step outside that ordering, and running it
    alongside any other policy in the same tick would silently break the
    no-first-mover-bias guarantee. Keeping this class read-in,
    compute, return-out keeps that guarantee intact regardless of what
    else is happening in the tick.

Design note -- why the robot is optional in RVO2's agent set:
    ``robot_visible`` mirrors the project's ``VisibilityPolicy`` split:
    when the robot must be invisible to pedestrians, it is simply never
    added to the RVO2 simulation, so pedestrian ORCA velocities are
    computed as if the robot did not exist. All bookkeeping here treats
    ``robot_id is None`` as "robot not represented in RVO2" rather than
    as an error case.
"""

from __future__ import annotations

from math import cos, hypot, pi, sin
from pathlib import Path
from typing import Any, Protocol, cast

import rvo2
import tomllib

import navcore.configs
from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.polygon import Polygon
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.components.velocity import Velocity
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.environment.environment import Environment

#: Key used for the robot's entry in ``compute_velocities``' return dict.
#: Pedestrian ids are non-negative ints assigned by ``CrowdBuilder``, so
#: -1 is guaranteed not to collide with a real pedestrian id.
ROBOT_VELOCITY_KEY = -1


class RVOSimulator(Protocol):
    def addAgent(self, position: tuple[float, float]) -> int: ...

    def addObstacle(
        self,
        vertices: list[tuple[float, float]],
    ) -> int: ...

    def processObstacles(self) -> None: ...

    def doStep(self) -> None: ...

    def setAgentPosition(
        self,
        agent_id: int,
        position: tuple[float, float],
    ) -> None: ...

    def setAgentVelocity(
        self,
        agent_id: int,
        velocity: tuple[float, float],
    ) -> None: ...

    def setAgentPrefVelocity(
        self,
        agent_id: int,
        velocity: tuple[float, float],
    ) -> None: ...

    def getAgentVelocity(
        self,
        agent_id: int,
    ) -> tuple[float, float]: ...


class BaseORCAPlanner:
    """Drives RVO2 ORCA simulation for a navcore ``Environment``.

    One instance owns one RVO2 simulator and its full agent/obstacle
    population. It is not itself a ``Policy`` -- it computes velocities
    for the *entire* crowd (and optionally the robot) in a single RVO2
    step, since ORCA is fundamentally a joint computation across all
    participating agents, not a per-agent decision made in isolation.

    Attributes:
        sim: The underlying RVO2 simulator instance.
        robot_id: This planner's RVO2 agent id for the robot, or
            ``None`` if the robot has not been added (either not yet
            initialized, or intentionally excluded via
            ``robot_visible=False``).
        pedestrian_ids: Maps each pedestrian's navcore ``id`` to its
            RVO2 agent id.
        initialized: Whether ``initialize`` has been called successfully.
    """

    CONFIG_DIR = Path(navcore.configs.__file__).parent

    with open(CONFIG_DIR / "env.toml", "rb") as f:
        env_config: dict[str, Any] = tomllib.load(f)

    TIME_STEP = env_config["policy"]["time_step"]

    def __init__(self, config_file: str) -> None:
        """Construct the RVO2 simulator from an ORCA config file.

        Args:
            config_file: Filename (relative to ``navcore/configs/``) of
                a TOML file containing an ``[orca]`` table with
                ``neighbor_dist``, ``max_neighbors``, ``time_horizon``,
                ``time_horizon_obst``, ``agent_radius``, and
                ``max_speed``.
        """
        with open(self.CONFIG_DIR / config_file, "rb") as f:
            config = tomllib.load(f)

        self._max_speed = config["orca"]["max_speed"]

        self.sim: RVOSimulator = cast(
            RVOSimulator,
            rvo2.PyRVOSimulator(  # type: ignore
                self.TIME_STEP,
                config["orca"]["neighbor_dist"],
                config["orca"]["max_neighbors"],
                config["orca"]["time_horizon"],
                config["orca"]["time_horizon_obst"],
                config["orca"]["agent_radius"],
                config["orca"]["max_speed"],
            ),
        )

        self.robot_id: int | None = None
        self.pedestrian_ids: dict[int, int] = {}

        self.initialized = False

    def initialize(self, env: Environment, robot_visible: bool) -> None:
        """Populate the RVO2 simulator from ``env``'s current state.

        Must be called once before the first ``compute_velocities``
        call, and again whenever the episode resets (new agent
        population, new obstacle layout) -- RVO2 has no notion of
        removing agents or obstacles individually, so a fresh episode
        means a fresh simulator population built here.

        Args:
            env: The environment to read initial agent poses and
                obstacle geometry from.
            robot_visible: Whether the robot should be represented as an
                RVO2 agent at all. When ``False``, pedestrian ORCA
                velocities are computed as if the robot did not exist,
                matching ``AsymmetricVisibility``.

        Raises:
            AssertionError: If the robot's or any pedestrian's pose has
                not been set yet.
        """
        robot = env.robot
        assert robot.pose is not None, "Robot pose must be set before initialize()."

        if robot_visible:
            self.robot_id = self.sim.addAgent((robot.pose.px, robot.pose.py))
        else:
            self.robot_id = None

        self.pedestrian_ids.clear()

        for ped in env.crowd.values():
            assert ped.pose is not None, (
                f"Pedestrian {ped.id} pose must be set before initialize()."
            )

            pose = ped.pose

            self.pedestrian_ids[ped.id] = self.sim.addAgent((pose.px, pose.py))

        for obstacle in env.obstacles.values():
            self._add_obstacle(obstacle)

        self.sim.processObstacles()

        self.initialized = True

    def compute_velocities(self, env: Environment) -> dict[int, Velocity]:
        """Run one ORCA step and return each agent's resulting velocity.

        This is the only method that advances RVO2's internal state. It
        does not write anything back onto ``env``'s agents -- applying
        the returned velocities (and integrating position from them) is
        the caller's responsibility, consistent with ``Simulation``'s
        compute-then-apply tick ordering.

        Args:
            env: The environment to read current poses/goals from before
                stepping. Must be the same population that
                ``initialize`` was called with (same pedestrian ids,
                same robot-visibility setting) -- this method does not
                detect population drift.

        Returns:
            A dict mapping each pedestrian's navcore ``id`` to its
            computed ``Velocity``, plus (only if the robot was added as
            an RVO2 agent) an entry keyed by ``ROBOT_VELOCITY_KEY`` for
            the robot's computed velocity.

        Raises:
            RuntimeError: If ``initialize`` has not been called yet.
        """
        if not self.initialized:
            raise RuntimeError("compute_velocities() called before initialize().")

        self._sync_agents(env)
        self._set_preferred_velocities(env)
        self.sim.doStep()

        velocities: dict[int, Velocity] = {}

        if self.robot_id is not None:
            robot_velocity: tuple[float, float] = self.sim.getAgentVelocity(
                self.robot_id
            )
            velocities[ROBOT_VELOCITY_KEY] = Velocity(
                robot_velocity[0], robot_velocity[1]
            )

        for ped_id, rvo_id in self.pedestrian_ids.items():
            ped_velocity: tuple[float, float] = self.sim.getAgentVelocity(rvo_id)
            velocities[ped_id] = Velocity(ped_velocity[0], ped_velocity[1])

        return velocities

    def _sync_agents(self, env: Environment) -> None:
        """Push current poses/velocities from ``env`` into RVO2.

        RVO2 does not read navcore's ``Agent`` state on its own -- every
        tick, positions and velocities must be copied in explicitly
        before ``doStep()`` so ORCA's neighbor reasoning reflects where
        agents actually are right now, not where they were at
        ``initialize`` time.
        """
        robot = env.robot
        assert robot.pose is not None

        if self.robot_id is not None:
            self.sim.setAgentPosition(self.robot_id, (robot.pose.px, robot.pose.py))
            if robot.velocity is None:
                self.sim.setAgentVelocity(self.robot_id, (0.0, 0.0))
            else:
                self.sim.setAgentVelocity(
                    self.robot_id,
                    (robot.velocity.vx, robot.velocity.vy),
                )

        for ped in env.crowd.values():
            assert ped.pose is not None
            assert ped.id is not None
            rvo_id = self.pedestrian_ids[ped.id]

            self.sim.setAgentPosition(rvo_id, (ped.pose.px, ped.pose.py))

            if ped.velocity is None:
                self.sim.setAgentVelocity(rvo_id, (0.0, 0.0))
            else:
                self.sim.setAgentVelocity(
                    rvo_id,
                    (ped.velocity.vx, ped.velocity.vy),
                )

    def _set_preferred_velocities(self, env: Environment) -> None:
        """Tell RVO2 each agent's unobstructed preferred velocity.

        ORCA needs a "what would this agent do with no one else around"
        input per agent, per step -- this computes that as a straight
        line toward each agent's current goal at its own ``v_pref``.
        """
        robot = env.robot
        assert robot.pose is not None
        assert robot.goal is not None

        if self.robot_id is not None:
            pref = self._preferred_velocity(
                robot.pose.px,
                robot.pose.py,
                robot.goal.gx,
                robot.goal.gy,
                robot.v_pref,
            )
            self.sim.setAgentPrefVelocity(self.robot_id, pref)

        for ped in env.crowd.values():
            assert ped.pose is not None
            assert ped.goal is not None
            assert ped.id is not None

            pref = self._preferred_velocity(
                ped.pose.px,
                ped.pose.py,
                ped.goal.gx,
                ped.goal.gy,
                ped.v_pref,
            )
            self.sim.setAgentPrefVelocity(self.pedestrian_ids[ped.id], pref)

    @staticmethod
    def _preferred_velocity(
        px: float, py: float, gx: float, gy: float, speed: float
    ) -> tuple[float, float]:
        """Return the straight-line velocity toward ``(gx, gy)`` at ``speed``.

        Returns ``(0.0, 0.0)`` if already (numerically) at the goal, to
        avoid a divide-by-zero when normalizing a zero-length vector.
        """
        dx = gx - px
        dy = gy - py

        distance = hypot(dx, dy)
        if distance <= 1e-6:
            return (0.0, 0.0)

        return (speed * dx / distance, speed * dy / distance)

    def _add_obstacle(self, obstacle: Obstacle) -> None:
        """Register one navcore ``Obstacle`` as an RVO2 obstacle polygon.

        RVO2 only understands polygonal obstacles, so ``Circle`` is
        approximated with a fixed-segment-count polygon. This is a
        one-time cost at ``initialize`` time, not per-tick, so the
        segment count is not currently exposed as a performance knob --
        revisit if very large or very small circular obstacles start
        showing visibly faceted ORCA behavior.

        Raises:
            TypeError: If ``obstacle.geometry`` is not one of
                ``Polygon``, ``Rectangle``, or ``Circle``.
        """
        geometry = obstacle.geometry

        if isinstance(geometry, Polygon):
            vertices = [(vertex.x, vertex.y) for vertex in geometry.vertices]

        elif isinstance(geometry, Rectangle):
            vertices = [(vertex.x, vertex.y) for vertex in geometry.vertices()]

        elif isinstance(geometry, Circle):
            num_segments = 16
            vertices = [
                (
                    geometry.center.x
                    + geometry.radius * cos(2.0 * pi * i / num_segments),
                    geometry.center.y
                    + geometry.radius * sin(2.0 * pi * i / num_segments),
                )
                for i in range(num_segments)
            ]

        else:
            raise TypeError(f"Unsupported obstacle geometry: {type(geometry).__name__}")

        self.sim.addObstacle(vertices)

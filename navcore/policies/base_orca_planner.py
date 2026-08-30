"""BaseORCAPlanner: a generic ORCA velocity engine over an opaque observation.

Wraps ``rvo2.PyRVOSimulator`` behind a contract that has no knowledge of
"robot" vs. "pedestrian", visibility, or ``navcore.environment.Environment``.
It knows exactly one thing: given a mapping from an opaque, caller-chosen
id to a ``FullState``, compute each id's ORCA velocity for this tick.

Why the caller owns population and visibility:
    Which agents exist, and which agents are visible to which other
    agents, is already decided elsewhere in the architecture
    (``VisibilityPolicy`` / ``NeighborQuery``). Duplicating that logic
    here -- e.g. a ``robot_visible`` flag -- would let this class silently
    diverge from the rest of the simulation's visibility rules. Instead,
    whatever calls this planner (typically an ``ORCAPolicy`` or the
    ``Simulation`` orchestrator's compute phase) is responsible for
    building the ``observation`` dict from exactly the agents that should
    participate in this ORCA computation, under whatever visibility rule
    is active. A caller that wants "robot invisible to pedestrians" simply
    runs two separate ``BaseORCAPlanner`` instances (or two
    ``initialize``/``compute_velocities`` passes) with different subsets
    of the population -- this class never has to know that distinction
    exists.

Why ``FullState`` is the observation's value type:
    ``FullState`` (pose, goal, velocity, radius, preferred_speed) is
    already the project's agent-type-agnostic snapshot of "everything a
    navigation computation needs to know about one agent, right now." It
    carries no notion of "Robot" or "Pedestrian". Using it here means
    this planner's input type doesn't have to be invented or duplicated,
    and any future agent kind (wheelchair, stretcher, delivery robot)
    works with zero changes to this file.

Why per-agent radius/speed instead of one global config:
    Pedestrians in this project have randomized per-agent radius and
    ``v_pref`` (see ``Pedestrian.__init__``). The previous version passed
    only ``position`` to RVO2's ``addAgent`` and relied on the
    simulator-wide default radius/max-speed from config, silently
    discarding that per-agent variability for every ORCA computation.
    Since ``FullState`` already carries the real per-agent values, they
    are now passed through to RVO2 directly. The ORCA *reasoning*
    parameters (``neighbor_dist``, ``max_neighbors``, ``time_horizon``,
    ``time_horizon_obst``) remain simulator-wide config, since those are
    properties of the ORCA algorithm's lookahead, not of any one agent's
    physical body.

Design note -- why this returns velocities instead of applying them:
    ``Simulation`` uses two-phase tick ordering (compute-then-apply) to
    avoid first-mover bias between agents. If this class wrote positions
    or velocities directly onto ``Agent`` instances, it would perform its
    own implicit "apply" step outside that ordering. Keeping this class
    read-in, compute, return-out keeps that guarantee intact regardless
    of what else is happening in the tick.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from math import cos, hypot, pi, sin
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast

import rvo2
import tomllib

import navcore.configs
from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.polygon import Polygon
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.components.state import FullState
from navcore.entities.components.velocity import Velocity
from navcore.entities.obstacles.obstacle import Obstacle

#: An opaque, caller-chosen identifier for one agent in the observation.
#: This planner never interprets it -- it is only used as a dict key to
#: correlate observation entries with computed velocities.
AgentId = TypeVar("AgentId", bound=Hashable)


def obstacle_to_vertices(obstacle: Obstacle) -> list[tuple[float, float]]:
    """Convert one navcore ``Obstacle`` into RVO2-native polygon vertices.

    Pure geometry conversion, deliberately free of any RVO2 simulator
    dependency, so callers can precompute this once per episode (when the
    obstacle layout is fixed) instead of paying for it on every
    ``BaseORCAPlanner.initialize()`` call.

    Raises:
        TypeError: If ``obstacle.geometry`` is not one of ``Polygon``,
            ``Rectangle``, or ``Circle``.
    """
    geometry = obstacle.geometry

    if isinstance(geometry, Polygon):
        return [(vertex.x, vertex.y) for vertex in geometry.vertices]

    if isinstance(geometry, Rectangle):
        return [(vertex.x, vertex.y) for vertex in geometry.vertices()]

    if isinstance(geometry, Circle):
        num_segments = 16
        return [
            (
                geometry.center.x + geometry.radius * cos(2.0 * pi * i / num_segments),
                geometry.center.y + geometry.radius * sin(2.0 * pi * i / num_segments),
            )
            for i in range(num_segments)
        ]

    raise TypeError(f"Unsupported obstacle geometry: {type(geometry).__name__}")


class RVOSimulator(Protocol):
    def addAgent(
        self,
        position: tuple[float, float],
        neighborDist: float,
        maxNeighbors: int,
        timeHorizon: float,
        timeHorizonObst: float,
        radius: float,
        maxSpeed: float,
        velocity: tuple[float, float],
    ) -> int: ...

    def addObstacle(self, vertices: list[tuple[float, float]]) -> int: ...

    def processObstacles(self) -> None: ...

    def doStep(self) -> None: ...

    def setAgentPosition(
        self, agent_id: int, position: tuple[float, float]
    ) -> None: ...

    def setAgentVelocity(
        self, agent_id: int, velocity: tuple[float, float]
    ) -> None: ...

    def setAgentPrefVelocity(
        self, agent_id: int, velocity: tuple[float, float]
    ) -> None: ...

    def getAgentVelocity(self, agent_id: int) -> tuple[float, float]: ...


class BaseORCAPlanner(Generic[AgentId]):
    """Computes one ORCA step's velocities for an arbitrary set of agents.

    One instance owns one RVO2 simulator. It knows nothing about agent
    roles, visibility, or the wider ``Environment`` -- it only knows how
    to turn ``Mapping[AgentId, FullState]`` into ``dict[AgentId, Velocity]``.

    Attributes:
        sim: The underlying RVO2 simulator instance.
        initialized: Whether ``initialize`` has been called successfully.
    """

    CONFIG_DIR = Path(navcore.configs.__file__).parent

    with open(CONFIG_DIR / "env.toml", "rb") as f:
        env_config: dict[str, Any] = tomllib.load(f)

    TIME_STEP = env_config["policy"]["time_step"]

    def __init__(self, config_file: str) -> None:
        """Construct the RVO2 simulator from an ORCA reasoning-parameter file.

        Args:
            config_file: Filename (relative to ``navcore/configs/``) of a
                TOML file containing an ``[orca]`` table with
                ``neighbor_dist``, ``max_neighbors``, ``time_horizon``,
                and ``time_horizon_obst``. These are simulator-wide ORCA
                lookahead parameters, not per-agent physical properties
                -- per-agent radius/speed come from each tick's
                ``FullState`` instead (see module docstring).
        """
        with open(self.CONFIG_DIR / config_file, "rb") as f:
            self.config = tomllib.load(f)

        orca = self.config["orca"]
        self._neighbor_dist: float = orca["neighbor_dist"]
        self._max_neighbors: int = orca["max_neighbors"]
        self._time_horizon: float = orca["time_horizon"]
        self._time_horizon_obst: float = orca["time_horizon_obst"]

        self.sim: RVOSimulator = cast(
            RVOSimulator,
            rvo2.PyRVOSimulator(self.TIME_STEP, 0.0, 0),  # type: ignore[arg-type]
        )

        self._rvo_ids: dict[AgentId, int] = {}
        self.initialized = False

    def initialize(
        self,
        observation: Mapping[AgentId, FullState],
        obstacle_vertices: Sequence[Sequence[tuple[float, float]]] = (),
    ) -> None:
        """Populate the RVO2 simulator from a starting observation.

        Must be called once before the first ``compute_velocities`` call,
        and again whenever the participating population changes (new
        episode, agents added/removed) -- RVO2 has no notion of adding or
        removing individual agents once constructed, so a changed
        population means a fresh simulator built here.

        Args:
            observation: Every agent that should participate in this
                planner's ORCA computation, keyed by an opaque id chosen
                by the caller. May be empty -- an empty observation is a
                valid (if uninteresting) simulation with zero agents.
            obstacles: Static obstacles the RVO2 simulator should account
                for. Defaults to none.
        """
        self._rvo_ids = {}

        for agent_id, state in observation.items():
            rvo_id = self.sim.addAgent(
                (state.pose.px, state.pose.py),
                self._neighbor_dist,
                self._max_neighbors,
                self._time_horizon,
                self._time_horizon_obst,
                state.radius,
                state.preferred_speed,
                (state.velocity.vx, state.velocity.vy),
            )
            self._rvo_ids[agent_id] = rvo_id

        for vertices in obstacle_vertices:
            self.sim.addObstacle(list(vertices))

        self.sim.processObstacles()
        self.initialized = True

    def compute_velocities(
        self, observation: Mapping[AgentId, FullState]
    ) -> dict[AgentId, Velocity]:
        """Run one ORCA step and return each agent's resulting velocity.

        Args:
            observation: This tick's current state for exactly the
                agents ``initialize`` was called with -- same ids, same
                population size. May be empty if ``initialize`` was
                given an empty observation.

        Returns:
            A dict mapping each ``observation`` key to its computed
            ``Velocity``. Empty if ``observation`` is empty.

        Raises:
            RuntimeError: If ``initialize`` has not been called yet.
            ValueError: If ``observation``'s ids don't match the
                population ``initialize`` built -- RVO2 cannot add or
                remove agents mid-simulator, so a mismatch here means
                the caller needs to call ``initialize`` again rather
                than ``compute_velocities``.
        """
        if not self.initialized:
            raise RuntimeError("compute_velocities() called before initialize().")

        if observation.keys() != self._rvo_ids.keys():
            raise ValueError(
                "compute_velocities() observation does not match the "
                "population passed to initialize(); call initialize() "
                "again if the participating agents have changed."
            )

        for agent_id, state in observation.items():
            rvo_id = self._rvo_ids[agent_id]
            self.sim.setAgentPosition(rvo_id, (state.pose.px, state.pose.py))
            self.sim.setAgentVelocity(rvo_id, (state.velocity.vx, state.velocity.vy))
            self.sim.setAgentPrefVelocity(
                rvo_id,
                self._preferred_velocity(
                    state.pose.px,
                    state.pose.py,
                    state.goal.gx,
                    state.goal.gy,
                    state.preferred_speed,
                ),
            )

        self.sim.doStep()

        return {
            agent_id: Velocity(*self.sim.getAgentVelocity(rvo_id))
            for agent_id, rvo_id in self._rvo_ids.items()
        }

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

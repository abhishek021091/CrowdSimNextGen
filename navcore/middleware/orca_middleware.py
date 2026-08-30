"""DecentralizedORCAPlanner: adapts BaseORCAPlanner to Step's per-agent,
sensor-limited call pattern.

``Step`` calls its planner once per tick, per observing agent, with
whatever neighbors that agent's sensor currently reports -- it never
hands the planner a stable, whole-episode population. ``BaseORCAPlanner``,
by design, is a centralized joint solver: it wants ``initialize()`` once
for a fixed population, then ``compute_velocities()`` many times against
that same population (see its module docstring). This class exists
*only* to reconcile that mismatch -- it is intentionally the one place
in the codebase that knows both "Step's calling convention" and
"BaseORCAPlanner's population model." Nothing downstream of ``Step``
should ever import this module.

Design note -- why neighbors get a synthetic goal:
    ``Step`` only gives this planner ``ObservableState`` for neighbors
    (pose, velocity, radius) -- deliberately not their goal, since goal
    is private intent, not something visibility is supposed to reveal
    (see ``VisibilityPolicy``). But ORCA fundamentally needs a preferred
    velocity per participant to compute half-plane constraints correctly
    for everyone in the local solve, not just for ``self``. The
    standard resolution (used by comparable ORCA-based crowd
    simulators) is to assume each unknown-intent neighbor's preferred
    velocity equals its *current* velocity -- i.e. "assume they keep
    doing what they're doing." This is implemented here as a synthetic
    one-step-ahead goal (``pose + velocity``) rather than special-casing
    ``BaseORCAPlanner`` to accept preferred velocities directly, so the
    underlying planner stays ignorant of this approximation entirely.
    This is an approximation, not ground truth -- it will be wrong
    whenever a neighbor is about to change direction, same as it is in
    any other ORCA-based crowd simulator using this technique.

Design note -- why a fresh BaseORCAPlanner is built every call:
    RVO2 cannot add or remove agents from a live simulator, and the
    neighbor set here changes every call (different agent, different
    tick, different sensor result). Reusing one persistent simulator
    across calls with a shrinking/growing population would violate
    ``BaseORCAPlanner.compute_velocities``'s population-match invariant.
    The cost is one simulator construction per call; see this class's
    module docstring in the review notes for the centralized
    alternative if this becomes a measured bottleneck.

Design note -- why obstacles are bound at construction, not per call:
    Obstacle layouts are static within a single episode (they only
    randomize *between* episodes -- see ``ScenarioConfig``). So the
    caller builds one ``DecentralizedORCAPlanner`` per episode with that
    episode's obstacles, rather than this class re-deriving them from
    something passed in on every tick.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.components.state import FullState, ObservableState
from navcore.entities.components.velocity import Velocity
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.policies.base_orca_planner import BaseORCAPlanner, obstacle_to_vertices

#: Internal key for "self" in the per-call local population. Chosen to
#: never collide with real agent ids: pedestrian ids are non-negative
#: ints, the robot's id (per Step.ROBOT_KEY) is -1, so a distinct
#: sentinel string can't collide with either without special-casing.
_SELF_KEY = "__self__"


class VelocityPlanner(Protocol):
    """What ``Step`` requires from any planner it drives.

    Kept here (rather than only implicit in ``Step``) so implementations
    have one explicit contract to satisfy, and so ``Step`` can be
    type-annotated against it without a forward-reference-only name.
    """

    def compute_velocities(
        self,
        self_id: int,
        self_state: FullState,
        observations: Mapping[int, ObservableState],
    ) -> tuple[Velocity, dict[int, Velocity]]:
        """Return ``(self's velocity, {neighbor_id: neighbor's velocity})``.

        Args:
            self_id: The calling agent's own id in ``observations``'
                id-space (``Step.ROBOT_KEY`` for the robot, ``ped.id``
                for a pedestrian).
            self_state: The calling agent's own full state, including
                its real goal and preferred speed.
            observations: Every neighbor currently visible to the
                calling agent, keyed by their id. Does not need to
                (and should not be relied on to) include ``self_id``.
        """
        ...


class DecentralizedORCAPlanner:
    """Runs one small, self-contained ORCA solve per ``Step`` call.

    Attributes:
        config_file: ORCA reasoning-parameter TOML, forwarded to each
            per-call ``BaseORCAPlanner`` (see its constructor).
        obstacles: This episode's static obstacles, forwarded to every
            per-call solve.
    """

    def __init__(
        self, config_file: str, obstacles: dict[str, Obstacle] | None = None
    ) -> None:
        self.config_file = config_file
        assert obstacles is not None
        self._obstacle_vertices: list[list[tuple[float, float]]] = [
            obstacle_to_vertices(obstacle) for obstacle in obstacles.values()
        ]

    def compute_velocities(
        self,
        self_id: int,
        self_state: FullState,
        observations: Mapping[int, ObservableState],
    ) -> tuple[Velocity, dict[int, Velocity]]:
        """Solve ORCA for ``self_state`` plus its visible neighbors.

        Builds a fresh, throwaway ``BaseORCAPlanner`` population of
        ``self`` (real goal) plus each neighbor (synthetic
        continue-as-is goal), steps it once, and splits the result back
        into ``(self_velocity, {neighbor_id: velocity})``.

        Returns:
            ``self``'s computed velocity, and a dict of each neighbor's
            computed velocity keyed by their real id (``self_id`` is
            never a key in the returned dict -- it's returned
            separately as the first element).
        """
        local_population: dict[object, FullState] = {_SELF_KEY: self_state}
        for neighbor_id, observed in observations.items():
            if neighbor_id == self_id:
                continue  # Step sometimes includes self in observations too.
            local_population[neighbor_id] = self._as_full_state(observed)

        planner: BaseORCAPlanner[object] = BaseORCAPlanner(self.config_file)
        planner.initialize(local_population, obstacle_vertices=self._obstacle_vertices)
        velocities = planner.compute_velocities(local_population)

        self_velocity = velocities.pop(_SELF_KEY)
        other_velocities: dict[int, Velocity] = {
            neighbor_id: velocity
            for neighbor_id, velocity in velocities.items()
            if isinstance(neighbor_id, int)
        }
        return self_velocity, other_velocities

    @staticmethod
    def _as_full_state(observed: ObservableState) -> FullState:
        """Approximate a neighbor's ``FullState`` from what's observable.

        The synthetic goal is one step ahead along the neighbor's
        current velocity, so ORCA's preferred-velocity computation
        reproduces that same velocity (see module docstring). If the
        neighbor is currently stationary, the goal is set to its current
        position so the preferred velocity comes out as zero too,
        matching ``BaseORCAPlanner._preferred_velocity``'s
        divide-by-zero guard.
        """
        pose = observed.pose
        velocity = observed.velocity
        if velocity.vx == 0.0 and velocity.vy == 0.0:
            goal = Goal(pose.px, pose.py)
            preferred_speed = 0.0
        else:
            goal = Goal(pose.px + velocity.vx, pose.py + velocity.vy)
            preferred_speed = (velocity.vx**2 + velocity.vy**2) ** 0.5

        return FullState(
            pose=Pose(pose.px, pose.py, pose.theta),
            goal=goal,
            velocity=Velocity(velocity.vx, velocity.vy),
            radius=observed.radius,
            preferred_speed=preferred_speed,
        )

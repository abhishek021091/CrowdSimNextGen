"""Mission: the "what am I trying to accomplish" abstraction.

A ``Mission`` answers exactly one question, once per simulation tick:
*where should this agent be heading right now?* It has no opinion on
*how* the agent gets there -- that is a ``Policy``'s job (ORCA, PID, an
RL model, ...). This split exists so that "goal reaching" vs. "sweep
the room" vs. "stay with my group" can vary independently of "avoid
collisions using ORCA" vs. "avoid collisions using a learned policy".

Design note -- why ``Mission`` returns a bare ``Vector2`` and not a
``Goal``:
    ``Goal`` (see ``navcore.entities.components.goal``) represents the
    agent's *persistent, semantic* destination. A ``Mission``'s target
    for a given tick is not always the goal itself -- a sweeping robot's
    target is the next coverage waypoint, and a group follower's target
    is a formation offset relative to its leader. Reusing ``Goal`` for
    these transient, tick-by-tick targets would overload its meaning.
    ``Vector2`` keeps ``Mission`` honest about the fact that it only
    produces *points*, not semantics.

Every concrete ``Mission`` must remain ignorant of rendering, physics,
and any specific ``Policy`` implementation -- it only ever reasons about
*where*, never *how fast* or *how*.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


@runtime_checkable
class Mission(Protocol):
    """Produces the agent's current motion target, once per tick.

    Implementations are injected into an agent (or into a per-agent
    controller) rather than subclassed by it -- this keeps ``Robot`` and
    ``Pedestrian`` free of mission-specific logic, and lets any agent
    type carry any mission.
    """

    def get_target(self, agent: Agent, neighbors: Sequence[ObservableState]) -> Vector2:
        """Return the target position the agent should move toward this tick.

        Args:
            agent: The agent this mission belongs to. Implementations
                may read (but must not mutate) ``agent.pose`` and
                ``agent.goal``.
            neighbors: Nearby agents' observable state, as produced by a
                ``NeighborQuery``. Most missions ignore this; it exists
                because collision-aware missions (e.g. ``SweepMission``,
                added separately) need it to decide when to override
                their baseline target.

        Returns:
            The target position, in world coordinates.
        """
        ...

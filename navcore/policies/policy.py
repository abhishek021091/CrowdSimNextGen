"""Policy: the "how do I move toward my target" abstraction.

A ``Policy`` converts (current agent state, this tick's target from
``Mission``, nearby visible agents) into a velocity command. ORCA, PID,
and an RL model are all just different implementations of this one
interface -- none of them need to know whether the target came from a
static goal, a sweep waypoint, or a local-avoidance replan.

Deliberately excluded from this interface: anything about *why* the
target is what it is (that's ``Mission``'s job) and anything about
*which* neighbors are visible (that's ``VisibilityPolicy`` /
``NeighborQuery``'s job, resolved before this is ever called).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState
from navcore.entities.components.velocity import Velocity

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


@runtime_checkable
class Policy(Protocol):
    """Computes a velocity command for one agent, for one tick."""

    def compute_velocity(
        self,
        agent: Agent,
        target: Vector2,
        neighbors: Sequence[ObservableState],
    ) -> Velocity:
        """Return the velocity command to move ``agent`` toward ``target``.

        Args:
            agent: The agent this velocity is being computed for.
                Implementations may read ``agent.pose``, ``agent.radius``,
                and ``agent.v_pref``, but must not mutate ``agent``.
            target: This tick's motion target, from the agent's
                ``Mission``.
            neighbors: Nearby, visible agents' observable state, from a
                ``NeighborQuery``. Policies with no avoidance behavior
                (rare) may simply ignore this.

        Returns:
            The velocity command for this tick, in world coordinates.
        """
        ...

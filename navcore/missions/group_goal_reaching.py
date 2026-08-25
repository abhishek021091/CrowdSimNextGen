"""GroupGoalReachingMission: goal-reaching for a member of a pedestrian group.

One instance is owned per group member (not per group) -- each member's
mission differs only in whether it is the leader or a follower, which
is resolved once at construction via ``group.is_leader(agent_id)``.

Design note -- why this needs an agent lookup:
    A follower's target depends on the *leader's current pose*, which
    this mission does not own and must not cache (the leader moves every
    tick). Rather than giving ``Mission`` implementations a hidden
    reference to the whole crowd, ``GroupGoalReachingMission`` is
    injected with a narrow ``AgentLookup`` callable that resolves a
    single agent id to its live ``Agent`` -- typically
    ``lambda agent_id: crowd_registry[agent_id]``. This keeps the
    dependency explicit and easy to fake in tests.

Formation offsets are expressed in world coordinates for this first
pass (a fixed vector added to the leader's position). Rotating the
offset by the leader's heading, so followers keep formation as the
leader turns, is a natural extension -- deliberately not implemented
yet, to keep this first version simple and testable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState
from navcore.entities.groups.group import Group

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent

#: Resolves an agent id to its live ``Agent`` instance.
AgentLookup = Callable[[str], "Agent"]


class GroupGoalReachingMission:
    """Goal-reaching mission for one member of a ``Group``.

    Attributes:
        agent_id: The id of the member this mission instance belongs to.
        group: The group this member belongs to.
        agent_lookup: Resolves other members' ids to their live agents.
        formation_offset: For followers only, the offset added to the
            leader's position to get this member's target. Ignored for
            the leader. Defaults to zero (follower targets the leader's
            exact position).
    """

    def __init__(
        self,
        agent_id: str,
        group: Group,
        agent_lookup: AgentLookup,
        formation_offset: Vector2 | None = None,
    ) -> None:
        if agent_id not in group.member_ids:
            raise ValueError(
                f"agent_id={agent_id!r} is not a member of group {group.id!r}."
            )
        if formation_offset is None:
            formation_offset = Vector2.zero()
        self.agent_id = agent_id
        self.group = group
        self.agent_lookup = agent_lookup
        self.formation_offset = formation_offset

    def get_target(self, agent: Agent, neighbors: Sequence[ObservableState]) -> Vector2:
        """Return the group goal (leader) or a formation offset (follower).

        Raises:
            RuntimeError: If this is a follower and the leader's pose has
                not been initialized yet.
        """
        if self.group.is_leader(self.agent_id):
            return self.group.goal

        leader = self.agent_lookup(self.group.leader_id)
        if leader.pose is None:
            raise RuntimeError(
                f"Leader {self.group.leader_id!r} has no pose yet; cannot "
                f"compute a formation target for follower {self.agent_id!r}."
            )
        leader_position = Vector2(leader.pose.px, leader.pose.py)
        return leader_position + self.formation_offset

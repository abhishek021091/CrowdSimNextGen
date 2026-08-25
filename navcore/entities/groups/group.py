"""Pedestrian group domain model.

``Group`` is a plain data-holder, deliberately symmetrical to
``navcore.entities.obstacles.obstacle.Obstacle`` in spirit: it records
*membership and shared intent*, and carries no behavior of its own.
Cohesion, separation, and leader-follower motion are not implemented
here -- they belong to ``GroupGoalReachingMission``, which reads a
``Group`` but never mutates it mid-tick.

Splitting and merging groups (planned) will be implemented as
operations that construct *new* ``Group`` instances with updated
membership, consistent with this module staying an immutable value
object.
"""

from __future__ import annotations

from dataclasses import dataclass

from navcore.entities.components.geometry.vector2 import Vector2


@dataclass(slots=True, frozen=True)
class Group:
    """A set of pedestrians sharing a destination and a leader.

    Attributes:
        id: Stable, caller-assigned unique identifier for this group.
        member_ids: The agent ids belonging to this group. Must contain
            at least one id, and must include ``leader_id``.
        goal: The group's shared destination, in world coordinates.
        leader_id: The agent id of the member other members' targets
            are computed relative to. Must be one of ``member_ids``.
    """

    id: str
    member_ids: tuple[str, ...]
    goal: Vector2
    leader_id: str

    def __post_init__(self) -> None:
        """Validate membership.

        Raises:
            ValueError: If ``member_ids`` is empty, or if ``leader_id``
                is not among ``member_ids``.
        """
        if len(self.member_ids) == 0:
            raise ValueError("Group requires at least one member.")
        if self.leader_id not in self.member_ids:
            raise ValueError(
                f"leader_id={self.leader_id!r} is not among "
                f"member_ids={self.member_ids!r}."
            )

    def is_leader(self, agent_id: str) -> bool:
        """Return whether ``agent_id`` is this group's leader."""
        return agent_id == self.leader_id

    def followers(self) -> tuple[str, ...]:
        """Return member ids other than the leader."""
        return tuple(m for m in self.member_ids if m != self.leader_id)

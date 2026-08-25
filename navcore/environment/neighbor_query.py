"""NeighborQuery: the single point where "who can see whom" is resolved.

Every ``Policy`` and ``Mission`` receives its neighbor list already
filtered by the active ``VisibilityPolicy`` -- none of them ever import
or reason about visibility themselves. This keeps ORCA, RL, or any
future policy identical in code regardless of which visibility
experiment is active; only the neighbor list they're handed changes.

Performance note: this is an O(n) scan per query (O(n^2) total per
tick across all agents). That is acceptable at pedestrian-crowd scales
but is the first place to optimize (e.g. a spatial hash / grid bucket)
if agent counts grow large enough for it to dominate tick time -- noted
here rather than pre-optimized, since premature spatial indexing would
obscure this class's one responsibility for no measured benefit yet.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from navcore.entities.components.state import ObservableState
from navcore.visibility.visibility_policy import VisibilityPolicy

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


@dataclass(slots=True)
class NeighborQuery:
    """Finds nearby, visible agents for a given observer.

    Attributes:
        visibility_policy: Governs which candidates are eligible to be
            returned, independent of distance.
    """

    visibility_policy: VisibilityPolicy

    def neighbors_of(
        self,
        observer: "Agent",
        candidates: Sequence["Agent"],
        radius: float,
    ) -> list[ObservableState]:
        """Return observable state for visible agents within ``radius``.

        Args:
            observer: The agent asking "who can I see nearby?".
            candidates: All agents to consider (typically the full
                crowd + robot(s)). ``observer`` itself is skipped even
                if present in this list.
            radius: Maximum distance, in world units, to consider a
                candidate a neighbor.

        Returns:
            Observable state for each candidate that is both within
            ``radius`` and visible to ``observer`` per
            ``visibility_policy``, in no particular order.

        Raises:
            RuntimeError: If ``observer`` or a visible-and-in-range
                candidate has no pose/velocity set yet (surfaced by
                ``Agent.get_observable_state``).
        """
        if observer.pose is None:
            raise RuntimeError("observer.pose must be set before querying neighbors.")

        result: list[ObservableState] = []
        for candidate in candidates:
            if candidate is observer:
                continue
            if candidate.pose is None:
                continue
            if not self.visibility_policy.is_visible(observer, candidate):
                continue

            distance = math.hypot(
                candidate.pose.px - observer.pose.px,
                candidate.pose.py - observer.pose.py,
            )
            if distance <= radius:
                result.append(candidate.get_observable_state())

        return result

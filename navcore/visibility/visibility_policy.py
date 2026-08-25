"""VisibilityPolicy: who can perceive whom, as a swappable strategy.

Exists so that "robots are invisible to pedestrians" (the default
experimental condition, chosen to make the robot minimally intruding)
can be toggled to "everyone sees everyone" for comparison runs, purely
via configuration -- never as an ``if agent_type == "Robot"`` scattered
through simulation code.

``VisibilityPolicy`` only governs what a ``Policy`` is *allowed to
perceive* as input. It never suppresses or alters physical outcomes:
the diagnostic collision detector at the ``Environment`` level (added
separately) still records an actual robot-pedestrian collision
regardless of visibility, since "least intruding" is only meaningful if
we can measure when it fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


@runtime_checkable
class VisibilityPolicy(Protocol):
    """Decides whether ``observer`` may perceive ``subject`` as a neighbor."""

    def is_visible(self, observer: Agent, subject: Agent) -> bool:
        """Return whether ``subject`` should appear in ``observer``'s neighbors."""
        ...


class FullVisibility:
    """Every agent can perceive every other agent.

    The comparison condition against :class:`AsymmetricVisibility`.
    """

    def is_visible(self, observer: Agent, subject: Agent) -> bool:
        """Return ``True`` for any distinct pair of agents."""
        return observer is not subject


class AsymmetricVisibility:
    """Robots are invisible to pedestrians; all other pairs see each other.

    This is the default experimental condition: it lets a robot plan
    around pedestrians while pedestrians behave as if the robot were not
    there, which is the "least intruding" behavior the project targets.

    Note:
        Agent kind is read from ``Agent.agent`` (the human-readable type
        name already set by ``Agent.__init__``, e.g. ``"Robot"`` or
        ``"Pedestrian"``) rather than an ``isinstance`` check, to avoid
        this module importing the concrete ``Robot``/``Pedestrian``
        classes and creating a dependency cycle with ``navcore.entities.agents``.
    """

    def is_visible(self, observer: Agent, subject: Agent) -> bool:
        """Return ``False`` only when a pedestrian observes a robot."""
        if observer is subject:
            return False
        return not (observer.agent == "Pedestrian" and subject.agent == "Robot")

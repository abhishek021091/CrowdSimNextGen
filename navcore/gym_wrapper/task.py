"""Task: the RL-specific "am I done, what's my reward" wrapper around a Mission.

Mission (navcore.missions.mission) deliberately knows only "where should
the agent go right now" -- see its own module docstring. Reward and
termination are RL concerns, not simulation concerns, so they don't
belong on Mission or on Environment. Task is the layer that adds
exactly those two things on top of a Mission it owns, which is what
lets CrowdSimEnv stay task-agnostic: swap the Task implementation and
the same env class supports goal-reaching, coverage, or any future task
without touching CrowdSimEnv itself.

A Task owns exactly one Mission instance per episode and rebuilds it in
`reset()`, since Mission implementations like SweepingMission carry
per-episode state (lane progress, sweep direction) that must not leak
across episodes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from navcore.entities.environment.environment import Environment
from navcore.missions.mission import Mission


@runtime_checkable
class Task(Protocol):
    """What CrowdSimEnv requires from any RL task."""

    @property
    def mission(self) -> Mission:
        """This task's current Mission, used as Step's robot_mission."""
        ...

    def reset(self, env: Environment) -> None:
        """Rebuild this task's Mission and any per-episode state.

        Called once per CrowdSimEnv.reset(), after the Environment has
        been rebuilt but before the first step() of the episode.
        """
        ...

    def reward(self, env: Environment, collided: bool) -> float:
        """Return this tick's scalar reward.

        Args:
            env: The environment after this tick's integration.
            collided: Whether Environment.did_collision_happened()
                flagged a collision on this tick.
        """
        ...

    def is_terminated(self, env: Environment, collided: bool) -> bool:
        """Return whether the episode ended due to task success/failure.

        Does not cover truncation (step-limit timeouts) -- that's
        CrowdSimEnv's concern, since it's a property of the training
        setup, not the task.
        """
        ...

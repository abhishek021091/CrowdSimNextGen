"""RL-specific missions and the baseline goal-reaching task.

Reward is dense progress-to-goal shaping (distance closed this tick)
plus a per-tick time penalty, a one-time goal bonus, and a collision
penalty. Progress shaping avoids the classic sparse-reward problem of
"only reward reaching the goal" -- with ORCA and pedestrians in the
mix, an untrained policy essentially never reaches the goal early in
training, so a sparse signal gives no gradient at all.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import ObservableState
from navcore.entities.environment.environment import Environment
from navcore.missions.goal_reaching import GoalReachingMission

if TYPE_CHECKING:
    from navcore.entities.agents.agent import Agent


class RLWaypointMission:
    """Mission whose target is supplied by an RL waypoint action.

    ``CrowdSimEnv`` owns this mission in waypoint action mode.  Each
    decoded action replaces the target via :meth:`set_target`; the target is
    then consumed by ``Step`` for that tick's ORCA preferred velocity.  The
    robot's persistent goal is intentionally not changed, so task success
    and reward calculations continue to use its semantic destination.
    """

    def __init__(self, initial_target: Vector2) -> None:
        self._target = initial_target

    def set_target(self, target: Vector2) -> None:
        """Set the world-coordinate waypoint for the next planning tick."""
        self._target = target

    def get_target(self, agent: Agent, neighbors: Sequence[ObservableState]) -> Vector2:
        """Return the current waypoint.

        ``agent`` and ``neighbors`` are accepted to satisfy the common
        :class:`Mission` protocol.  The action-selected waypoint is already
        the complete target, so neither is consulted here.
        """
        return self._target


#: Matches Step._compute_velocities's own hardcoded goal-reach radius.
#: TODO: both should read from one shared config value instead of two
#: independent hardcoded constants.


class GoalReachingTask:
    """Navigate the robot to its goal while avoiding collisions.

    Attributes:
        collision_penalty: Added to reward the tick a collision fires.
            Should outweigh any plausible accumulated progress reward,
            so the policy can never treat a collision as a cheap
            shortcut to the goal.
        goal_bonus: One-time reward on the tick the robot's real goal
            (not any mission target) is reached.
        step_penalty: Small negative per-tick reward, so the policy
            prefers reaching the goal quickly over loitering.
        progress_weight: Multiplier on distance-to-goal closed this
            tick. At 1.0, a robot moving straight at v_pref earns
            reward roughly equal to distance covered.
    """

    def __init__(
        self,
        collision_penalty: float = -25.0,
        goal_bonus: float = 50.0,
        step_penalty: float = -0.01,
        progress_weight: float = 2.0,
    ) -> None:
        self.collision_penalty = collision_penalty
        self.goal_bonus = goal_bonus
        self.step_penalty = step_penalty
        self.progress_weight = progress_weight

        self._mission = GoalReachingMission()
        self._prev_distance: float | None = None

    @property
    def mission(self) -> GoalReachingMission:
        return self._mission

    def reset(self, env: Environment) -> None:
        self._mission = GoalReachingMission()
        self._prev_distance = self._distance_to_goal(env)

    def reward(self, env: Environment, collided: bool) -> float:
        distance = self._distance_to_goal(env)
        assert self._prev_distance is not None, "reward() called before reset()."
        progress = self._prev_distance - distance
        self._prev_distance = distance

        reward = self.step_penalty + self.progress_weight * progress
        if collided:
            reward += self.collision_penalty
        if self._reached_goal(env):
            reward += self.goal_bonus
        return reward

    def is_terminated(self, env: Environment, collided: bool) -> bool:
        return self._reached_goal(env) or collided

    def _reached_goal(self, env: Environment) -> bool:
        return self._distance_to_goal(env) <= env.info.goal_reach_tolerance

    @staticmethod
    def _distance_to_goal(env: Environment) -> float:
        robot = env.robot
        assert robot.pose is not None and robot.goal is not None
        return math.hypot(robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py)

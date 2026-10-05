"""GoalReachingTask: the baseline navigate-to-goal RL task.

Reward is dense progress-to-goal shaping (distance closed this tick)
plus a per-tick time penalty, a one-time goal bonus, and a collision
penalty. Progress shaping avoids the classic sparse-reward problem of
"only reward reaching the goal" -- with ORCA and pedestrians in the
mix, an untrained policy essentially never reaches the goal early in
training, so a sparse signal gives no gradient at all.
"""

from __future__ import annotations

import math

from navcore.entities.environment.environment import Environment
from navcore.missions.goal_reaching import GoalReachingMission


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
        collision_penalty: float = -10.0,
        out_bound_penalty: float = 0.0,
        goal_bonus: float = 10.0,
        step_penalty: float = 0.0,
        progress_weight: float = 2.0,
        alpha: float = 0.95,
    ) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha!r}.")
        self.collision_penalty = collision_penalty
        self.out_bound_penalty = out_bound_penalty
        self.goal_bonus = goal_bonus
        self.step_penalty = step_penalty
        self.progress_weight = progress_weight
        self.alpha = alpha

        self._mission = GoalReachingMission()
        self._prev_distance: float | None = None
        self._smoothed_progress: float = 0.0

    def reset(self, env: Environment) -> None:
        self._mission = GoalReachingMission()
        self._prev_distance = self._distance_to_goal(env)
        self._smoothed_progress = 0.0

    def reward(self, env: Environment, collided: bool, out_of_bounds: bool) -> float:
        distance = self._distance_to_goal(env)
        assert self._prev_distance is not None, "reward() called before reset()."
        progress = self._prev_distance - distance
        self._prev_distance = distance

        self._smoothed_progress = self.alpha * self._smoothed_progress + progress

        reward = self.step_penalty + self.progress_weight * self._smoothed_progress
        if collided:
            reward += self.collision_penalty
        if out_of_bounds:
            reward += self.out_bound_penalty
        if self._reached_goal(env):
            reward += self.goal_bonus
        return reward

    def is_terminated(
        self, env: Environment, collided: bool, out_of_bounds: bool
    ) -> bool:
        return self._reached_goal(env) or collided or out_of_bounds

    def _reached_goal(self, env: Environment) -> bool:
        return (
            self._distance_to_goal(env)
            <= env.robot.radius + env.info.goal_reach_tolerance
        )

    @staticmethod
    def _distance_to_goal(env: Environment) -> float:
        robot = env.robot
        assert robot.pose is not None and robot.goal is not None
        return math.hypot(robot.goal.gx - robot.pose.px, robot.goal.gy - robot.pose.py)

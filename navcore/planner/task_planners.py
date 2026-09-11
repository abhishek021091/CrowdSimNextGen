"""Small orchestrators for non-coverage navigation tasks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.gym_wrapper.rl_missions import RLWaypointMission
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.missions.goal_reaching import GoalReachingMission
from navcore.step.step import Step


@dataclass(slots=True)
class MissionMetrics:
    success: bool = False
    collision: bool = False
    timeout: bool = False
    steps: int = 0
    path_length: float = 0.0
    min_separation: float = float("inf")

    def report(self) -> dict[str, float | bool | int]:
        result = asdict(self)
        if math.isinf(self.min_separation):
            result["min_separation"] = 0.0
        return result


class GoalPlanner:
    def __init__(self) -> None:
        self.builder = EnvironmentBuilder()
        self.env = self.builder.build_environment()
        self.step_driver = Step(DecentralizedORCAPlanner("orca.toml", self.env.obstacles), self.env, False, robot_mission=GoalReachingMission())

    def run(self, max_steps: int = 500) -> MissionMetrics:
        metrics = MissionMetrics()
        for _ in range(max_steps):
            pose = self.env.robot.pose
            previous_position = None if pose is None else (pose.px, pose.py)
            result = self.step_driver.step()
            metrics.steps += 1
            if previous_position is not None and self.env.robot.pose is not None:
                metrics.path_length += math.hypot(self.env.robot.pose.px - previous_position[0], self.env.robot.pose.py - previous_position[1])
            for ped in self.env.crowd.values():
                if self.env.robot.pose is not None and ped.pose is not None:
                    metrics.min_separation = min(metrics.min_separation, math.hypot(self.env.robot.pose.px-ped.pose.px, self.env.robot.pose.py-ped.pose.py) - self.env.robot.radius - ped.radius)
            metrics.collision = self.env.did_collision_happened()
            metrics.success = result.robot_reached_goal
            if metrics.success or metrics.collision:
                return metrics
        metrics.timeout = True
        return metrics


class WaypointPlanner(GoalPlanner):
    def __init__(self, waypoint: Vector2) -> None:
        super().__init__()
        self.mission = RLWaypointMission(waypoint)
        self.step_driver.robot_mission = self.mission

    def set_waypoint(self, waypoint: Vector2) -> None:
        self.mission.set_target(waypoint)


class RLPlanner:
    """Gym lifecycle entry point; an external policy supplies actions."""
    def __init__(self) -> None:
        from navcore.gym_wrapper.crowd_sim_env import CrowdSimEnv
        from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask
        self.env = CrowdSimEnv(GoalReachingTask())

    def run(self, policy, max_steps: int = 500) -> None:
        observation, _ = self.env.reset()
        for _ in range(max_steps):
            observation, _, terminated, truncated, _ = self.env.step(policy(observation))
            if terminated or truncated:
                return

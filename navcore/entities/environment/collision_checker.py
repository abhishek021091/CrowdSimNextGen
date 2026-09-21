from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navcore.entities.agents.agent import Agent
from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.rectangle import Rectangle

if TYPE_CHECKING:
    from navcore.entities.environment.environment import Environment


class CollisionChecker:
    def __init__(
        self,
        agent: Agent,
        env: Environment,
    ) -> None:
        self.agent = agent
        self.env = env

    def check_collision(self) -> bool:
        """Check if the agent is in collision with any neighbor or obstacle."""
        agent_pose = self.agent.pose
        for ped in self.env.crowd.values():
            ped_pose = ped.pose
            assert agent_pose is not None, "Agent pose is None"
            assert ped_pose is not None, "Observable state pose is None"
            distance = np.linalg.norm(
                [agent_pose.px - ped_pose.px, agent_pose.py - ped_pose.py]
            )
            if distance < (
                self.agent.radius + ped.radius + self.env.info.safety_distance
            ):
                return True
        for obstacle in self.env.obstacles.values():
            obstacle_pose = obstacle.geometry.center
            assert agent_pose is not None, "Agent pose is None"
            if isinstance(obstacle.geometry, Circle):
                distance = np.linalg.norm(
                    [agent_pose.px - obstacle_pose.x, agent_pose.py - obstacle_pose.y]
                )
                if distance < (
                    self.agent.radius
                    + obstacle.geometry.radius
                    + self.env.info.safety_distance
                ):
                    return True
            elif isinstance(obstacle.geometry, Rectangle):
                # Calculate the effective radius of the agent
                effective_radius = self.agent.radius + self.env.info.safety_distance

                # Calculate rectangle bounds
                half_width = obstacle.geometry.width / 2.0
                half_height = obstacle.geometry.height / 2.0

                min_x = obstacle_pose.x - half_width
                max_x = obstacle_pose.x + half_width
                min_y = obstacle_pose.y - half_height
                max_y = obstacle_pose.y + half_height

                # Find the closest point on the rectangle to the agent's center
                closest_x = max(min_x, min(agent_pose.px, max_x))
                closest_y = max(min_y, min(agent_pose.py, max_y))

                # Calculate the squared distance from the agent's center to this closest point
                dx = agent_pose.px - closest_x
                dy = agent_pose.py - closest_y

                # If the squared distance is less than the squared radius, they intersect
                if (dx * dx + dy * dy) <= (effective_radius * effective_radius):
                    return True
            return False

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
            if distance < (self.agent.radius + ped.radius):
                return True
        for obstacle in self.env.obstacles.values():
            obstacle_pose = obstacle.geometry.center
            assert agent_pose is not None, "Agent pose is None"
            if isinstance(obstacle.geometry, Circle):
                distance = np.linalg.norm(
                    [agent_pose.px - obstacle_pose.x, agent_pose.py - obstacle_pose.y]
                )
                if distance < (self.agent.radius + obstacle.geometry.radius):
                    return True
            elif isinstance(obstacle.geometry, Rectangle):
                # Check if the agent is within the rectangle bounds
                half_width = obstacle.geometry.width / 2
                half_height = obstacle.geometry.height / 2
                if (
                    obstacle_pose.x - half_width
                    <= agent_pose.px
                    <= obstacle_pose.x + half_width
                    and obstacle_pose.y - half_height
                    <= agent_pose.py
                    <= obstacle_pose.y + half_height
                ):
                    return True
        return False

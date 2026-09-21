# navcore/entities/environment/collision_checker.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navcore.entities.agents.agent import Agent
from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.containment import point_in_geometry
from navcore.entities.components.geometry.polygon import Polygon
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.components.geometry.vector2 import Vector2

if TYPE_CHECKING:
    from navcore.entities.environment.environment import Environment


def _point_to_segment_distance(
    point: Vector2, seg_start: Vector2, seg_end: Vector2
) -> float:
    ab = seg_end - seg_start
    ab_len_sq = ab.magnitude_squared()
    if ab_len_sq <= 1e-12:
        return point.distance_to(seg_start)
    t = max(0.0, min(1.0, (point - seg_start).dot(ab) / ab_len_sq))
    closest = seg_start + ab * t
    return point.distance_to(closest)


class CollisionChecker:
    def __init__(self, agent: Agent, env: Environment) -> None:
        self.agent = agent
        self.env = env

    def check_collision(self) -> bool:
        """Return whether the agent collides with any neighbor or obstacle.

        A "boundary" obstacle (id == "boundary", the convention already
        used by boustropheden.py and geometry_conversion.py) is treated
        with inverted polygon semantics: solid on the *outside*, not the
        inside, since it represents the arena's outer limit rather than
        something to stay clear of. Every other Polygon obstacle uses
        ordinary too-close-or-overlapping semantics.
        """
        agent_pose = self.agent.pose
        assert agent_pose is not None, "Agent pose is None"

        for ped in self.env.crowd.values():
            ped_pose = ped.pose
            assert ped_pose is not None, "Observable state pose is None"
            distance = np.linalg.norm(
                [agent_pose.px - ped_pose.px, agent_pose.py - ped_pose.py]
            )
            if distance < (
                self.agent.radius + ped.radius + self.env.info.safety_distance
            ):
                return True

        effective_radius = self.agent.radius + self.env.info.safety_distance

        for obstacle in self.env.obstacles.values():
            if isinstance(obstacle.geometry, Circle):
                center = obstacle.geometry.center
                distance = np.linalg.norm(
                    [agent_pose.px - center.x, agent_pose.py - center.y]
                )
                if distance < (
                    self.agent.radius
                    + obstacle.geometry.radius
                    + self.env.info.safety_distance
                ):
                    return True

            elif isinstance(obstacle.geometry, Rectangle):
                center = obstacle.geometry.center
                half_width = obstacle.geometry.width / 2.0
                half_height = obstacle.geometry.height / 2.0

                min_x, max_x = center.x - half_width, center.x + half_width
                min_y, max_y = center.y - half_height, center.y + half_height

                closest_x = max(min_x, min(agent_pose.px, max_x))
                closest_y = max(min_y, min(agent_pose.py, max_y))

                dx = agent_pose.px - closest_x
                dy = agent_pose.py - closest_y
                if (dx * dx + dy * dy) <= (
                    (effective_radius + self.env.info.safety_distance)
                    * (effective_radius + self.env.info.safety_distance)
                ):
                    return True

            elif isinstance(obstacle.geometry, Polygon):
                point = Vector2(agent_pose.px, agent_pose.py)
                nearest = min(
                    _point_to_segment_distance(point, edge.start, edge.end)
                    for edge in obstacle.geometry.edges()
                )
                if obstacle.id == "boundary":
                    inside = point_in_geometry(point, obstacle.geometry)
                    if (
                        not inside
                        or nearest <= effective_radius + self.env.info.safety_distance
                    ):
                        return True
                elif nearest <= effective_radius + self.env.info.safety_distance:
                    return True

        return False

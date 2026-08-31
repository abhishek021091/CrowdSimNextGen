from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot
from typing import Any

import numpy as np
from numpy.typing import NDArray

from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.components.state import ObservableState
from navcore.entities.obstacles import Obstacle
from navcore.policies.base_orca_planner import obstacle_to_vertices


class SAT:
    def __init__(
        self,
        obs: dict[int, ObservableState],
        agent: Robot | Pedestrian,
        obstacles: dict[str, Obstacle],
    ) -> None:
        self.obs = obs
        self.agent = agent
        self.obstacles = obstacles

    def _check_collision(self) -> bool:
        if self.agent.pose is None:
            raise RuntimeError("Agent pose is missing.")
        if self.agent.velocity is None:
            raise RuntimeError("Agent velocity is missing.")

        collision_detected = self.checkIntrusionSAT()
        obstacle_collision_detector = ObstacleCollisionDetector()
        obstacle_collision_detected = (
            obstacle_collision_detector.collides_with_any_obstacle(
                self.agent, self.obstacles
            ).collided
        )

        return collision_detected or obstacle_collision_detected

    def checkIntrusionSAT(self) -> bool:
        TIME_HORIZON = 7.0
        EPS = 1e-8

        assert self.agent.pose is not None
        assert self.agent.velocity is not None
        agent_pos = np.array(
            [self.agent.pose.px, self.agent.pose.py],
            dtype=np.float64,
        )
        agent_vel = np.array(
            [self.agent.velocity.vx, self.agent.velocity.vy],
            dtype=np.float64,
        )

        observations = list(self.obs.values())
        if not observations:
            return False

        obs_radius = np.array([obs.radius for obs in observations], dtype=np.float64)
        obs_pos = np.array(
            [[obs.pose.px, obs.pose.py] for obs in observations],
            dtype=np.float64,
        )
        obs_vel = np.array(
            [[obs.velocity.vx, obs.velocity.vy] for obs in observations],
            dtype=np.float64,
        )

        safe_dist = (
            self.agent.radius
            + obs_radius
            + float(self.agent.config["safety"]["margin"])
        )

        agent_speed = np.linalg.norm(agent_vel)
        if agent_speed <= EPS:
            # Fallback for stationary agent: direct overlap / clearance check.
            dist = np.linalg.norm(obs_pos - agent_pos, axis=1)
            return bool((dist <= safe_dist).any())

        agent_vel_unit = agent_vel / agent_speed
        perp_vec = np.array([-agent_vel_unit[1], agent_vel_unit[0]], dtype=np.float64)

        rel_pos = obs_pos - agent_pos
        rel_vel = obs_vel - agent_vel

        proj_x = np.dot(rel_pos, agent_vel_unit)
        proj_y = np.dot(rel_pos, perp_vec)

        rel_vel_x = np.dot(rel_vel, agent_vel_unit)
        rel_vel_y = np.dot(rel_vel, perp_vec)

        entry_x, exit_x = self.sat_interval(proj_x, rel_vel_x, safe_dist)
        entry_y, exit_y = self.sat_interval(proj_y, rel_vel_y, safe_dist)

        entry = np.maximum(entry_x, entry_y)
        exit = np.minimum(exit_x, exit_y)

        collision = (entry <= exit) & (exit >= 0.0) & (entry <= TIME_HORIZON)
        return bool(collision.any())

    def sat_interval(
        self,
        center_projection: NDArray[np.float64],
        relative_velocity: NDArray[np.float64],
        safe_dist: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        EPS = 1e-8

        entry = np.empty_like(center_projection, dtype=np.float64)
        exit = np.empty_like(center_projection, dtype=np.float64)

        moving = np.abs(relative_velocity) > EPS

        t1 = (-safe_dist - center_projection) / relative_velocity
        t2 = (safe_dist - center_projection) / relative_velocity

        entry[moving] = np.minimum(t1[moving], t2[moving])
        exit[moving] = np.maximum(t1[moving], t2[moving])

        inside = np.abs(center_projection) <= safe_dist

        entry[~moving] = np.where(inside[~moving], -np.inf, np.inf)
        exit[~moving] = np.where(inside[~moving], np.inf, -np.inf)

        return entry, exit


@dataclass(slots=True)
class CollisionResult:
    collided: bool
    obstacle_index: int | None = None
    overlap: float = 0.0
    axis_x: float = 0.0
    axis_y: float = 0.0


class ObstacleCollisionDetector:
    """SAT-based collision detector for a circular agent against obstacles.
    Obstacles can be polygon, rectangle, or circle shapes as long as they
    are accepted by `obstacle_to_vertices(...)`.
    """

    EPS = 1e-12

    @classmethod
    def collides_with_any_obstacle(
        cls,
        agent: Robot | Pedestrian,
        obstacles: Sequence[Obstacle] | dict[Any, Obstacle],
    ) -> CollisionResult:
        if agent.pose is None:
            raise ValueError("Agent pose is None.")
        if getattr(agent, "radius", None) is None:
            raise ValueError("Agent radius is missing.")

        obstacle_items = (
            obstacles.values() if isinstance(obstacles, dict) else obstacles
        )
        circle = (float(agent.pose.px), float(agent.pose.py), float(agent.radius))

        best = CollisionResult(False)

        for idx, obstacle in enumerate(obstacle_items):
            vertices = obstacle_to_vertices(obstacle)
            result = cls.circle_vs_polygon(circle, vertices)
            if result.collided:
                result.obstacle_index = idx
                return result

            if result.overlap > best.overlap:
                best = CollisionResult(
                    collided=False,
                    obstacle_index=idx,
                    overlap=result.overlap,
                    axis_x=result.axis_x,
                    axis_y=result.axis_y,
                )

        return best

    @classmethod
    def circle_vs_polygon(
        cls,
        circle: tuple[float, float, float],
        polygon: Sequence[tuple[float, float]],
    ) -> CollisionResult:
        cx, cy, _ = circle

        if not polygon:
            return CollisionResult(False)

        min_overlap = float("inf")
        best_axis = (0.0, 0.0)

        # 1) Test all polygon edge normals
        n = len(polygon)
        for i in range(n):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % n]
            edge_x = bx - ax
            edge_y = by - ay
            axis_x, axis_y = cls._normalize(-edge_y, edge_x)
            if axis_x == 0.0 and axis_y == 0.0:
                continue

            overlap = cls._project_circle_polygon_overlap(
                circle, polygon, axis_x, axis_y
            )
            if overlap <= 0.0:
                return CollisionResult(False)
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = (axis_x, axis_y)

        # 2) Test axis from circle center to closest polygon point
        closest_x, closest_y = cls._closest_point_on_polygon(cx, cy, polygon)
        axis_x, axis_y = cls._normalize(cx - closest_x, cy - closest_y)
        if axis_x != 0.0 or axis_y != 0.0:
            overlap = cls._project_circle_polygon_overlap(
                circle, polygon, axis_x, axis_y
            )
            if overlap <= 0.0:
                return CollisionResult(False)
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = (axis_x, axis_y)

        axis_x, axis_y = cls._normalize(*best_axis)
        return CollisionResult(True, None, min_overlap, axis_x, axis_y)

    @staticmethod
    def _project_circle_polygon_overlap(
        circle: tuple[float, float, float],
        polygon: Sequence[tuple[float, float]],
        axis_x: float,
        axis_y: float,
    ) -> float:
        cx, cy, cr = circle

        circle_center_proj = cx * axis_x + cy * axis_y
        circle_min = circle_center_proj - cr
        circle_max = circle_center_proj + cr

        poly_projs = [px * axis_x + py * axis_y for px, py in polygon]
        poly_min = min(poly_projs)
        poly_max = max(poly_projs)

        return min(circle_max, poly_max) - max(circle_min, poly_min)

    @staticmethod
    def _closest_point_on_polygon(
        px: float,
        py: float,
        polygon: Sequence[tuple[float, float]],
    ) -> tuple[float, float]:
        best_dist = float("inf")
        best_point = polygon[0]

        for i in range(len(polygon)):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % len(polygon)]
            qx, qy = ObstacleCollisionDetector._closest_point_on_segment(
                px, py, ax, ay, bx, by
            )
            dist = (qx - px) ** 2 + (qy - py) ** 2
            if dist < best_dist:
                best_dist = dist
                best_point = (qx, qy)

        return best_point

    @staticmethod
    def _closest_point_on_segment(
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> tuple[float, float]:
        ab_x = bx - ax
        ab_y = by - ay
        ap_x = px - ax
        ap_y = py - ay
        ab_len_sq = ab_x * ab_x + ab_y * ab_y

        if ab_len_sq <= ObstacleCollisionDetector.EPS:
            return ax, ay

        t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
        t = max(0.0, min(1.0, t))
        return ax + t * ab_x, ay + t * ab_y

    @staticmethod
    def _normalize(x: float, y: float) -> tuple[float, float]:
        mag = hypot(x, y)
        if mag <= ObstacleCollisionDetector.EPS:
            return 0.0, 0.0
        return x / mag, y / mag

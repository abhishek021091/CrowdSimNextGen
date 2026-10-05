# tests/utils/metrics.py
"""Trajectory recording and metric computation for navigation regression tests."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from navcore.entities.environment.environment import Environment


@dataclass(frozen=True, slots=True)
class Thresholds:
    safe_human_distance: float = 0.15  # surface-to-surface, metres
    safe_obstacle_distance: float = 0.05
    max_oscillations: int = 10
    oscillation_velocity_threshold: float = 0.2
    deadlock_window_s: float = 12.0
    deadlock_min_displacement: float = 0.5
    min_path_efficiency: float = 0.5
    time_budget_factor: float = 2.5
    time_budget_offset_s: float = 5.0
    speed_tolerance: float = 1e-6


@dataclass(frozen=True, slots=True)
class PathFrame:
    """Frame aligned with the straight robot-start -> robot-goal line."""

    origin: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    length: float

    @classmethod
    def from_points(cls, start, goal) -> "PathFrame":
        origin = np.asarray(start, dtype=float)
        delta = np.asarray(goal, dtype=float) - origin
        length = float(np.hypot(delta[0], delta[1]))
        if length < 1e-9:
            raise ValueError("Robot start and goal coincide.")
        direction = delta / length
        normal = np.array([-direction[1], direction[0]])
        return cls(origin, direction, normal, length)

    def longitudinal(self, points) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.origin) @ self.direction

    def lateral(self, points) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.origin) @ self.normal


@dataclass(slots=True)
class Trajectory:
    dt: float
    robot_radius: float = 0.3
    human_ids: list[int] = field(default_factory=list)
    human_radii: np.ndarray = field(default_factory=lambda: np.zeros(0))
    robot_xy: list[tuple[float, float]] = field(default_factory=list)
    robot_velocity: list[tuple[float, float]] = field(default_factory=list)
    human_xy: list[np.ndarray] = field(default_factory=list)

    def record(self, env: Environment) -> None:
        robot = env.robot
        assert robot.pose is not None and robot.velocity is not None
        if not self.human_xy:
            self.human_ids = sorted(env.crowd)
            self.human_radii = np.array(
                [env.crowd[i].radius for i in self.human_ids], dtype=float
            )
            self.robot_radius = float(robot.radius)
        self.robot_xy.append((float(robot.pose.px), float(robot.pose.py)))
        self.robot_velocity.append((float(robot.velocity.vx), float(robot.velocity.vy)))
        self.human_xy.append(
            np.array(
                [[env.crowd[i].pose.px, env.crowd[i].pose.py] for i in self.human_ids],
                dtype=float,
            ).reshape(-1, 2)
        )

    def __len__(self) -> int:
        return len(self.robot_xy)

    def robot_array(self) -> np.ndarray:
        return np.asarray(self.robot_xy, dtype=float)

    def velocity_array(self) -> np.ndarray:
        return np.asarray(self.robot_velocity, dtype=float)

    def human_array(self) -> np.ndarray:
        return np.stack(self.human_xy) if self.human_xy else np.zeros((0, 0, 2))


class DeadlockDetector:
    """Flags a deadlock when the robot barely moves over a sliding window."""

    def __init__(self, window_steps: int, min_displacement: float) -> None:
        self._positions: deque[tuple[float, float]] = deque(maxlen=window_steps + 1)
        self._window_steps = window_steps
        self._min_displacement = min_displacement

    def update(self, position: tuple[float, float]) -> bool:
        self._positions.append(position)
        if len(self._positions) <= self._window_steps:
            return False
        oldest, newest = self._positions[0], self._positions[-1]
        return math.hypot(newest[0] - oldest[0], newest[1] - oldest[1]) < (
            self._min_displacement
        )


def count_reversals(series: np.ndarray, threshold: float) -> int:
    """Sign reversals of a signal, ignoring samples below ``threshold``."""
    last = 0
    count = 0
    for value in series:
        if abs(value) < threshold:
            continue
        sign = 1 if value > 0 else -1
        if last and sign != last:
            count += 1
        last = sign
    return count


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    termination: str
    steps: int
    collision: bool
    goal_reached: bool
    timed_out: bool
    left_map: bool
    deadlock: bool
    time_to_goal: float | None
    min_human_distance: float
    min_obstacle_distance: float
    path_length: float
    path_efficiency: float
    average_speed: float
    max_speed: float
    max_lateral_deviation: float
    oscillation_count: int

    def summary(self) -> str:
        ttg = "n/a" if self.time_to_goal is None else f"{self.time_to_goal:.1f}s"
        return (
            f"termination={self.termination} steps={self.steps} "
            f"collision={self.collision} goal={self.goal_reached} "
            f"timeout={self.timed_out} left_map={self.left_map} "
            f"deadlock={self.deadlock} time_to_goal={ttg} "
            f"min_human_dist={self.min_human_distance:.3f} "
            f"min_obstacle_dist={self.min_obstacle_distance:.3f} "
            f"path_len={self.path_length:.2f} eff={self.path_efficiency:.3f} "
            f"avg_speed={self.average_speed:.3f} max_speed={self.max_speed:.3f} "
            f"max_lat_dev={self.max_lateral_deviation:.2f} "
            f"oscillations={self.oscillation_count}"
        )


def compute_episode_metrics(
    trajectory: Trajectory,
    frame: PathFrame,
    obstacle_polygons: list,
    termination: str,
    oscillation_threshold: float,
) -> EpisodeMetrics:
    robot = trajectory.robot_array()
    velocity = trajectory.velocity_array()
    humans = trajectory.human_array()
    steps = len(robot) - 1
    elapsed = steps * trajectory.dt

    segments = np.diff(robot, axis=0)
    path_length = float(np.hypot(segments[:, 0], segments[:, 1]).sum())
    speeds = np.hypot(velocity[:, 0], velocity[:, 1])

    if humans.shape[1] > 0:
        distances = (
            np.hypot(
                humans[..., 0] - robot[:, None, 0],
                humans[..., 1] - robot[:, None, 1],
            )
            - trajectory.human_radii[None, :]
            - trajectory.robot_radius
        )
        min_human = float(distances.min())
    else:
        min_human = math.inf

    if obstacle_polygons:
        union = unary_union(obstacle_polygons)
        min_obstacle = (
            min(Point(x, y).distance(union) for x, y in robot) - trajectory.robot_radius
        )
    else:
        min_obstacle = math.inf

    longitudinal_v = velocity @ frame.direction
    lateral_v = velocity @ frame.normal
    oscillations = count_reversals(
        longitudinal_v, oscillation_threshold
    ) + count_reversals(lateral_v, oscillation_threshold)

    goal_reached = termination == "goal"
    return EpisodeMetrics(
        termination=termination,
        steps=steps,
        collision=termination == "collision",
        goal_reached=goal_reached,
        timed_out=termination == "timeout",
        left_map=termination == "out_of_bounds",
        deadlock=termination == "deadlock",
        time_to_goal=elapsed if goal_reached else None,
        min_human_distance=min_human,
        min_obstacle_distance=float(min_obstacle),
        path_length=path_length,
        path_efficiency=frame.length / path_length if path_length > 1e-9 else 0.0,
        average_speed=path_length / elapsed if elapsed > 0 else 0.0,
        max_speed=float(speeds.max()),
        max_lateral_deviation=float(np.abs(frame.lateral(robot)).max()),
        oscillation_count=oscillations,
    )

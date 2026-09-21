"""ObservationEncoder: turns live Environment state into a fixed-shape
Gymnasium observation for the robot.
...
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from shapely.geometry import LinearRing
from shapely.geometry import Polygon as ShapelyPolygon

from navcore.entities.components.sensors.obstacle_detector import (
    RAY_FEATURE_DIM,
    ObstacleDetector,
    ObstacleDetectorConfig,
    scan_to_features,
)
from navcore.entities.components.state import ObservableState
from navcore.entities.environment.environment import Environment
from navcore.entities.obstacles.geometry_conversion import (
    arena_boundary_ring,
    obstacle_to_shapely_polygon,
)

_NEIGHBOR_FEATURES = 5
_ROBOT_FEATURES = 8


class ObservationEncoder:
    """Encodes ``Environment`` into a fixed-shape observation for the robot.

    Attributes:
        max_neighbors: Fixed neighbor-slot count.
        obstacle_detector: Ray-casting sensor used to build
            "ray_features". Stateless itself (see its own docstring);
            this class owns the per-episode obstacle/boundary geometry
            it's cast against.
    """

    def __init__(
        self,
        max_neighbors: int,
        history_steps: int = 8,
        obstacle_detector_config: ObstacleDetectorConfig | None = None,
    ) -> None:
        if max_neighbors <= 0:
            raise ValueError(f"max_neighbors must be positive, got {max_neighbors!r}.")
        if history_steps <= 0:
            raise ValueError(f"history_steps must be positive, got {history_steps!r}.")
        self.max_neighbors = max_neighbors
        self.history_steps = history_steps
        self._neighbor_history: dict[int, deque[np.ndarray]] = {}

        self.obstacle_detector = ObstacleDetector(obstacle_detector_config)
        # Static per-episode obstacle/boundary geometry, in world-frame
        # shapely form -- built once per episode (see reset()), not
        # every encode() call. Obstacles are static within an episode
        # (EnvironmentBuilder only rebuilds them at reset()), so
        # reconstructing these shapely Polygons every tick would waste
        # ~num_obstacles Polygon constructions on every one of n_steps
        # ticks, across every parallel env, for geometry that never
        # changes between resets.
        self._cached_obstacle_polygons: list[ShapelyPolygon] = []
        self._cached_boundary_ring: LinearRing | None = None

    def reset(self, env: Environment | None = None) -> None:
        """Discard temporal state at an episode boundary.

        Args:
            env: The new episode's Environment. When given, also
                refreshes the cached obstacle/boundary geometry used
                for ray-casting. Callers that only need the
                temporal-history reset (e.g. ``PolicyFieldVisualizer``,
                which re-queries the same env's fixed obstacle layout
                many times per call) may omit it -- the geometry cache
                is left untouched.
        """
        self._neighbor_history.clear()
        if env is not None:
            self._cached_obstacle_polygons = [
                obstacle_to_shapely_polygon(obstacle)
                for key, obstacle in env.obstacles.items()
                if key != "boundary"
            ]
            self._cached_boundary_ring = arena_boundary_ring(env)

    @property
    def space(self) -> spaces.Dict:
        inf = np.float32(np.inf)
        return spaces.Dict(
            {
                "robot": spaces.Box(
                    -inf, inf, shape=(_ROBOT_FEATURES,), dtype=np.float32
                ),
                "neighbors": spaces.Box(
                    -inf,
                    inf,
                    shape=(self.max_neighbors, _NEIGHBOR_FEATURES),
                    dtype=np.float32,
                ),
                "neighbor_mask": spaces.MultiBinary(self.max_neighbors),
                "neighbor_history": spaces.Box(
                    -inf,
                    inf,
                    shape=(
                        self.history_steps,
                        self.max_neighbors,
                        _NEIGHBOR_FEATURES,
                    ),
                    dtype=np.float32,
                ),
                "neighbor_history_mask": spaces.MultiBinary(
                    (self.history_steps, self.max_neighbors)
                ),
                "ray_features": spaces.Box(
                    -inf,
                    inf,
                    shape=(self.obstacle_detector.config.num_rays, RAY_FEATURE_DIM),
                    dtype=np.float32,
                ),
            }
        )

    def encode(self, env: Environment) -> dict[str, npt.NDArray[np.float32]]:
        robot = env.robot
        if robot.pose is None or robot.velocity is None or robot.goal is None:
            raise RuntimeError(
                "Robot must have pose, velocity, and goal set before encoding."
            )
        if robot.sensor is None:
            raise RuntimeError("Robot sensor must be initialized before encoding.")

        if self._cached_boundary_ring is None:
            # Defensive fallback for a caller that never called
            # reset(env) -- normal CrowdSimEnv usage always does, so
            # this path shouldn't fire in the training loop.
            self.reset(env)

        neighbor_obs = robot.sensor.observe(env, robot_visible=False)

        robot_features: npt.NDArray[np.float32] = np.array(
            [
                robot.goal.gx - robot.pose.px,
                robot.goal.gy - robot.pose.py,
                robot.velocity.vx,
                robot.velocity.vy,
                robot.v_pref,
                robot.radius,
                math.cos(robot.pose.theta),
                math.sin(robot.pose.theta),
            ],
            dtype=np.float32,
        )

        neighbors, mask, history, history_mask = self._encode_neighbors(
            robot.pose.px, robot.pose.py, neighbor_obs
        )
        ray_features = self._encode_ray_features(robot.pose.px, robot.pose.py)

        return {
            "robot": robot_features,
            "neighbors": neighbors,
            "neighbor_mask": mask,
            "neighbor_history": history,
            "neighbor_history_mask": history_mask,
            "ray_features": ray_features,
        }

    def _encode_ray_features(
        self, robot_x: float, robot_y: float
    ) -> npt.NDArray[np.float32]:
        """Cast this tick's ray fan and convert it to CNN-ready features.

        heading fixed at 0.0 -- world-frame ray fan, not
        robot-heading-relative. Resolves ObstacleDetector's own open
        question #1: the robot is holonomic (robot.toml's
        `chassis = "holonomic"`), so there's no orientation-constrained
        motion a heading-relative fan needs to track, and
        `scan_to_features()` independently assumes ray 0 sits at angle
        0 -- consistent with heading=0.0, silently wrong otherwise. Do
        not change this without also fixing `scan_to_features()` to
        read `scan.ray_angles`.
        """
        scan = self.obstacle_detector.sense(
            robot_x,
            robot_y,
            self._cached_obstacle_polygons,
            boundary=self._cached_boundary_ring,
            heading=0.0,
        )
        return scan_to_features(scan, self.obstacle_detector.config.max_range)

    # _encode_neighbors() and _neighbor_features() unchanged
    def _encode_neighbors(
        self,
        robot_x: float,
        robot_y: float,
        observation: dict[int, ObservableState],
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
    ]:
        def distance(obs: ObservableState) -> float:
            return math.hypot(obs.pose.px - robot_x, obs.pose.py - robot_y)

        features_by_id = {
            pedestrian_id: self._neighbor_features(robot_x, robot_y, obs)
            for pedestrian_id, obs in observation.items()
        }
        for pedestrian_id, features in features_by_id.items():
            track = self._neighbor_history.setdefault(
                pedestrian_id, deque(maxlen=self.history_steps)
            )
            track.append(features)

        nearest = sorted(observation.items(), key=lambda item: distance(item[1]))[
            : self.max_neighbors
        ]

        neighbors = np.zeros((self.max_neighbors, _NEIGHBOR_FEATURES), dtype=np.float32)
        mask = np.zeros(self.max_neighbors, dtype=np.int8)
        history = np.zeros(
            (self.history_steps, self.max_neighbors, _NEIGHBOR_FEATURES),
            dtype=np.float32,
        )
        history_mask = np.zeros((self.history_steps, self.max_neighbors), dtype=np.int8)

        for i, (pedestrian_id, _) in enumerate(nearest):
            neighbors[i] = features_by_id[pedestrian_id]
            mask[i] = 1
            track = self._neighbor_history[pedestrian_id]
            start = self.history_steps - len(track)
            history[start:, i] = track
            history_mask[start:, i] = 1

        return neighbors, mask, history, history_mask

    @staticmethod
    def _neighbor_features(
        robot_x: float, robot_y: float, obs: ObservableState
    ) -> np.ndarray:
        return np.asarray(
            (
                obs.pose.px - robot_x,
                obs.pose.py - robot_y,
                obs.velocity.vx,
                obs.velocity.vy,
                obs.radius,
            ),
            dtype=np.float32,
        )

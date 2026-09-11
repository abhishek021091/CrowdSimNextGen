"""ObservationEncoder: turns live Environment state into a fixed-shape
Gymnasium observation for the robot.

Kept separate from CrowdSimEnv so the encoding scheme can be swapped or
unit-tested independently of episode/reward machinery.

Design choices:
    - Reads the robot's actual RangeSensor observation, not env.crowd
      directly. Using ground truth here would violate the same
      information boundary already enforced for ORCA (robot only sees
      ObservableState -- pose, velocity, radius -- never a neighbor's
      goal or intent).
    - Neighbor position is given relative to the robot (absolute
      velocity is kept as-is); a policy trained on relative geometry
      generalizes across arbitrary start/goal placements, whereas
      world-frame coordinates would overfit to this arena's layout.
    - Neighbors are sorted by distance ascending and truncated/
      zero-padded to a fixed max_neighbors. A neighbor_mask array is
      included alongside the padded block: zero-padding alone cannot
      be told apart from a real neighbor sitting at relative (0, 0),
      and the mask removes that ambiguity for one bit per slot.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np
from gymnasium import spaces

from navcore.entities.components.state import ObservableState
from navcore.entities.environment.environment import Environment

#: Per-neighbor feature layout: [rel_px, rel_py, vx, vy, radius].
_NEIGHBOR_FEATURES = 5
#: Robot self-feature layout:
#: [rel_goal_x, rel_goal_y, vx, vy, v_pref, radius, cos(theta), sin(theta)].
_ROBOT_FEATURES = 8


class ObservationEncoder:
    """Encodes ``Environment`` into a fixed-shape observation for the robot.

    Attributes:
        max_neighbors: Fixed neighbor-slot count. When more than
            max_neighbors are within sensor range, the closest ones are
            kept and farther ones are dropped.
    """

    def __init__(self, max_neighbors: int, history_steps: int = 8) -> None:
        if max_neighbors <= 0:
            raise ValueError(f"max_neighbors must be positive, got {max_neighbors!r}.")
        if history_steps <= 0:
            raise ValueError(f"history_steps must be positive, got {history_steps!r}.")
        self.max_neighbors = max_neighbors
        self.history_steps = history_steps
        self._neighbor_history: dict[int, deque[np.ndarray]] = {}

    def reset(self) -> None:
        """Discard all temporal state at an episode boundary."""
        self._neighbor_history.clear()

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
            }
        )

    def encode(self, env: Environment) -> dict[str, np.ndarray]:
        robot = env.robot
        if robot.pose is None or robot.velocity is None or robot.goal is None:
            raise RuntimeError(
                "Robot must have pose, velocity, and goal set before encoding."
            )
        if robot.sensor is None:
            raise RuntimeError("Robot sensor must be initialized before encoding.")

        # robot_visible only affects whether *pedestrians* see the
        # robot in their own sensor calls -- it has no effect when the
        # observing agent is the robot itself. Passed as False purely
        # for interface compliance with RangeSensor.observe's signature.
        neighbor_obs = robot.sensor.observe(env, robot_visible=False)

        robot_features = np.array(
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

        return {
            "robot": robot_features,
            "neighbors": neighbors,
            "neighbor_mask": mask,
            "neighbor_history": history,
            "neighbor_history_mask": history_mask,
        }

    def _encode_neighbors(
        self,
        robot_x: float,
        robot_y: float,
        observation: dict[int, ObservableState],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        nearest = sorted(
            observation.items(), key=lambda item: distance(item[1])
        )[: self.max_neighbors]

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

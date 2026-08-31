from __future__ import annotations

import math
from collections import deque
from typing import Any

from ..state import SceneSnapshot

METRIC_KEYS = (
    "distance_to_goal",
    "speed",
    "clearance",
    "reward",
    "collisions",
    "discomfort",
    "orca_solve_time",
    "episode_reward",
)

METRIC_LABELS = {
    "distance_to_goal": "Dist to goal (m)",
    "speed": "Robot speed (m/s)",
    "clearance": "Min clearance (m)",
    "reward": "Step reward",
    "collisions": "Cumulative collisions",
    "discomfort": "Discomfort",
    "orca_solve_time": "ORCA solve (ms)",
    "episode_reward": "Episode reward (cum.)",
}


class MetricsPanel:
    """A small grid of live line plots for research-style diagnostics.

    Metrics computable from the scene alone (distance-to-goal, speed,
    clearance) are always populated. RL/planner-specific metrics
    (reward, discomfort, ORCA solve time, episode reward) are optional
    and read from ``scene.extra["metrics"]`` -- a plain dict the
    stepper/environment can populate each step; missing keys simply
    leave that subplot flat at zero rather than erroring.
    """

    def __init__(self, axes: dict[str, Any], scheme: Any, history: int = 300) -> None:
        self.axes = axes
        self.scheme = scheme
        self.history = history
        self._data: dict[str, deque[float]] = {k: deque(maxlen=history) for k in METRIC_KEYS}
        self._lines: dict[str, Any] = {}
        self._episode_reward_running = 0.0
        self._cumulative_collisions = 0

        for key, ax in axes.items():
            ax.set_facecolor(scheme.panel_face)
            ax.tick_params(labelsize=6, colors=scheme.text_color)
            ax.set_title(METRIC_LABELS.get(key, key), fontsize=7.5, color=scheme.text_color, loc="left")
            for spine in ax.spines.values():
                spine.set_color(scheme.panel_edge)
            (line,) = ax.plot([], [], linewidth=1.1, color=scheme.robot_edge)
            self._lines[key] = line

    def set_scheme(self, scheme: Any) -> None:
        self.scheme = scheme
        for key, ax in self.axes.items():
            ax.set_facecolor(scheme.panel_face)
            ax.tick_params(colors=scheme.text_color)
            ax.title.set_color(scheme.text_color)
            for spine in ax.spines.values():
                spine.set_color(scheme.panel_edge)

    def reset(self) -> None:
        for d in self._data.values():
            d.clear()
        self._episode_reward_running = 0.0
        self._cumulative_collisions = 0

    def update(self, scene: SceneSnapshot, collisions_so_far: int) -> None:
        robot = scene.robot
        speed = math.hypot(robot.vx, robot.vy) if robot else 0.0
        dist_to_goal = (
            math.hypot(robot.goal_x - robot.x, robot.goal_y - robot.y)
            if robot and robot.goal_x is not None else 0.0
        )
        clearance = min(
            (math.hypot(p.x - robot.x, p.y - robot.y) - p.radius - robot.radius
             for p in scene.pedestrians.values()),
            default=math.inf,
        ) if robot else math.inf
        clearance = 0.0 if clearance == math.inf else clearance

        external = scene.extra.get("metrics", {}) or {}
        reward = float(external.get("reward", 0.0))
        self._episode_reward_running += reward

        values = {
            "distance_to_goal": dist_to_goal,
            "speed": speed,
            "clearance": clearance,
            "reward": reward,
            "collisions": float(collisions_so_far),
            "discomfort": float(external.get("discomfort", 0.0)),
            "orca_solve_time": float(external.get("orca_solve_time_ms", 0.0)),
            "episode_reward": self._episode_reward_running,
        }
        for key in METRIC_KEYS:
            if key not in self.axes:
                continue
            self._data[key].append(values[key])
            ys = list(self._data[key])
            xs = list(range(len(ys)))
            self._lines[key].set_data(xs, ys)
            ax = self.axes[key]
            ax.set_xlim(0, max(10, len(ys)))
            lo, hi = min(ys), max(ys)
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            pad = 0.1 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

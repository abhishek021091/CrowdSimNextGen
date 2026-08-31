from __future__ import annotations

import math
import time
from typing import Any

from ..state import SceneSnapshot


class StatisticsPanel:
    """A dedicated Axes rendered as a borderless text panel.

    Kept as its own Axes (rather than an ``ax.text`` box on the main
    scene) so it can host a monospace, left-aligned block without
    fighting the scene's own transform/zorder concerns.
    """

    def __init__(self, ax: Any, scheme: Any, goal_threshold: float = 0.1) -> None:
        self.ax = ax
        self.scheme = scheme
        self.goal_threshold = goal_threshold
        self.collisions = 0
        self.near_collisions = 0
        self.min_distance = math.inf
        self.near_collision_threshold = 0.3
        self._last_frame_time = time.perf_counter()
        self._fps = 0.0
        self._planner_latency_ms = 0.0
        self._cpu_ms = 0.0

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self._text = self.ax.text(
            0.05, 0.97, "", transform=self.ax.transAxes, va="top", ha="left",
            fontsize=9.5, family="monospace", color=scheme.text_color,
        )

    def set_scheme(self, scheme: Any) -> None:
        self.scheme = scheme
        self._text.set_color(scheme.text_color)
        self.ax.set_facecolor(scheme.panel_face)

    def mark_frame_start(self) -> float:
        return time.perf_counter()

    def mark_frame_end(self, start: float) -> None:
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps = 0.85 * self._fps + 0.15 * (1.0 / dt)
        self._cpu_ms = (now - start) * 1000.0

    def set_planner_latency_ms(self, value: float) -> None:
        self._planner_latency_ms = value

    def _update_collision_stats(self, scene: SceneSnapshot) -> None:
        if scene.robot is None:
            return
        min_d = math.inf
        for ped in scene.pedestrians.values():
            surface_dist = math.hypot(ped.x - scene.robot.x, ped.y - scene.robot.y)
            surface_dist -= ped.radius + scene.robot.radius
            min_d = min(min_d, surface_dist)
        if min_d is math.inf:
            return
        self.min_distance = min(self.min_distance, min_d)
        if min_d <= 0:
            self.collisions += 1
        elif min_d <= self.near_collision_threshold:
            self.near_collisions += 1

    def update(self, scene: SceneSnapshot, env: Any) -> None:
        self._update_collision_stats(scene)

        robot = scene.robot
        speed = math.hypot(robot.vx, robot.vy) if robot else 0.0
        heading_deg = math.degrees(robot.heading) if robot else 0.0
        dist_to_goal = (
            math.hypot(robot.goal_x - robot.x, robot.goal_y - robot.y)
            if robot and robot.goal_x is not None else None
        )
        eta = (dist_to_goal / speed) if (dist_to_goal is not None and speed > 1e-3) else None

        crowd_speeds = [math.hypot(p.vx, p.vy) for p in scene.pedestrians.values()]
        avg_crowd_speed = sum(crowd_speeds) / len(crowd_speeds) if crowd_speeds else 0.0

        lines = [
            "SIMULATION",
            f"  time        {scene.sim_time:8.2f} s",
            f"  step        {scene.step:8d}",
            f"  fps         {self._fps:8.1f}",
            "",
            "ROBOT",
            f"  speed       {speed:8.2f} m/s",
            f"  heading     {heading_deg:8.1f} deg",
            f"  dist->goal  {dist_to_goal if dist_to_goal is not None else float('nan'):8.2f} m",
            f"  eta         {eta if eta is not None else float('nan'):8.1f} s",
            "",
            "CROWD",
            f"  count       {len(scene.pedestrians):8d}",
            f"  avg speed   {avg_crowd_speed:8.2f} m/s",
            "",
            "SAFETY",
            f"  collisions  {self.collisions:8d}",
            f"  near-miss   {self.near_collisions:8d}",
            f"  min dist    {self.min_distance if self.min_distance != math.inf else float('nan'):8.2f} m",
            "",
            "PERFORMANCE",
            f"  cpu/frame   {self._cpu_ms:8.1f} ms",
            f"  planner     {self._planner_latency_ms:8.1f} ms",
        ]
        text = "\n".join(l.replace("nan", " n/a") for l in lines)
        self._text.set_text(text)

    def reset(self) -> None:
        self.collisions = 0
        self.near_collisions = 0
        self.min_distance = math.inf

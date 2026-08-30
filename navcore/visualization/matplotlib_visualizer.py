from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon

from navcore.policies.base_orca_planner import obstacle_to_vertices


@dataclass(slots=True)
class TrailConfig:
    enabled: bool = True
    length: int = 30


@dataclass(slots=True)
class GroupVisual:
    color: Any
    leader_id: int | None = None


class MatplotlibVisualizer:
    """Visualize a navcore environment in a CrowdNav-like style.

    Design choices:
    - one color per group
    - leader and followers of the same group share the same color
    - leaders are visually emphasized with a thicker outline
    - goals are shown with matching color x markers
    - trails show recent motion
    - robot remains distinct from pedestrians
    - obstacles are drawn as light gray polygons
    """

    ROBOT_COLOR = "tab:red"
    ROBOT_GOAL_COLOR = "tab:red"
    OBSTACLE_EDGE_COLOR = "0.35"
    OBSTACLE_FACE_COLOR = "0.90"
    UNGROUPED_COLOR = "0.45"
    LEADER_EDGE_COLOR = "black"

    def __init__(
        self,
        env: Any,
        stepper: Any | None = None,
        *,
        interval: int = 80,
        trail_length: int = 30,
        padding: float = 1.5,
        title: str = "CrowdSimNextGen Visualization",
        show_ids: bool = True,
        show_status: bool = True,
    ) -> None:
        self.env = env
        self.stepper = stepper
        self.interval = interval
        self.padding = padding
        self.title = title
        self.show_ids = show_ids
        self.show_status = show_status
        self.trails = TrailConfig(enabled=True, length=trail_length)

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.2)

        self._robot_patch: MplCircle | None = None
        self._robot_goal_artist = None
        self._robot_id_artist = None
        (self._robot_trail,) = self.ax.plot(
            [], [], linestyle="--", linewidth=1.2, color=self.ROBOT_COLOR
        )

        self._ped_patches: dict[int, MplCircle] = {}
        self._ped_goal_artists: dict[int, Any] = {}
        self._ped_id_artists: dict[int, Any] = {}
        self._ped_trails: dict[int, Any] = {}

        self._robot_history: deque[tuple[float, float]] = deque(maxlen=trail_length)
        self._ped_history: dict[int, deque[tuple[float, float]]] = {}

        self._obstacle_patches: list[Any] = []
        self._status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        self._anim: FuncAnimation | None = None
        self._paused = False
        self._closed = False
        self._step_count = 0

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self._setup_scene()

    def _on_close(self, _event: Any) -> None:
        self._closed = True
        if self._anim is not None and self._anim.event_source is not None:
            self._anim.event_source.stop()

    def _groups(self) -> dict[int, Any]:
        groups_obj = getattr(self.env, "groups", None)
        if not groups_obj:
            return {}
        if isinstance(groups_obj, dict):
            return groups_obj
        return {
            getattr(group, "id", idx): group for idx, group in enumerate(groups_obj)
        }

    def _group_map(self) -> dict[int, int]:
        member_to_group: dict[int, int] = {}
        for group_id, group in self._groups().items():
            member_ids = getattr(group, "member_ids", ())
            for member_id in member_ids:
                member_to_group[int(member_id)] = int(group_id)
        return member_to_group

    def _group_visuals(self) -> dict[int, GroupVisual]:
        visuals: dict[int, GroupVisual] = {}
        for group_id, group in self._groups().items():
            visuals[int(group_id)] = GroupVisual(
                color=plt.get_cmap("tab20")(int(group_id) % plt.get_cmap("tab20").N),
                leader_id=int(getattr(group, "leader_id", -1))
                if getattr(group, "leader_id", None) is not None
                else None,
            )
        return visuals

    def _group_color(self, group_id: int) -> Any:
        cmap = plt.get_cmap("tab20")
        return cmap(group_id % cmap.N)

    def _ped_style(self, ped_id: int) -> tuple[Any, bool]:
        group_map = self._group_map()
        if ped_id not in group_map:
            return self.UNGROUPED_COLOR, False
        group_id = group_map[ped_id]
        visuals = self._group_visuals()
        gv = visuals.get(group_id)
        return (
            gv.color if gv is not None else self._group_color(group_id)
        ), ped_id == (gv.leader_id if gv is not None else None)

    def _setup_scene(self) -> None:
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_title(self.title)

        self._draw_obstacles()
        self._draw_agents()
        self._fit_bounds()
        self._draw_legend()

    def _draw_legend(self) -> None:
        self.ax.plot([], [], "o", label="Robot", color=self.ROBOT_COLOR)
        group_ids = sorted(self._groups().keys())
        for group_id in group_ids:
            self.ax.plot(
                [],
                [],
                "o",
                label=f"Group {group_id}",
                color=self._group_color(group_id),
            )
        if not group_ids:
            self.ax.plot([], [], "o", label="Pedestrian", color=self.UNGROUPED_COLOR)
        self.ax.plot([], [], "x", label="Goal", color="black")
        self.ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    def _draw_obstacles(self) -> None:
        self._obstacle_patches.clear()
        obstacles = getattr(self.env, "obstacles", {}) or {}
        for obstacle in obstacles.values():
            vertices = obstacle_to_vertices(obstacle)
            patch = MplPolygon(
                vertices,
                closed=True,
                fill=True,
                facecolor=self.OBSTACLE_FACE_COLOR,
                edgecolor=self.OBSTACLE_EDGE_COLOR,
                linewidth=1.2,
                alpha=0.7,
            )
            self.ax.add_patch(patch)
            self._obstacle_patches.append(patch)

    def _draw_agents(self) -> None:
        robot = self.env.robot
        if robot.pose is not None:
            self._robot_patch = MplCircle(
                (robot.pose.px, robot.pose.py),
                robot.radius,
                facecolor=self.ROBOT_COLOR,
                edgecolor="black",
                linewidth=1.8,
                alpha=0.35,
            )
            self.ax.add_patch(self._robot_patch)
            self._robot_history.append((robot.pose.px, robot.pose.py))
            if robot.goal is not None:
                (self._robot_goal_artist,) = self.ax.plot(
                    [robot.goal.gx],
                    [robot.goal.gy],
                    marker="x",
                    markersize=10,
                    linestyle="None",
                    color=self.ROBOT_GOAL_COLOR,
                    markeredgewidth=2.0,
                )
            if self.show_ids:
                self._robot_id_artist = self.ax.text(
                    robot.pose.px,
                    robot.pose.py,
                    "R",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    weight="bold",
                )

        crowd = getattr(self.env, "crowd", {}) or {}
        for ped_id, ped in crowd.items():
            self._ped_history[ped_id] = deque(maxlen=self.trails.length)
            color, is_leader = self._ped_style(ped_id)

            if ped.pose is not None:
                patch = MplCircle(
                    (ped.pose.px, ped.pose.py),
                    ped.radius,
                    facecolor=color,
                    edgecolor=self.LEADER_EDGE_COLOR if is_leader else color,
                    linewidth=2.4 if is_leader else 1.4,
                    alpha=0.38 if is_leader else 0.28,
                )
                self.ax.add_patch(patch)
                self._ped_patches[ped_id] = patch
                self._ped_history[ped_id].append((ped.pose.px, ped.pose.py))
                (trail,) = self.ax.plot(
                    [], [], linestyle="--", linewidth=1.0, color=color, alpha=0.9
                )
                self._ped_trails[ped_id] = trail

                if self.show_ids:
                    self._ped_id_artists[ped_id] = self.ax.text(
                        ped.pose.px,
                        ped.pose.py,
                        f"{ped_id}{'L' if is_leader else ''}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                        weight="bold" if is_leader else "normal",
                    )

            if ped.goal is not None:
                (goal_artist,) = self.ax.plot(
                    [ped.goal.gx],
                    [ped.goal.gy],
                    marker="x",
                    markersize=8,
                    linestyle="None",
                    color=color,
                    markeredgewidth=1.8,
                )
                self._ped_goal_artists[ped_id] = goal_artist

    def _fit_bounds(self) -> None:
        xs: list[float] = []
        ys: list[float] = []

        robot = self.env.robot
        if robot.pose is not None:
            xs.append(float(robot.pose.px))
            ys.append(float(robot.pose.py))
        if robot.goal is not None:
            xs.append(float(robot.goal.gx))
            ys.append(float(robot.goal.gy))

        for ped in getattr(self.env, "crowd", {}).values():
            if ped.pose is not None:
                xs.append(float(ped.pose.px))
                ys.append(float(ped.pose.py))
            if ped.goal is not None:
                xs.append(float(ped.goal.gx))
                ys.append(float(ped.goal.gy))

        for obstacle in getattr(self.env, "obstacles", {}).values():
            vertices = obstacle_to_vertices(obstacle)
            for x, y in vertices:
                xs.append(float(x))
                ys.append(float(y))

        if not xs or not ys:
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            return

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        xpad = max(self.padding, 0.1 * max(1.0, xmax - xmin))
        ypad = max(self.padding, 0.1 * max(1.0, ymax - ymin))

        self.ax.set_xlim(xmin - xpad, xmax + xpad)
        self.ax.set_ylim(ymin - ypad, ymax + ypad)

    def _agent_at_goal(self, pose: Any, goal: Any, threshold: float = 0.1) -> bool:
        if pose is None or goal is None:
            return False
        return hypot(goal.gx - pose.px, goal.gy - pose.py) <= threshold

    def _update_robot(self) -> None:
        robot = self.env.robot
        if robot.pose is None:
            return

        if self._robot_patch is None:
            self._robot_patch = MplCircle(
                (robot.pose.px, robot.pose.py),
                robot.radius,
                facecolor=self.ROBOT_COLOR,
                edgecolor="black",
                linewidth=1.8,
                alpha=0.35,
            )
            self.ax.add_patch(self._robot_patch)
        else:
            self._robot_patch.center = (robot.pose.px, robot.pose.py)
            self._robot_patch.radius = robot.radius

        self._robot_history.append((float(robot.pose.px), float(robot.pose.py)))
        rx, ry = zip(*self._robot_history) if self._robot_history else ([], [])
        self._robot_trail.set_data(rx, ry)

        if robot.goal is not None:
            if self._robot_goal_artist is None:
                (self._robot_goal_artist,) = self.ax.plot(
                    [robot.goal.gx],
                    [robot.goal.gy],
                    marker="x",
                    markersize=10,
                    linestyle="None",
                    color=self.ROBOT_GOAL_COLOR,
                    markeredgewidth=2.0,
                )
            else:
                self._robot_goal_artist.set_data([robot.goal.gx], [robot.goal.gy])

        if self.show_ids:
            if self._robot_id_artist is None:
                self._robot_id_artist = self.ax.text(
                    robot.pose.px,
                    robot.pose.py,
                    "R",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    weight="bold",
                )
            else:
                self._robot_id_artist.set_position((robot.pose.px, robot.pose.py))

    def _update_crowd(self) -> None:
        crowd = getattr(self.env, "crowd", {}) or {}

        for ped_id in list(self._ped_patches):
            if ped_id not in crowd:
                self._ped_patches[ped_id].remove()
                self._ped_patches.pop(ped_id, None)
                self._ped_trails.pop(ped_id, None)
                if ped_id in self._ped_goal_artists:
                    self._ped_goal_artists[ped_id].remove()
                    self._ped_goal_artists.pop(ped_id, None)
                if ped_id in self._ped_id_artists:
                    self._ped_id_artists[ped_id].remove()
                    self._ped_id_artists.pop(ped_id, None)
                self._ped_history.pop(ped_id, None)

        for ped_id, ped in crowd.items():
            if ped.pose is None:
                continue

            color, is_leader = self._ped_style(ped_id)

            if ped_id not in self._ped_patches:
                patch = MplCircle(
                    (ped.pose.px, ped.pose.py),
                    ped.radius,
                    facecolor=color,
                    edgecolor=self.LEADER_EDGE_COLOR if is_leader else color,
                    linewidth=2.4 if is_leader else 1.4,
                    alpha=0.38 if is_leader else 0.28,
                )
                self.ax.add_patch(patch)
                self._ped_patches[ped_id] = patch

                (trail,) = self.ax.plot(
                    [], [], linestyle="--", linewidth=1.0, color=color, alpha=0.9
                )
                self._ped_trails[ped_id] = trail
                self._ped_history[ped_id] = deque(maxlen=self.trails.length)

                if ped.goal is not None:
                    (self._ped_goal_artists[ped_id],) = self.ax.plot(
                        [ped.goal.gx],
                        [ped.goal.gy],
                        marker="x",
                        markersize=8,
                        linestyle="None",
                        color=color,
                        markeredgewidth=1.8,
                    )

                if self.show_ids and ped_id not in self._ped_id_artists:
                    self._ped_id_artists[ped_id] = self.ax.text(
                        ped.pose.px,
                        ped.pose.py,
                        f"{ped_id}{'L' if is_leader else ''}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                        weight="bold" if is_leader else "normal",
                    )

            patch = self._ped_patches[ped_id]
            patch.center = (ped.pose.px, ped.pose.py)
            patch.radius = ped.radius
            patch.set_facecolor(color)
            patch.set_edgecolor(self.LEADER_EDGE_COLOR if is_leader else color)
            patch.set_linewidth(2.4 if is_leader else 1.4)
            patch.set_alpha(0.38 if is_leader else 0.28)

            self._ped_history[ped_id].append((float(ped.pose.px), float(ped.pose.py)))
            hx, hy = (
                zip(*self._ped_history[ped_id])
                if self._ped_history[ped_id]
                else ([], [])
            )
            self._ped_trails[ped_id].set_data(hx, hy)
            self._ped_trails[ped_id].set_color(color)

            if ped.goal is not None:
                if ped_id not in self._ped_goal_artists:
                    (self._ped_goal_artists[ped_id],) = self.ax.plot(
                        [ped.goal.gx],
                        [ped.goal.gy],
                        marker="x",
                        markersize=8,
                        linestyle="None",
                        color=color,
                        markeredgewidth=1.8,
                    )
                else:
                    self._ped_goal_artists[ped_id].set_data(
                        [ped.goal.gx], [ped.goal.gy]
                    )
                    self._ped_goal_artists[ped_id].set_color(color)

            if self.show_ids:
                if ped_id not in self._ped_id_artists:
                    self._ped_id_artists[ped_id] = self.ax.text(
                        ped.pose.px,
                        ped.pose.py,
                        f"{ped_id}{'L' if is_leader else ''}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                        weight="bold" if is_leader else "normal",
                    )
                else:
                    self._ped_id_artists[ped_id].set_position(
                        (ped.pose.px, ped.pose.py)
                    )

    def _update_status(self) -> None:
        if not self.show_status:
            self._status_text.set_text("")
            return

        robot = self.env.robot
        reached = self._agent_at_goal(robot.pose, robot.goal, threshold=0.1)
        crowd_done = all(
            self._agent_at_goal(ped.pose, ped.goal, threshold=0.1)
            for ped in getattr(self.env, "crowd", {}).values()
        )
        self._status_text.set_text(
            f"step: {self._step_count}\n"
            f"robot goal: {'yes' if reached else 'no'}\n"
            f"crowd goal: {'yes' if crowd_done else 'no'}"
        )

    def _frame(self, _frame_index: int):
        if self._closed:
            return []

        if self.stepper is not None and not self._paused:
            self.stepper.step()
            self._step_count += 1

        self._update_robot()
        self._update_crowd()
        self._update_status()
        return []

    def show(self, *, block: bool = True) -> FuncAnimation:
        """Render the animation and start stepping if a stepper is set."""
        self._anim = FuncAnimation(
            self.fig,
            self._frame,
            interval=self.interval,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=block)
        self._closed = True
        if self._anim is not None and self._anim.event_source is not None:
            self._anim.event_source.stop()
        return self._anim

    def save(self, path: str | Path, *, fps: int = 12, dpi: int = 150) -> None:
        """Save the animation to a file."""
        if self._anim is None:
            self._anim = FuncAnimation(
                self.fig,
                self._frame,
                interval=self.interval,
                blit=False,
                cache_frame_data=False,
            )
        self._anim.save(str(path), fps=fps, dpi=dpi)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def reset_view(self) -> None:
        self._fit_bounds()
        self.fig.canvas.draw_idle()


def visualize_environment(
    env: Any, *, title: str = "CrowdSimNextGen Visualization"
) -> MatplotlibVisualizer:
    """Show a static snapshot of the environment."""
    viz = MatplotlibVisualizer(env=env, stepper=None, title=title)
    viz.show()
    return viz


def visualize_simulation(
    env: Any,
    stepper: Any,
    *,
    interval: int = 80,
    trail_length: int = 30,
    title: str = "CrowdSimNextGen Simulation",
) -> MatplotlibVisualizer:
    """Show a live simulation that advances one Step per frame."""
    viz = MatplotlibVisualizer(
        env=env,
        stepper=stepper,
        interval=interval,
        trail_length=trail_length,
        title=title,
    )
    viz.show()
    return viz

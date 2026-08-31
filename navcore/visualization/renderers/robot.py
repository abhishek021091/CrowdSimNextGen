from __future__ import annotations

import math
from typing import Any

from matplotlib.patches import Circle, FancyArrow

from ..config import LayerToggles
from ..state import SceneSnapshot
from .base import Renderer


class RobotRenderer(Renderer):
    """Draws the robot body plus its motion/goal/sensing annotations.

    Distinct from pedestrians by color, a thicker outline, and an "R"
    label so it stands out even in dense crowds.
    """

    def __init__(self, ax: Any, scheme: Any, sensing_radius: float = 5.0) -> None:
        super().__init__(ax, scheme)
        self.sensing_radius = sensing_radius
        self._body: Circle | None = None
        self._sensing_ring: Circle | None = None
        self._heading_arrow: FancyArrow | None = None
        self._velocity_arrow: FancyArrow | None = None
        self._pref_velocity_arrow: FancyArrow | None = None
        self._goal_line = None
        self._goal_marker = None
        self._label = None

    def clear(self) -> None:
        for artist in (
            self._body,
            self._sensing_ring,
            self._heading_arrow,
            self._velocity_arrow,
            self._pref_velocity_arrow,
            self._goal_line,
            self._goal_marker,
            self._label,
        ):
            if artist is not None:
                artist.remove()
        self._body = self._sensing_ring = None
        self._heading_arrow = self._velocity_arrow = self._pref_velocity_arrow = None
        self._goal_line = self._goal_marker = self._label = None

    def update(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        robot = scene.robot
        if robot is None:
            return

        if self._body is None:
            self._body = Circle(
                (robot.x, robot.y),
                robot.radius,
                facecolor=self.scheme.robot_color,
                edgecolor=self.scheme.robot_edge,
                linewidth=0.3,
                alpha=0.3,
                zorder=2,
            )
            self.ax.add_patch(self._body)
        else:
            self._body.center = (robot.x, robot.y)
            self._body.radius = robot.radius
            self._body.set_facecolor(self.scheme.robot_color)
            self._body.set_edgecolor(self.scheme.robot_edge)

        # if toggles.ids and self._label is None:
        #     self._label = self.ax.text(
        #         robot.x, robot.y, "R", ha="center", va="center",
        #         fontsize=9, color="white", weight="bold", zorder=7,
        #     )
        if self._label is not None:
            self._label.set_visible(toggles.ids)
            self._label.set_position((robot.x, robot.y))

        self._update_sensing_ring(robot, toggles)
        self._update_heading(robot, toggles)
        self._update_velocity(robot, toggles)
        self._update_pref_velocity(robot, toggles)
        self._update_goal(robot, toggles)

    def _update_sensing_ring(self, robot, toggles: LayerToggles) -> None:
        show = toggles.sensor
        if not show:
            if self._sensing_ring is not None:
                self._sensing_ring.set_visible(False)
            return
        if self._sensing_ring is None:
            self._sensing_ring = Circle(
                (robot.x, robot.y),
                self.sensing_radius,
                fill=False,
                edgecolor=self.scheme.sensor_range_color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.5,
                zorder=2,
            )
            self.ax.add_patch(self._sensing_ring)
        self._sensing_ring.set_visible(True)
        self._sensing_ring.center = (robot.x, robot.y)

    def _arrow(
        self, existing: FancyArrow | None, x, y, vx, vy, color, scale, width, zorder
    ):
        length = math.hypot(vx, vy)
        if length < 1e-6:
            if existing is not None:
                existing.set_visible(False)
            return existing
        dx, dy = vx * scale, vy * scale
        if existing is not None:
            existing.remove()
        arrow = FancyArrow(
            x,
            y,
            dx,
            dy,
            width=width,
            head_width=width * 4,
            head_length=width * 5,
            length_includes_head=True,
            color=color,
            alpha=0.9,
            zorder=zorder,
        )
        self.ax.add_patch(arrow)
        return arrow

    def _update_heading(self, robot, toggles: LayerToggles) -> None:
        if not toggles.heading:
            if self._heading_arrow is not None:
                self._heading_arrow.set_visible(False)
            return
        hx, hy = math.cos(robot.heading), math.sin(robot.heading)
        self._heading_arrow = self._arrow(
            self._heading_arrow,
            robot.x,
            robot.y,
            hx,
            hy,
            self.scheme.robot_edge,
            robot.radius,
            0.03,
            8,
        )

    def _update_velocity(self, robot, toggles: LayerToggles) -> None:
        if not toggles.velocity:
            if self._velocity_arrow is not None:
                self._velocity_arrow.set_visible(False)
            return
        self._velocity_arrow = self._arrow(
            self._velocity_arrow,
            robot.x,
            robot.y,
            robot.vx,
            robot.vy,
            "#2ec4b6",
            0.6,
            0.035,
            7,
        )

    def _update_pref_velocity(self, robot, toggles: LayerToggles) -> None:
        if not toggles.preferred_velocity or robot.pref_vx is None:
            if self._pref_velocity_arrow is not None:
                self._pref_velocity_arrow.set_visible(False)
            return
        self._pref_velocity_arrow = self._arrow(
            self._pref_velocity_arrow,
            robot.x,
            robot.y,
            robot.pref_vx,
            robot.pref_vy or 0.0,
            self.scheme.orca_color,
            0.6,
            0.03,
            7,
        )

    def _update_goal(self, robot, toggles: LayerToggles) -> None:
        if not toggles.goals or robot.goal_x is None:
            if self._goal_line is not None:
                self._goal_line.set_visible(False)
            if self._goal_marker is not None:
                self._goal_marker.set_visible(False)
            return
        if self._goal_line is None:
            (self._goal_line,) = self.ax.plot(
                [],
                [],
                linestyle=":",
                linewidth=1.0,
                color=self.scheme.robot_color,
                alpha=0.5,
                zorder=1,
            )
            (self._goal_marker,) = self.ax.plot(
                [],
                [],
                marker="x",
                markersize=11,
                linestyle="None",
                color=self.scheme.robot_color,
                markeredgewidth=2.2,
                zorder=5,
            )
        self._goal_line.set_visible(True)
        self._goal_marker.set_visible(True)
        self._goal_line.set_data([robot.x, robot.goal_x], [robot.y, robot.goal_y])
        self._goal_marker.set_data([robot.goal_x], [robot.goal_y])

    def bounds(self, scene: SceneSnapshot):
        xs, ys = [], []
        robot = scene.robot
        if robot is not None:
            xs.append(robot.x)
            ys.append(robot.y)
            if robot.goal_x is not None:
                xs.append(robot.goal_x)
                ys.append(robot.goal_y)
        return xs, ys

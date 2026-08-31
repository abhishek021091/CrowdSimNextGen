"""Composition root for the main 2D scene axes.

``SceneRenderer`` doesn't draw agents/obstacles itself -- it owns the
axes chrome (grid, legend, title) and fans out ``update`` calls to the
single-responsibility renderers, then aggregates their bounds for the
camera to fit.
"""

from __future__ import annotations

from typing import Any

from .config import ColorScheme, LayerToggles
from .renderers import CrowdRenderer, ObstacleRenderer, OverlayRenderer, RobotRenderer, SensorRenderer
from .state import SceneSnapshot


class SceneRenderer:
    def __init__(
        self,
        ax: Any,
        scheme: ColorScheme,
        robot: RobotRenderer,
        crowd: CrowdRenderer,
        obstacles: ObstacleRenderer,
        sensor: SensorRenderer,
        overlay: OverlayRenderer,
        title: str,
    ) -> None:
        self.ax = ax
        self.scheme = scheme
        self.robot = robot
        self.crowd = crowd
        self.obstacles = obstacles
        self.sensor = sensor
        self.overlay = overlay
        self.title = title
        self._legend = None
        self._apply_chrome()

    def _apply_chrome(self) -> None:
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(self.title, color=self.scheme.text_color, fontsize=12, weight="bold")
        self.ax.set_facecolor(self.scheme.axes_face)

    def set_scheme(self, scheme: ColorScheme) -> None:
        self.scheme = scheme
        self._apply_chrome()
        for renderer in (self.robot, self.crowd, self.obstacles, self.sensor, self.overlay):
            renderer.set_scheme(scheme)
        self._draw_legend()

    def build_static(self, obstacles: dict[Any, Any]) -> None:
        self.obstacles.build(obstacles)
        self._draw_legend()

    def _draw_legend(self) -> None:
        # Proxy artists only -- cheap, and rebuilt just on scheme/group
        # changes rather than every frame.
        for artist in list(self.ax.lines):
            if getattr(artist, "_is_legend_proxy", False):
                artist.remove()
        proxies = []
        (p,) = self.ax.plot([], [], "o", color=self.scheme.robot_color, label="Robot")
        p._is_legend_proxy = True
        proxies.append(p)
        (p,) = self.ax.plot([], [], "o", color=self.scheme.ungrouped_ped_color, label="Pedestrian")
        p._is_legend_proxy = True
        proxies.append(p)
        (p,) = self.ax.plot([], [], "x", color=self.scheme.text_color, label="Goal")
        p._is_legend_proxy = True
        proxies.append(p)
        legend = self.ax.legend(
            handles=proxies, loc="upper right", fontsize=8, framealpha=0.9,
            facecolor=self.scheme.panel_face, edgecolor=self.scheme.panel_edge,
            labelcolor=self.scheme.text_color,
        )
        self._legend = legend

    def update(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        self.ax.grid(toggles.grid, alpha=self.scheme.grid_alpha, color=self.scheme.grid_color)
        self.robot.update(scene, toggles)
        self.crowd.update(scene, toggles)
        self.obstacles.update(scene, toggles)
        self.sensor.update(scene, toggles)
        self.overlay.update(scene, toggles)

    def collect_bounds(self, scene: SceneSnapshot) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for renderer in (self.robot, self.crowd, self.obstacles):
            rx, ry = renderer.bounds(scene)
            xs.extend(rx)
            ys.extend(ry)
        return xs, ys

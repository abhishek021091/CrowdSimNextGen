from __future__ import annotations

import math
from typing import Any

from matplotlib.patches import FancyArrow, Wedge

from ..config import LayerToggles
from ..state import SceneSnapshot
from .base import Renderer


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OverlayRenderer(Renderer):
    """Planning + reactive-avoidance debug overlay: planned path/waypoints,
    ORCA preferred/chosen velocity, velocity-obstacle cones, and predicted
    future trajectories for the crowd. All optional and duck-typed against
    ``scene.extra``, same pattern as ``SensorRenderer``.
    """

    def __init__(self, ax: Any, scheme: Any) -> None:
        super().__init__(ax, scheme)
        self._path_line = None
        self._waypoint_markers = None
        self._target_marker = None
        self._orca_pref_arrow: FancyArrow | None = None
        self._orca_chosen_arrow: FancyArrow | None = None
        self._cones: list[Wedge] = []
        self._prediction_lines: list[Any] = []

    def clear(self) -> None:
        for artist in (self._path_line, self._waypoint_markers, self._target_marker,
                       self._orca_pref_arrow, self._orca_chosen_arrow):
            if artist is not None:
                artist.remove()
        for c in self._cones:
            c.remove()
        for p in self._prediction_lines:
            p.remove()
        self._cones.clear()
        self._prediction_lines.clear()
        self._path_line = self._waypoint_markers = self._target_marker = None
        self._orca_pref_arrow = self._orca_chosen_arrow = None

    def update(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        self._update_path(scene, toggles)
        self._update_orca(scene, toggles)
        self._update_predictions(scene, toggles)

    def _update_path(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        path = scene.extra.get("path")
        if not toggles.planning or not path:
            if self._path_line is not None:
                self._path_line.set_visible(False)
            if self._waypoint_markers is not None:
                self._waypoint_markers.set_visible(False)
            if self._target_marker is not None:
                self._target_marker.set_visible(False)
            return

        xs = [_get(p, "x", p[0] if isinstance(p, (tuple, list)) else None) for p in path]
        ys = [_get(p, "y", p[1] if isinstance(p, (tuple, list)) else None) for p in path]
        if self._path_line is None:
            (self._path_line,) = self.ax.plot(
                [], [], linestyle="-", linewidth=1.6, color=self.scheme.path_color,
                alpha=0.7, zorder=3,
            )
            (self._waypoint_markers,) = self.ax.plot(
                [], [], marker="o", markersize=4, linestyle="None",
                color=self.scheme.path_color, alpha=0.7, zorder=3,
            )
            (self._target_marker,) = self.ax.plot(
                [], [], marker="*", markersize=12, linestyle="None",
                color=self.scheme.path_color, zorder=4,
            )
        self._path_line.set_visible(True)
        self._waypoint_markers.set_visible(True)
        self._path_line.set_data(xs, ys)
        self._waypoint_markers.set_data(xs, ys)
        if xs:
            self._target_marker.set_visible(True)
            self._target_marker.set_data([xs[0]], [ys[0]])
        else:
            self._target_marker.set_visible(False)

    def _arrow(self, existing, x, y, vx, vy, color, scale, width, zorder):
        length = math.hypot(vx, vy)
        if existing is not None:
            existing.remove()
        if length < 1e-6:
            return None
        arrow = FancyArrow(
            x, y, vx * scale, vy * scale, width=width, head_width=width * 4,
            head_length=width * 5, length_includes_head=True, color=color,
            alpha=0.85, zorder=zorder,
        )
        self.ax.add_patch(arrow)
        return arrow

    def _update_orca(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        orca = scene.extra.get("orca")
        for cone in self._cones:
            cone.remove()
        self._cones.clear()

        if not toggles.orca or orca is None or scene.robot is None:
            if self._orca_pref_arrow is not None:
                self._orca_pref_arrow.remove()
                self._orca_pref_arrow = None
            if self._orca_chosen_arrow is not None:
                self._orca_chosen_arrow.remove()
                self._orca_chosen_arrow = None
            return

        rx, ry = scene.robot.x, scene.robot.y
        pref = _get(orca, "preferred_velocity")
        chosen = _get(orca, "chosen_velocity")
        if pref is not None:
            self._orca_pref_arrow = self._arrow(
                self._orca_pref_arrow, rx, ry, _get(pref, "vx", 0.0), _get(pref, "vy", 0.0),
                self.scheme.orca_color, 0.6, 0.025, 8,
            )
        if chosen is not None:
            self._orca_chosen_arrow = self._arrow(
                self._orca_chosen_arrow, rx, ry, _get(chosen, "vx", 0.0), _get(chosen, "vy", 0.0),
                "#06d6a0", 0.6, 0.025, 8,
            )

        if toggles.collision_cones:
            cones = _get(orca, "collision_cones", []) or []
            for cone in cones:
                apex_x = _get(cone, "apex_x", rx)
                apex_y = _get(cone, "apex_y", ry)
                angle = math.degrees(_get(cone, "angle", 0.0))
                half_width = math.degrees(_get(cone, "half_width", 0.15))
                radius = _get(cone, "radius", 3.0)
                wedge = Wedge(
                    (apex_x, apex_y), radius, angle - half_width, angle + half_width,
                    facecolor="#ef476f", edgecolor="none", alpha=0.15, zorder=2,
                )
                self.ax.add_patch(wedge)
                self._cones.append(wedge)

    def _update_predictions(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        for line in self._prediction_lines:
            line.remove()
        self._prediction_lines.clear()
        predictions = scene.extra.get("predictions")
        if not toggles.prediction or not predictions:
            return
        items = predictions.items() if isinstance(predictions, dict) else enumerate(predictions)
        for _agent_id, traj in items:
            xs = [_get(p, "x", p[0] if isinstance(p, (tuple, list)) else None) for p in traj]
            ys = [_get(p, "y", p[1] if isinstance(p, (tuple, list)) else None) for p in traj]
            if not xs:
                continue
            (line,) = self.ax.plot(
                xs, ys, linestyle=(0, (2, 2)), linewidth=1.0,
                color=self.scheme.prediction_color, alpha=0.5, zorder=3,
            )
            self._prediction_lines.append(line)

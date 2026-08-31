from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from ..config import LayerToggles
from ..state import SceneSnapshot
from .base import Renderer


class SensorRenderer(Renderer):
    """Renders robot perception: LiDAR rays, a visibility polygon, and
    range. Entirely optional -- if ``scene.extra["sensor"]`` is absent
    (the env doesn't expose sensing), this layer draws nothing and costs
    nothing beyond one dict lookup per frame.

    Expected (duck-typed) payload in ``scene.extra["sensor"]``, any subset:
        rays: list[(x0, y0, x1, y1, hit: bool)]
        visibility_polygon: list[(x, y)]
        detected_pedestrian_ids: set[int]
        detected_obstacle_ids: set[int]
    """

    def __init__(self, ax: Any, scheme: Any) -> None:
        super().__init__(ax, scheme)
        self._rays: LineCollection | None = None
        self._visibility_poly: Polygon | None = None

    def clear(self) -> None:
        if self._rays is not None:
            self._rays.remove()
            self._rays = None
        if self._visibility_poly is not None:
            self._visibility_poly.remove()
            self._visibility_poly = None

    def update(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        sensor = scene.extra.get("sensor")
        if not toggles.sensor or sensor is None:
            if self._rays is not None:
                self._rays.set_visible(False)
            if self._visibility_poly is not None:
                self._visibility_poly.set_visible(False)
            return

        rays = _get(sensor, "rays")
        if rays:
            segments = [[(r[0], r[1]), (r[2], r[3])] for r in rays]
            colors = [
                self.scheme.sensor_hit_color if (len(r) > 4 and r[4]) else self.scheme.sensor_ray_color
                for r in rays
            ]
            if self._rays is None:
                self._rays = LineCollection([], linewidths=0.7, alpha=0.55, zorder=2)
                self.ax.add_collection(self._rays)
            self._rays.set_visible(True)
            self._rays.set_segments(segments)
            self._rays.set_color(colors)
        elif self._rays is not None:
            self._rays.set_visible(False)

        poly = _get(sensor, "visibility_polygon")
        if poly and len(poly) >= 3:
            if self._visibility_poly is None:
                self._visibility_poly = Polygon(
                    poly, closed=True, fill=True,
                    facecolor=self.scheme.sensor_range_color,
                    edgecolor="none", alpha=0.08, zorder=0,
                )
                self.ax.add_patch(self._visibility_poly)
            else:
                self._visibility_poly.set_xy(np.asarray(poly))
            self._visibility_poly.set_visible(True)
        elif self._visibility_poly is not None:
            self._visibility_poly.set_visible(False)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

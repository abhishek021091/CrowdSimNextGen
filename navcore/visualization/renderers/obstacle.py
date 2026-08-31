from __future__ import annotations

from typing import Any

from matplotlib.patches import Polygon

from ..config import LayerToggles
from ..state import SceneSnapshot
from .base import Renderer


class ObstacleRenderer(Renderer):
    """Draws static obstacles as filled polygons with optional labels.

    Obstacles rarely move, so vertices are cached from ``env`` once and
    only re-read if the visualizer explicitly calls ``reset``.
    """

    def __init__(self, ax: Any, scheme: Any, obstacle_to_vertices) -> None:
        super().__init__(ax, scheme)
        self._obstacle_to_vertices = obstacle_to_vertices
        self._patches: list[Polygon] = []
        self._labels: list[Any] = []
        self._vertex_cache: list[list[tuple[float, float]]] = []
        self._built = False

    def clear(self) -> None:
        for patch in self._patches:
            patch.remove()
        for label in self._labels:
            label.remove()
        self._patches.clear()
        self._labels.clear()
        self._vertex_cache.clear()
        self._built = False

    def build(self, obstacles: dict[Any, Any]) -> None:
        self.clear()
        for idx, (obstacle_id, obstacle) in enumerate(obstacles.items()):
            vertices = self._obstacle_to_vertices(obstacle)
            self._vertex_cache.append(list(vertices))
            patch = Polygon(
                vertices, closed=True, fill=True,
                facecolor=self.scheme.obstacle_face,
                edgecolor=self.scheme.obstacle_edge,
                linewidth=1.2, alpha=0.75, zorder=1,
            )
            self.ax.add_patch(patch)
            self._patches.append(patch)
            cx = sum(v[0] for v in vertices) / len(vertices)
            cy = sum(v[1] for v in vertices) / len(vertices)
            label = self.ax.text(
                cx, cy, str(obstacle_id), ha="center", va="center",
                fontsize=7, color=self.scheme.obstacle_edge, alpha=0.8, zorder=2,
            )
            label.set_visible(False)
            self._labels.append(label)
        self._built = True

    def update(self, scene: SceneSnapshot, toggles: LayerToggles) -> None:
        for label in self._labels:
            label.set_visible(toggles.labels)
        for patch in self._patches:
            patch.set_facecolor(self.scheme.obstacle_face)
            patch.set_edgecolor(self.scheme.obstacle_edge)

    def bounds(self, scene: SceneSnapshot):
        xs, ys = [], []
        for vertices in self._vertex_cache:
            for x, y in vertices:
                xs.append(float(x))
                ys.append(float(y))
        return xs, ys

"""VispyRenderer: a fast, dependency-light renderer for interactive viewing.

Draws whatever ``RenderFrame`` it's given -- agents as circles, static
obstacles as outlines -- using VisPy's GPU-accelerated canvas. This is
the default renderer for testing/qualitative-review runs; training
runs simply never construct one (see ``rendering.renderer.Renderer``
for why that's zero-cost).

This renderer opens its own VisPy canvas rather than assuming it is
hosted inside the existing PySide6 ``MainWindow`` -- embedding it as
the main window's central widget is a natural follow-up once this is
proven standalone, not done here.

Note:
    Unlike every other file delivered so far, this one has not been
    executed -- the sandbox used to build/test the rest of this batch
    has no ``vispy`` install and no display/GPU backend. It's written
    directly against the ``Renderer`` protocol and VisPy's documented
    ``scene`` API, but please run it locally (``pip install vispy``)
    before relying on it; treat it as reviewed-but-unverified, not
    tested like the other files.
"""

from __future__ import annotations

import numpy as np
from vispy import app, scene

from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.polygon import Polygon
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.rendering.render_frame import RenderFrame

_AGENT_COLORS: dict[str, str] = {
    "Robot": "#e63946",
    "Pedestrian": "#457b9d",
}
_DEFAULT_AGENT_COLOR = "#6c757d"
_OBSTACLE_COLOR = "#2b2d42"
_CIRCLE_SEGMENTS = 32


class VispyRenderer:
    """Draws agents and obstacles for one episode using a VisPy canvas."""

    def __init__(self, title: str = "navcore") -> None:
        self._canvas = scene.SceneCanvas(title=title, keys="interactive", show=True)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.PanZoomCamera(aspect=1.0)
        self._view.camera.set_range()

        self._agents = scene.visuals.Markers(parent=self._view.scene)
        self._obstacle_lines: list[scene.visuals.Line] = []
        self._obstacles_drawn = False

    def render(self, frame: RenderFrame) -> None:
        if not self._obstacles_drawn:
            self._draw_obstacles(frame.obstacles)
            self._obstacles_drawn = True

        n = len(frame.agent_kinds)
        colors = [
            _AGENT_COLORS.get(kind, _DEFAULT_AGENT_COLOR) for kind in frame.agent_kinds
        ]
        sizes = frame.agent_radii * 2.0 * self._pixels_per_unit()

        self._agents.set_data(
            pos=frame.agent_positions,
            face_color=np.array(colors) if n else None,
            size=sizes if n else np.array([]),
            edge_width=0,
        )
        self._canvas.update()
        app.process_events()

    def reset(self, episode: int) -> None:
        """Clear per-episode drawing state; obstacles are redrawn next render()."""
        for line in self._obstacle_lines:
            line.parent = None
        self._obstacle_lines.clear()
        self._obstacles_drawn = False

    def close(self) -> None:
        self._canvas.close()

    def _pixels_per_unit(self) -> float:
        """Rough world-to-pixel scale, used only to size agent markers.

        VisPy's ``PanZoomCamera`` makes an exact conversion awkward to
        query directly; this is a cosmetic sizing heuristic that no
        simulation logic depends on -- getting it slightly wrong only
        makes markers a bit too big or small on screen.
        """
        rect = self._view.camera.rect
        if rect is None or rect.width == 0:
            return 20.0
        return self._canvas.size[0] / rect.width

    def _draw_obstacles(self, obstacles: tuple[Obstacle, ...]) -> None:
        for obstacle in obstacles:
            points = _outline(obstacle)
            if points is None:
                continue
            line = scene.visuals.Line(
                pos=points, color=_OBSTACLE_COLOR, width=2, parent=self._view.scene
            )
            self._obstacle_lines.append(line)


def _outline(obstacle: Obstacle) -> np.ndarray | None:
    """Return a closed polyline approximating ``obstacle``'s geometry.

    Returns ``None`` if the geometry kind isn't one this renderer knows
    how to draw yet -- callers should skip it rather than fail.
    """
    geometry = obstacle.geometry
    if isinstance(geometry, Polygon):
        verts = [(v.x, v.y) for v in geometry.vertices]
        verts.append(verts[0])
        return np.array(verts)
    if isinstance(geometry, Rectangle):
        verts = [(v.x, v.y) for v in geometry.vertices()]
        verts.append(verts[0])
        return np.array(verts)
    if isinstance(geometry, Circle):
        # endpoint=False + manual close avoids a near-but-not-exact seam from
        # floating-point rounding of cos(2*pi) vs. cos(0.0).
        angles = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_SEGMENTS, endpoint=False)
        cx, cy = geometry.center.x, geometry.center.y
        points = np.column_stack(
            (
                cx + geometry.radius * np.cos(angles),
                cy + geometry.radius * np.sin(angles),
            )
        )
        return np.vstack([points, points[0]])
    return None

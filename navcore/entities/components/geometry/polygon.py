from collections.abc import Iterator
from dataclasses import dataclass, field

from navcore.entities.components.geometry.geometry import Geometry
from navcore.entities.components.geometry.line import Line
from navcore.entities.components.geometry.vector2 import Vector2


@dataclass(slots=True, frozen=True)
class Polygon(Geometry):
    """A simple (non-self-intersecting) polygon defined by local vertices.

    Attributes:
        vertices: The polygon's vertices, in order (clockwise or
            counter-clockwise), expressed in the local frame. At least
            three vertices are required.
    """

    vertices: tuple[Vector2, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize ``vertices`` to a tuple and validate the vertex count.

        Raises:
            ValueError: If fewer than three vertices are provided.
        """
        vertices = tuple(self.vertices)
        object.__setattr__(self, "vertices", vertices)
        if len(vertices) < 3:
            raise ValueError(
                f"Polygon requires at least 3 vertices, got {len(vertices)}."
            )

    def edges(self) -> Iterator[Line]:
        """Yield the polygon's edges, in vertex order, wrapping around.

        Yields:
            A :class:`Line` for each consecutive pair of vertices,
            including the closing edge from the last vertex to the first.
        """
        n = len(self.vertices)
        for i in range(n):
            yield Line(self.vertices[i], self.vertices[(i + 1) % n])

    def area(self) -> float:
        """Return the polygon's area via the shoelace formula."""
        return abs(self._signed_area())

    def centroid(self) -> Vector2:
        """Return the polygon's area-weighted centroid.

        Falls back to the arithmetic mean of the vertices for degenerate
        (zero-area) polygons.
        """
        signed_area = self._signed_area()
        if signed_area == 0.0:
            n = len(self.vertices)
            avg_x = sum(v.x for v in self.vertices) / n
            avg_y = sum(v.y for v in self.vertices) / n
            return Vector2(avg_x, avg_y)

        cx = 0.0
        cy = 0.0
        n = len(self.vertices)
        for i in range(n):
            x1, y1 = self.vertices[i].x, self.vertices[i].y
            x2, y2 = self.vertices[(i + 1) % n].x, self.vertices[(i + 1) % n].y
            cross = x1 * y2 - x2 * y1
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross

        factor = 1.0 / (6.0 * signed_area)
        return Vector2(cx * factor, cy * factor)

    def _signed_area(self) -> float:
        """Return the signed area (positive if vertices are CCW)."""
        n = len(self.vertices)
        total = 0.0
        for i in range(n):
            x1, y1 = self.vertices[i].x, self.vertices[i].y
            x2, y2 = self.vertices[(i + 1) % n].x, self.vertices[(i + 1) % n].y
            total += x1 * y2 - x2 * y1
        return total / 2.0

from dataclasses import dataclass

from navcore.entities.components.geometry.geometry import Geometry, Vector2


@dataclass(slots=True, frozen=True)
class Line(Geometry):
    """A straight line segment between two points.

    Attributes:
        start: The segment's start point, in local coordinates.
        end: The segment's end point, in local coordinates.
    """

    start: Vector2
    end: Vector2

    def length(self) -> float:
        """Return the length of the segment."""
        return self.start.distance_to(self.end)

    def direction(self) -> Vector2:
        """Return the unit vector pointing from ``start`` to ``end``.

        Raises:
            ValueError: If ``start`` and ``end`` coincide.
        """
        return (self.end - self.start).normalize()

    def midpoint(self) -> Vector2:
        """Return the point halfway between ``start`` and ``end``."""
        return (self.start + self.end) * 0.5

    def area(self) -> float:
        """A line segment encloses no area.

        Returns:
            Always ``0.0``.
        """
        return 0.0

    def centroid(self) -> Vector2:
        """Return the midpoint of the segment."""
        return self.midpoint()

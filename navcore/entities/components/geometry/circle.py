import math
from dataclasses import dataclass

from navcore.entities.components.geometry.geometry import Geometry, Vector2


@dataclass(slots=True, frozen=True)
class Circle(Geometry):
    """A circle centered on the local origin.

    Only the radius is stored; the center is implicitly ``(0, 0)`` in the
    local frame, since world placement belongs to ``Pose``.

    Attributes:
        radius: The circle's radius. Must be strictly positive.
    """

    radius: float

    def __post_init__(self) -> None:
        """Validate the radius.

        Raises:
            ValueError: If ``radius`` is not strictly positive.
        """
        if self.radius <= 0.0:
            raise ValueError(f"Circle radius must be positive, got {self.radius!r}.")

    def area(self) -> float:
        """Return the circle's area (``pi * r^2``)."""
        return math.pi * self.radius**2

    def circumference(self) -> float:
        """Return the circle's circumference (``2 * pi * r``)."""
        return 2.0 * math.pi * self.radius

    def centroid(self) -> Vector2:
        """Return the local center of the circle, ``(0, 0)``."""
        return Vector2.zero()

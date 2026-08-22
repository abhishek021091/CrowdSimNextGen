from dataclasses import dataclass

from navcore.entities.components.geometry.geometry import Geometry, Vector2


@dataclass(slots=True, frozen=True)
class Rectangle(Geometry):
    """An axis-aligned rectangle centered on the local origin.

    Only extents are stored; the rectangle is implicitly centered at
    ``(0, 0)`` in the local frame, since world placement and orientation
    belong to ``Pose``.

    Attributes:
        width: Full extent along the local x-axis. Must be strictly positive.
        height: Full extent along the local y-axis. Must be strictly positive.
    """

    width: float
    height: float

    def __post_init__(self) -> None:
        """Validate the dimensions.

        Raises:
            ValueError: If ``width`` or ``height`` is not strictly positive.
        """
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(
                "Rectangle width and height must be positive, "
                f"got width={self.width!r}, height={self.height!r}."
            )

    @property
    def half_width(self) -> float:
        """Return half of :attr:`width`."""
        return self.width / 2.0

    @property
    def half_height(self) -> float:
        """Return half of :attr:`height`."""
        return self.height / 2.0

    def area(self) -> float:
        """Return the rectangle's area (``width * height``)."""
        return self.width * self.height

    def centroid(self) -> Vector2:
        """Return the local center of the rectangle, ``(0, 0)``."""
        return Vector2.zero()

    def vertices(self) -> tuple[Vector2, Vector2, Vector2, Vector2]:
        """Return the four local corners in counter-clockwise order.

        Returns:
            Corners starting at bottom-left: bottom-left, bottom-right,
            top-right, top-left.
        """
        hw, hh = self.half_width, self.half_height
        return (
            Vector2(-hw, -hh),
            Vector2(hw, -hh),
            Vector2(hw, hh),
            Vector2(-hw, hh),
        )

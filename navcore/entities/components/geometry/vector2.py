from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Vector2:
    """An immutable 2D vector (or point).

    ``Vector2`` is used both as a free vector (direction + magnitude) and
    as a position, depending on context. It supports the standard set of
    vector-algebra operations via operator overloading.

    Attributes:
        x: The x component.
        y: The y component.
    """

    x: float
    y: float

    # -- construction helpers -------------------------------------------------

    @classmethod
    def zero(cls) -> Vector2:
        """Return the zero vector ``(0, 0)``."""
        return cls(0.0, 0.0)

    @classmethod
    def from_tuple(cls, values: tuple[float, float]) -> Vector2:
        """Construct a ``Vector2`` from an ``(x, y)`` tuple."""
        x, y = values
        return cls(x, y)

    # -- operator overloads -----------------------------------------------------

    def __add__(self, other: Vector2) -> Vector2:
        """Add two vectors component-wise."""
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2) -> Vector2:
        """Subtract another vector from this one, component-wise."""
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2:
        """Scale this vector by ``scalar`` (``vector * scalar``)."""
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector2:
        """Scale this vector by ``scalar`` (``scalar * vector``)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector2:
        """Divide this vector by ``scalar``.

        Raises:
            ZeroDivisionError: If ``scalar`` is zero.
        """
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide a Vector2 by zero.")
        return Vector2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vector2:
        """Return the negated vector."""
        return Vector2(-self.x, -self.y)

    def __iter__(self) -> Iterator[float]:
        """Iterate over ``(x, y)``, enabling ``tuple(v)`` and unpacking."""
        yield self.x
        yield self.y

    # -- vector algebra -----------------------------------------------------

    def dot(self, other: Vector2) -> float:
        """Return the dot product with ``other``."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vector2) -> float:
        """Return the scalar (z-component) of the 2D cross product."""
        return self.x * other.y - self.y * other.x

    def magnitude(self) -> float:
        """Return the Euclidean length of the vector."""
        return math.hypot(self.x, self.y)

    def magnitude_squared(self) -> float:
        """Return the squared Euclidean length.

        Prefer this over :meth:`magnitude` when only comparing lengths,
        since it avoids a square root.
        """
        return self.x * self.x + self.y * self.y

    def normalize(self) -> Vector2:
        """Return a unit-length vector pointing in the same direction.

        Raises:
            ValueError: If this is the zero vector.
        """
        mag = self.magnitude()
        if mag == 0.0:
            raise ValueError("Cannot normalize a zero-length vector.")
        return Vector2(self.x / mag, self.y / mag)

    def distance_to(self, other: Vector2) -> float:
        """Return the Euclidean distance between this point and ``other``."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def angle(self) -> float:
        """Return the angle of this vector from the positive x-axis.

        Returns:
            The angle in radians, in the range ``(-pi, pi]``.
        """
        return math.atan2(self.y, self.x)

    def rotate(self, angle_rad: float) -> Vector2:
        """Return this vector rotated counter-clockwise by ``angle_rad``.

        Args:
            angle_rad: Rotation angle in radians.
        """
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return Vector2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )

    def to_tuple(self) -> tuple[float, float]:
        """Return this vector as an ``(x, y)`` tuple."""
        return (self.x, self.y)

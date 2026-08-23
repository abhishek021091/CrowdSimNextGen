import math
from dataclasses import dataclass
from pathlib import Path

import tomllib

import navcore.configs
from navcore.entities.components.geometry.geometry import Geometry, Vector2


@dataclass(slots=True, frozen=True)
class Circle(Geometry):
    """A circle centered on the local origin.

    Only the radius is stored; the center is implicitly ``(0, 0)`` in the
    local frame, since world placement belongs to ``Pose``.

    Attributes:
        radius: The circle's radius. Must be strictly positive.
    """

    center: Vector2
    radius: float

    def __post_init__(self) -> None:
        """Validate the radius.

        Raises:
            ValueError: If ``radius`` is not strictly positive.
        """
        assert navcore.configs.__file__ is not None
        env_path = Path(navcore.configs.__file__).parent / "env.toml"
        with open(Path(env_path), "rb") as f:
            env = tomllib.load(f)

        if (
            env["arenaSize"]["width"] <= self.center[0]
            or -env["arenaSize"]["width"] >= self.center[0]
        ):
            raise ValueError(
                f"Circle center x-coordinate is out of bounds, got {self.center[0]}."
            )
        if (
            env["arenaSize"]["height"] <= self.center[1]
            or -env["arenaSize"]["height"] >= self.center[1]
        ):
            raise ValueError(
                f"Circle center y-coordinate is out of bounds, got {self.center[1]}."
            )
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
        return self.center

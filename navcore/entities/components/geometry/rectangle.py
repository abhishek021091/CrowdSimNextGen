from dataclasses import dataclass
from pathlib import Path

import tomllib

import navcore.configs
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

    center: Vector2
    width: float
    height: float

    assert navcore.configs.__file__ is not None
    env_path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(Path(env_path), "rb") as f:
        env_config = tomllib.load(f)

    def __post_init__(self) -> None:
        """Validate the dimensions.

        Raises:
            ValueError: If ``width`` or ``height`` is not strictly positive.
        """

        if (
            self.env_config["arenaSize"]["width"] <= self.center[0]
            or -self.env_config["arenaSize"]["width"] >= self.center[0]
        ):
            raise ValueError(
                f"Rectangle center x-coordinate is out of bounds, got {self.center[0]}."
            )
        if (
            self.env_config["arenaSize"]["height"] <= self.center[1]
            or -self.env_config["arenaSize"]["height"] >= self.center[1]
        ):
            raise ValueError(
                f"Rectangle center y-coordinate is out of bounds, got {self.center[1]}."
            )
        if (
            self.env_config["arenaSize"]["width"] <= self.width / 2.0
            or -self.env_config["arenaSize"]["width"] >= self.width / 2.0
        ):
            raise ValueError(f"Rectangle width is out of bounds, got {self.width!r}.")
        if (
            self.env_config["arenaSize"]["height"] <= self.height / 2.0
            or -self.env_config["arenaSize"]["height"] >= self.height / 2.0
        ):
            raise ValueError(f"Rectangle height is out of bounds, got {self.height!r}.")

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
        return self.center

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

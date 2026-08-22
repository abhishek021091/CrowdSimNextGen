"""Wall obstacle: a straight, thick wall segment.

``Wall`` composes an ``Obstacle`` rather than subclassing it (see
``obstacle.py`` for the rationale). It stores its own domain-specific
fields -- the wall's endpoints and thickness -- and exposes a
``to_obstacle()`` factory that builds the generic ``Obstacle`` consumed
by the rest of the system.

Geometry note:
    A wall's shape is represented as a ``LineSegment`` centerline, with
    ``thickness`` stored on ``Wall`` itself rather than baked into the
    geometry. This keeps the obstacle model decoupled from any one
    geometric encoding of "thickness" -- a future obstacle kind is free
    to represent a thick wall as an actual thin ``Rectangle`` instead,
    without any change to ``Obstacle`` or to how walls are consumed
    elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from navcore.entities.components.geometry import LineSegment, Vector2

from .obstacle import Obstacle


@dataclass(slots=True, frozen=True)
class Wall:
    """A straight wall between two points, with a configurable thickness.

    Attributes:
        id: Stable, caller-assigned unique identifier for this wall.
        start: The wall's starting endpoint, in world coordinates.
        end: The wall's ending endpoint, in world coordinates.
        thickness: The wall's thickness, in the same units as the
            geometry. Must be strictly positive.
        name: Optional human-readable label.
        traversable: Whether agents may pass through this wall. Defaults
            to ``False`` -- walls are solid by default.
        visible: Whether this wall is part of the visible environment.
    """

    id: str
    start: Vector2
    end: Vector2
    thickness: float
    name: str | None = None
    traversable: bool = False
    visible: bool = True

    def __post_init__(self) -> None:
        """Validate that ``thickness`` is physically meaningful.

        Raises:
            ValueError: If ``thickness`` is not strictly positive.
        """
        if self.thickness <= 0.0:
            raise ValueError(
                f"Wall thickness must be positive, got {self.thickness!r}."
            )

    def centerline(self) -> LineSegment:
        """Return the wall's centerline as a ``LineSegment``."""
        return LineSegment(self.start, self.end)

    def to_obstacle(self) -> Obstacle:
        """Build the generic ``Obstacle`` representation of this wall."""
        return Obstacle(
            id=self.id,
            geometry=self.centerline(),
            name=self.name,
            traversable=self.traversable,
            visible=self.visible,
        )

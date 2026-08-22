"""Pillar obstacle: a circular, static column.

``Pillar`` composes an ``Obstacle`` rather than subclassing it, mirroring
the pattern used by ``Wall`` and ``Boundary`` (see ``obstacle.py`` for
the rationale).
"""

from __future__ import annotations

from dataclasses import dataclass

from navcore.entities.components.geometry import Circle, Vector2

from .obstacle import Obstacle


@dataclass(slots=True, frozen=True)
class Pillar:
    """A circular pillar at a fixed location.

    Attributes:
        id: Stable, caller-assigned unique identifier for this pillar.
        center: The pillar's center, in world coordinates.
        radius: The pillar's radius. Must be strictly positive.
        name: Optional human-readable label.
        traversable: Whether agents may pass through this pillar. Defaults
            to ``False`` -- pillars are solid by default.
        visible: Whether this pillar is part of the visible environment.
    """

    id: str
    center: Vector2
    radius: float
    name: str | None = None
    traversable: bool = False
    visible: bool = True

    def __post_init__(self) -> None:
        """Validate that ``radius`` is physically meaningful.

        Raises:
            ValueError: If ``radius`` is not strictly positive.
        """
        if self.radius <= 0.0:
            raise ValueError(f"Pillar radius must be positive, got {self.radius!r}.")

    def to_obstacle(self) -> Obstacle:
        """Build the generic ``Obstacle`` representation of this pillar."""
        return Obstacle(
            id=self.id,
            geometry=Circle(self.center, self.radius),
            name=self.name,
            traversable=self.traversable,
            visible=self.visible,
        )

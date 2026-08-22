"""Static obstacle domain model for navcore.

Defines the ``Obstacle`` value object shared by every concrete obstacle
kind (walls, pillars, boundaries, and future kinds). ``Obstacle`` is
deliberately the *only* type in this package that any other module needs
to know about generically -- concrete obstacle kinds (``Wall``,
``Pillar``, ``Boundary``, and future kinds such as ``Furniture`` or
``Shelf``) each *compose* an ``Obstacle`` via a ``to_obstacle()`` factory
method rather than subclassing it. This keeps the hierarchy flat: adding
a new obstacle kind never requires touching this file.

This module knows nothing about rendering, collision detection, physics,
Gazebo/VisPy, or planners (ORCA/RVO2). It only describes *what* static
obstacles exist and *where*, geometrically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from navcore.entities.components.geometry import Geometry, Rectangle, Vector2


@dataclass(slots=True, frozen=True)
class Obstacle:
    """A single static obstacle occupying space in the environment.

    ``Obstacle`` is a plain data-holder: it pairs an identity with a
    shape (``Geometry``) and a small set of environment-agnostic flags.
    It carries no behavior beyond simple serialization and formatting --
    spatial queries such as bounding boxes and point-containment are
    left as explicit placeholders for a future module.

    Attributes:
        id: Stable, caller-assigned unique identifier for this obstacle.
        geometry: The obstacle's shape, in world/absolute coordinates.
        name: Optional human-readable label, useful for debugging,
            logging, and UIs.
        traversable: Whether agents may pass through this obstacle.
            Defaults to ``False``. This is a static classification flag
            only -- no collision or navigation logic reads or enforces
            it here.
        visible: Whether this obstacle should be treated as part of the
            visible environment. Defaults to ``True``. This is a
            classification flag only -- no rendering happens here.
    """

    id: str
    geometry: Geometry
    name: str | None = None
    traversable: bool = False
    visible: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize this obstacle to a plain, JSON-friendly ``dict``.

        Returns:
            A dictionary with the obstacle's scalar fields, plus a
            ``"geometry_type"`` field naming the concrete geometry class.
            The geometry object itself is not deep-serialized here --
            that is the geometry package's concern, not this layer's.
        """
        return {
            "id": self.id,
            "geometry_type": type(self.geometry).__name__,
            "name": self.name,
            "traversable": self.traversable,
            "visible": self.visible,
        }

    def bounding_box(self) -> Rectangle:
        """Return an axis-aligned bounding box enclosing this obstacle.

        Placeholder. Computing a bounding box is a spatial-indexing
        concern that belongs to a future module built *on top of* this
        one -- deliberately not implemented here.

        Raises:
            NotImplementedError: Always. This is a placeholder signature
                for a future spatial-query module.
        """
        raise NotImplementedError(
            "Obstacle.bounding_box() is a placeholder. Bounding-box "
            "computation belongs to a future spatial-indexing module."
        )

    def contains(self, point: Vector2) -> bool:
        """Return whether ``point`` lies within this obstacle's geometry.

        Placeholder. Point-containment is a collision-detection concern
        that belongs to a future module built *on top of* this one --
        deliberately not implemented here.

        Args:
            point: The point to test, in world/absolute coordinates.

        Raises:
            NotImplementedError: Always. This is a placeholder signature
                for a future collision-detection module.
        """
        raise NotImplementedError(
            "Obstacle.contains() is a placeholder. Point-containment "
            "belongs to a future collision-detection module."
        )

    def __repr__(self) -> str:
        """Return a concise, developer-friendly representation."""
        label = f" name={self.name!r}" if self.name is not None else ""
        return (
            f"{type(self).__name__}(id={self.id!r},"
            f" geometry={type(self.geometry).__name__},"
            f"{label} traversable={self.traversable}, visible={self.visible})"
        )

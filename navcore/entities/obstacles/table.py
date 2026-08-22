"""Table obstacle: a static piece of furniture with a configurable footprint.

``Table`` composes an ``Obstacle`` rather than subclassing it, mirroring
the pattern used by ``Wall``, ``Pillar``, and ``Boundary`` (see
``obstacle.py`` for the rationale).

Shape note:
    Unlike ``Wall`` (always a line) or ``Pillar`` (always a circle), a
    table's footprint genuinely varies by furniture type -- round
    tables, rectangular tables, and irregular/custom tables all show up
    in real environments. Rather than three near-duplicate classes
    (``CircularTable``, ``RectangularTable``, ``PolygonalTable``), this
    module keeps a single ``Table`` that holds whichever geometry fits
    (``Circle | Rectangle | Polygon``), plus three small named
    constructors -- :meth:`Table.circular`, :meth:`Table.rectangular`,
    :meth:`Table.polygonal` -- so callers don't need to import or
    construct geometry objects themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from navcore.entities.components.geometry import Circle, Polygon, Rectangle, Vector2

from .obstacle import Obstacle

#: The geometry kinds a table's footprint may take.
type TableShape = Circle | Rectangle | Polygon


@dataclass(slots=True, frozen=True)
class Table:
    """A static table (or similar furniture) with a caller-chosen footprint.

    Attributes:
        id: Stable, caller-assigned unique identifier for this table.
        shape: The table's footprint, in world coordinates -- a
            ``Circle`` for a round table, a ``Rectangle`` for a
            rectangular table, or a ``Polygon`` for an irregular one.
        name: Optional human-readable label.
        traversable: Whether agents may pass through this table.
            Defaults to ``False`` -- tables are solid by default.
        visible: Whether this table is part of the visible environment.
    """

    id: str
    shape: TableShape
    name: str | None = None
    traversable: bool = False
    visible: bool = True

    def __post_init__(self) -> None:
        """Validate that ``shape`` is one of the supported geometry kinds.

        Raises:
            TypeError: If ``shape`` is not a ``Circle``, ``Rectangle``,
                or ``Polygon``.
        """
        if not isinstance(self.shape, (Circle, Rectangle, Polygon)):
            raise TypeError(
                "Table shape must be Circle, Rectangle, or Polygon, "
                f"got {type(self.shape).__name__}."
            )

    @classmethod
    def circular(
        cls,
        id: str,
        center: Vector2,
        radius: float,
        *,
        name: str | None = None,
        traversable: bool = False,
        visible: bool = True,
    ) -> Table:
        """Build a round table without constructing a ``Circle`` directly.

        Args:
            id: Stable, caller-assigned unique identifier for this table.
            center: The table's center, in world coordinates.
            radius: The table's radius.
            name: Optional human-readable label.
            traversable: Whether agents may pass through this table.
            visible: Whether this table is part of the visible environment.
        """
        return cls(
            id=id,
            shape=Circle(center, radius),
            name=name,
            traversable=traversable,
            visible=visible,
        )

    @classmethod
    def rectangular(
        cls,
        id: str,
        center: Vector2,
        width: float,
        height: float,
        *,
        name: str | None = None,
        traversable: bool = False,
        visible: bool = True,
    ) -> Table:
        """Build a rectangular table without constructing a ``Rectangle`` directly.

        Args:
            id: Stable, caller-assigned unique identifier for this table.
            center: The table's center, in world coordinates.
            width: The table's width.
            height: The table's height.
            name: Optional human-readable label.
            traversable: Whether agents may pass through this table.
            visible: Whether this table is part of the visible environment.
        """
        return cls(
            id=id,
            shape=Rectangle(center, width, height),
            name=name,
            traversable=traversable,
            visible=visible,
        )

    @classmethod
    def polygonal(
        cls,
        id: str,
        vertices: tuple[Vector2, ...],
        *,
        name: str | None = None,
        traversable: bool = False,
        visible: bool = True,
    ) -> Table:
        """Build an irregularly-shaped table without constructing a ``Polygon`` directly.

        Args:
            id: Stable, caller-assigned unique identifier for this table.
            vertices: The table's outline, in order, in world coordinates.
            name: Optional human-readable label.
            traversable: Whether agents may pass through this table.
            visible: Whether this table is part of the visible environment.
        """
        return cls(
            id=id,
            shape=Polygon(vertices),
            name=name,
            traversable=traversable,
            visible=visible,
        )

    def to_obstacle(self) -> Obstacle:
        """Build the generic ``Obstacle`` representation of this table."""
        return Obstacle(
            id=self.id,
            geometry=self.shape,
            name=self.name,
            traversable=self.traversable,
            visible=self.visible,
        )

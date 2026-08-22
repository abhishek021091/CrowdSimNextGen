"""Boundary obstacle: the outer perimeter of an environment, with gates.

``Boundary`` composes an ``Obstacle`` rather than subclassing it (see
``obstacle.py`` for the rationale). It represents the outer walls of an
environment as a single closed ``Polygon``, optionally interrupted by
one or more ``BoundaryGate`` openings.

Design note:
    ``BoundaryGate`` is intentionally minimal -- it only records *where*
    an opening is (which edge, how far along it, how wide). It carries
    no semantics such as "entry", "exit", "door", or "spawn point";
    those are higher-level concepts that belong to simulation code built
    on top of this module, not to the boundary's geometric description
    of itself. Keeping the gate this bare means whatever meaning a gate
    eventually takes on can be layered on later without ever touching
    this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from navcore.entities.components.geometry import Polygon, Vector2

from .obstacle import Obstacle


@dataclass(slots=True, frozen=True)
class BoundaryGate:
    """A single opening in a boundary edge.

    A gate is purely geometric: it states that a given edge has an
    opening starting ``offset`` units along it and spanning ``width``
    units. It does not know whether that opening is an entrance, an
    exit, or anything else -- that meaning is assigned by higher-level
    code, not here.

    Attributes:
        edge_index: Index of the boundary edge the gate sits on, where
            edge ``i`` connects ``vertices[i]`` to ``vertices[i + 1]``
            (indices wrap around at the end of the vertex list).
        offset: Distance along the edge, measured from its start vertex,
            to the start of the opening. Must be non-negative.
        width: The width of the opening, measured along the edge. Must
            be strictly positive.
    """

    edge_index: int
    offset: float
    width: float

    def __post_init__(self) -> None:
        """Validate the gate's geometric parameters.

        Raises:
            ValueError: If ``edge_index`` is negative, ``offset`` is
                negative, or ``width`` is not strictly positive.
        """
        if self.edge_index < 0:
            raise ValueError(
                f"edge_index must be non-negative, got {self.edge_index!r}."
            )
        if self.offset < 0.0:
            raise ValueError(f"offset must be non-negative, got {self.offset!r}.")
        if self.width <= 0.0:
            raise ValueError(f"width must be positive, got {self.width!r}.")


@dataclass(slots=True, frozen=True)
class Boundary:
    """The outer perimeter of an environment, as a closed polygon.

    Attributes:
        id: Stable, caller-assigned unique identifier for this boundary.
        vertices: The boundary's vertices, in order, in world
            coordinates. At least three vertices are required.
        gates: Openings along the boundary's edges. Defaults to an empty
            tuple, i.e. a fully closed boundary.
        name: Optional human-readable label.
        traversable: Whether agents may pass through the boundary line
            itself, as opposed to through one of its gates. Defaults to
            ``False``.
        visible: Whether this boundary is part of the visible environment.
    """

    id: str
    vertices: tuple[Vector2, ...]
    gates: tuple[BoundaryGate, ...] = field(default_factory=tuple)
    name: str | None = None
    traversable: bool = False
    visible: bool = True

    def __post_init__(self) -> None:
        """Validate that the boundary and its gates are geometrically sound.

        Raises:
            ValueError: If fewer than three vertices are given, or if any
                gate references an edge index outside the valid range for
                this boundary's vertex count.
        """
        if len(self.vertices) < 3:
            raise ValueError(
                f"Boundary requires at least 3 vertices, got {len(self.vertices)}."
            )
        edge_count = len(self.vertices)
        for gate in self.gates:
            if gate.edge_index >= edge_count:
                raise ValueError(
                    f"BoundaryGate.edge_index={gate.edge_index} is out of "
                    f"range for a boundary with {edge_count} edges."
                )

    def polygon(self) -> Polygon:
        """Return the boundary's shape as a ``Polygon``."""
        return Polygon(self.vertices)

    def to_obstacle(self) -> Obstacle:
        """Build the generic ``Obstacle`` representation of this boundary.

        Note:
            Gates are boundary-specific metadata and are not encoded in
            the resulting ``Obstacle`` -- the generic obstacle model has
            no concept of an opening. Code that needs gate information
            should keep working with the ``Boundary`` instance directly
            rather than the flattened ``Obstacle``.
        """
        return Obstacle(
            id=self.id,
            geometry=self.polygon(),
            name=self.name,
            traversable=self.traversable,
            visible=self.visible,
        )

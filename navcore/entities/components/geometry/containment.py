"""Point-in-shape containment tests for the local-frame geometry types.

Design note -- why this is a free function, not ``Obstacle.contains()``:
    ``Obstacle.contains()`` (see ``navcore.entities.obstacles.obstacle``)
    is an existing, deliberately-documented placeholder: its docstring
    reserves point-containment for "a future collision-detection
    module." Rather than silently filling in someone else's marked
    placeholder, this module implements the underlying geometry math as
    a standalone utility. Callers that need "does this obstacle contain
    this point" (rasterization, and eventually the real
    collision-detection module) can use ``point_in_geometry`` directly;
    wiring ``Obstacle.contains()`` to delegate to it is a small,
    separate decision left for whoever builds that module.

Only ``Circle``, ``Rectangle``, and ``Polygon`` enclose area and can
meaningfully "contain" a point. ``Line`` has zero area (see
``Line.area()``) and is excluded on purpose.
"""

from __future__ import annotations

from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.geometry import Geometry
from navcore.entities.components.geometry.polygon import Polygon
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.components.geometry.vector2 import Vector2


def point_in_geometry(point: Vector2, geometry: Geometry) -> bool:
    """Return whether ``point`` lies within ``geometry``.

    Args:
        point: The point to test, in the same coordinate frame as
            ``geometry`` (world coordinates, for the obstacle geometries
            this module is intended for).
        geometry: The shape to test against. Must be a ``Circle``,
            ``Rectangle``, or ``Polygon``.

    Returns:
        ``True`` if ``point`` is inside or on the boundary of
        ``geometry``.

    Raises:
        TypeError: If ``geometry`` is a shape with no enclosed area
            (e.g. ``Line``), or any other unrecognized ``Geometry``
            subtype.
    """
    if isinstance(geometry, Circle):
        return _point_in_circle(point, geometry)
    if isinstance(geometry, Rectangle):
        return _point_in_rectangle(point, geometry)
    if isinstance(geometry, Polygon):
        return _point_in_polygon(point, geometry)
    raise TypeError(
        f"point_in_geometry() has no containment test for "
        f"{type(geometry).__name__}; it either encloses no area or is "
        f"not yet supported."
    )


def _point_in_circle(point: Vector2, circle: Circle) -> bool:
    return point.distance_to(circle.center) <= circle.radius


def _point_in_rectangle(point: Vector2, rectangle: Rectangle) -> bool:
    # Deliberately computed from ``center``/``half_width``/``half_height``
    # rather than ``Rectangle.vertices()``: ``vertices()`` returns
    # corners in a frame centered on the local origin, not offset by
    # ``rectangle.center`` -- using it here would silently ignore where
    # the rectangle actually sits.
    dx = abs(point.x - rectangle.center.x)
    dy = abs(point.y - rectangle.center.y)
    return dx <= rectangle.half_width and dy <= rectangle.half_height


def _point_in_polygon(point: Vector2, polygon: Polygon) -> bool:
    # Standard even-odd ray casting: cast a ray in +x from ``point`` and
    # count edge crossings. Odd count => inside. O(n) in vertex count,
    # which is fine here since this only runs once per grid cell at
    # ``CoverageGrid`` construction time, not per simulation tick.
    inside = False
    vertices = polygon.vertices
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i].x, vertices[i].y
        x2, y2 = vertices[(i + 1) % n].x, vertices[(i + 1) % n].y

        crosses = (y1 > point.y) != (y2 > point.y)
        if crosses:
            x_intersection = x1 + (point.y - y1) * (x2 - x1) / (y2 - y1)
            if point.x < x_intersection:
                inside = not inside
    return inside

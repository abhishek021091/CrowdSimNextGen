"""World-frame shapely conversion for Obstacle geometry.

navcore's own geometry types (Circle/Rectangle/Polygon) are pure
local-frame math -- see Geometry's own docstring: "world placement
belongs to Pose." In practice this project has no separate Pose layer
for obstacles: Obstacle.geometry is already world-frame for Polygon
(ObstacleBuilder.build_boundary, Table.polygonal), but Circle/Rectangle
carry `center` as a field on the geometry object itself and are
otherwise local-frame -- every consumer that needs world-frame obstacle
geometry has to translate by `.center` itself.

Three other call sites in this project already do that translation by
hand: `boustropheden._rectangle_to_world_vertices`/
`_circle_to_world_vertices` (correct), and
`base_orca_planner.obstacle_to_vertices` (which has a documented,
UNFIXED bug -- it returns Rectangle vertices in local frame, per
ObstacleCollisionDetector's own docstring caveat, and SAT inherits that
bug). This module is a fourth, correct implementation, factored out so
ray-casting doesn't either duplicate a fifth copy or silently inherit
obstacle_to_vertices's bug.

Not consolidated with those three call sites here. That is a real,
pre-existing duplication problem (see learnings-and-workflow), but
collapsing all four into one shared utility touches boustropheden.py,
base_orca_planner.py, and SAT, each already tested against current
behavior. Flagged as a follow-up, not attempted as a side effect of
wiring ray_features.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from shapely.geometry import LinearRing
from shapely.geometry import Polygon as ShapelyPolygon

from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.polygon import Polygon as NavPolygon
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.obstacles.obstacle import Obstacle

if TYPE_CHECKING:
    from navcore.entities.environment.environment import Environment

_CIRCLE_SEGMENTS = 16


def obstacle_to_shapely_polygon(obstacle: Obstacle) -> ShapelyPolygon:
    """Convert one Obstacle's geometry into a world-frame shapely Polygon.

    Raises:
        TypeError: If obstacle.geometry is not Polygon, Rectangle, or Circle.
    """
    geometry = obstacle.geometry

    if isinstance(geometry, NavPolygon):
        return ShapelyPolygon([(v.x, v.y) for v in geometry.vertices])

    if isinstance(geometry, Rectangle):
        hw, hh = geometry.half_width, geometry.half_height
        cx, cy = geometry.center.x, geometry.center.y
        return ShapelyPolygon(
            [
                (cx - hw, cy - hh),
                (cx + hw, cy - hh),
                (cx + hw, cy + hh),
                (cx - hw, cy + hh),
            ]
        )

    if isinstance(geometry, Circle):
        cx, cy, r = geometry.center.x, geometry.center.y, geometry.radius
        return ShapelyPolygon(
            [
                (
                    cx + r * math.cos(2.0 * math.pi * i / _CIRCLE_SEGMENTS),
                    cy + r * math.sin(2.0 * math.pi * i / _CIRCLE_SEGMENTS),
                )
                for i in range(_CIRCLE_SEGMENTS)
            ]
        )

    raise TypeError(
        f"obstacle_to_shapely_polygon() cannot convert geometry "
        f"{type(geometry).__name__}."
    )


def arena_boundary_ring(env: Environment) -> LinearRing:
    """Return the environment's outer boundary as a world-frame line.

    Uses `env.obstacles["boundary"]` if present (a walled `Boundary`),
    else falls back to the axis-aligned arena rectangle from
    `env.info.arena_width`/`arena_height` -- the same fallback
    `boustropheden._boundary_polygon_from_env` uses, since
    `ObstacleBuilder.build_boundary()` is not currently invoked by
    `EnvironmentBuilder`. A `LinearRing` (a line, not a filled shape)
    on purpose -- `ObstacleDetector.sense()`'s `boundary` argument
    must be a line geometry, since the robot sits inside it; passing a
    filled Polygon would make every ray "hit" it at distance 0.
    """
    boundary_obstacle = env.obstacles.get("boundary")
    if boundary_obstacle is not None and isinstance(
        boundary_obstacle.geometry, NavPolygon
    ):
        return LinearRing([(v.x, v.y) for v in boundary_obstacle.geometry.vertices])

    half_width = float(env.info.arena_width) / 2.0
    half_height = float(env.info.arena_height) / 2.0
    return LinearRing(
        [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]
    )

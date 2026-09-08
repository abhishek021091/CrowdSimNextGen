"""Convex decomposition of the environment's free space via interior extension
of edges, per Nielsen, Sung & Nielsen (2019), "Convex Decomposition for a
Coverage Path Planning for Autonomous Vehicles: Interior Extension of Edges,"
Sensors 19(19):4165.

BCD's job here is strictly *decomposition*: split free space into convex
cells plus their adjacency graph. It says nothing about sweep order,
sweep direction, or in-cell lane pattern -- callers hand the resulting
graph to a separate traversal step that drives ``SweepingMission`` per
cell with explicit start/end points. Project convention: "BCD
decomposes; it does not sweep." This is also why ``decompose()`` takes
no sweep axis: convexity, not sweep direction, is this module's only
concern -- direction is decided per cell by whatever consumes the graph.

The algorithm, in three phases:

1. Interior extension of edges (``_edge_extension_split``): at every
   reflex vertex of the free-space region, both edges meeting at that
   vertex are extended, as rays continuing each edge's own line, until
   they hit a wall (boundary or obstacle). Splitting along every such
   cut yields an initial set of sub-polygons that are all convex by
   construction.
2. Convex merge option search (``_find_mergeable_polygon_options``):
   adjacent initial sub-polygons are recombined wherever their union is
   still convex, producing a superset of candidate merged cells (`Ω` in
   the paper). Exhaustively checking every subset is combinatorial, so
   this implements the paper's Algorithm 1 approximation: an option of
   size `i` is only built from two already-verified options of size
   `floor(i/2)` and `ceil(i/2)`, not every possible split of `i`.
3. Merge option selection (``_select_merge_options``): the final cell
   set is chosen by solving the paper's integer program exactly --
   minimize total cell width subject to every initial sub-polygon being
   covered by exactly one selected option -- via ``scipy.optimize.milp``.

Correctness note -- why reflex vertices come from the free-space polygon's
rings, not from testing obstacles independently:
    A convex obstacle (e.g. any rectangular ``Table``) is, by definition,
    convex as its own shape -- none of its corners would ever test as
    reflex if checked against the obstacle's own polygon. But every one
    of those corners *is* a reflex point of the surrounding free space:
    a table in an open room creates concave notches in the space around
    it, exactly like a pillar does. Testing reflex-ness on the actual
    free-space polygon's exterior ring (the boundary) and interior rings
    (each obstacle becomes a hole) gets this right automatically, since
    ring orientation (CCW exterior, CW holes) already encodes which side
    is "inside" the free space at each vertex.

Why shapely and scipy, when the rest of the geometry package is
hand-rolled:
    Convex-hull area comparisons, robust polygon splitting/union, and
    integer-program set partitioning are each substantial, well-tested
    problems in their own right. This module is the one place in the
    project allowed to depend on them; its public surface (``decompose``,
    ``DecompositionResult``, ``DecomposedCell``, ``CellAdjacency``) speaks
    only navcore's own types.

Performance caveat:
    Phase 2's approximation is still combinatorial in the number of
    initial sub-polygons -- each level does a full pairwise product (or
    combinations, when merging same-size options) against the previous
    levels. For a handful of tables this is fine; for scenes producing
    hundreds of initial sub-polygons (many overlapping/adjacent
    obstacles), expect Phase 2 to dominate runtime. Phase 3's IP is
    always feasible (every singleton option is available as a
    fallback), but MILP solve time also grows with ``|Ω|``. Neither
    phase currently caches or reuses work across repeated calls; this
    module assumes decomposition happens once per episode, not per tick.

Known caveat:
    Edge-extension rays that graze a boundary/obstacle vertex tangentially
    (touching a corner without truly crossing) are treated as valid stopping
    points, same as any first-contact ray cast. This can occasionally produce
    a slightly shorter cut than a stricter "must actually cross" test would,
    for pathological vertex-aligned geometry.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from shapely.geometry import GeometryCollection, LineString, MultiPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import split, unary_union

from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.polygon import Polygon as NavPolygon
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.environment.environment import Environment
from navcore.entities.obstacles.obstacle import Obstacle

#: Segments used to tessellate a Circle obstacle into a polygon. Matches
#: obstacle_to_vertices's tessellation density for consistency.
_CIRCLE_SEGMENTS = 16

_EPS = 1e-9
#: Relative tolerance for the hull-area-vs-polygon-area convexity test.
#: Looser than _EPS on purpose -- polygons here have already passed
#: through one or more split/union operations, which accumulate enough
#: floating-point drift that a truly convex result can differ from its
#: own convex hull's area by more than 1e-9.
_CONVEXITY_REL_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class DecomposedCell:
    """One convex cell of free space produced by decomposition.

    Attributes:
        id: Identifier for this cell, unique within one
            ``DecompositionResult`` and contiguous from 0.
        vertices: The cell's outline, in order, in world coordinates.
        area: The cell's area, in world units squared.
    """

    id: int
    vertices: tuple[Vector2, ...]
    area: float

    def polygon(self) -> NavPolygon:
        """Return this cell's outline as a navcore ``Polygon``."""
        return NavPolygon(self.vertices)


@dataclass(frozen=True, slots=True)
class CellAdjacency:
    """One shared-edge relationship between two decomposed cells."""

    cell_a: int
    cell_b: int


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """The output of ``decompose()``: convex cells plus their adjacency graph."""

    cells: tuple[DecomposedCell, ...]
    adjacencies: tuple[CellAdjacency, ...]

    def neighbors_of(self, cell_id: int) -> tuple[int, ...]:
        """Return the ids of cells adjacent to ``cell_id``, sorted ascending."""
        neighbors: set[int] = set()
        for adj in self.adjacencies:
            if adj.cell_a == cell_id:
                neighbors.add(adj.cell_b)
            elif adj.cell_b == cell_id:
                neighbors.add(adj.cell_a)
        return tuple(sorted(neighbors))

    def cell_containing(self, point: Vector2) -> DecomposedCell | None:
        """Return the cell whose polygon contains ``point``, if any."""
        p = ShapelyPoint(point.x, point.y)
        for cell in self.cells:
            shapely_poly = ShapelyPolygon([(v.x, v.y) for v in cell.vertices])
            if shapely_poly.buffer(_EPS).contains(p):
                return cell
        return None


@dataclass(slots=True)
class _MergeOption:
    """One candidate merged cell: a set of initial sub-polygon indices whose
    union is convex, plus that union's geometry and width.

    Internal to this module -- callers only ever see the final selected
    cells via ``DecomposedCell``.
    """

    indices: frozenset[int]
    polygon: ShapelyPolygon
    width: float


# -- public entry points ----------------------------------------------------


def decompose(env: Environment) -> DecompositionResult:
    """Decompose ``env``'s free space into convex cells.

    Args:
        env: The live environment to decompose. Its obstacles are read,
            not mutated.

    Returns:
        The decomposed cells and their adjacency graph, in navcore's
        own geometry types.
    """
    boundary_polygon = _boundary_polygon_from_env(env)
    interior_obstacles = [
        obstacle for key, obstacle in env.obstacles.items() if key != "boundary"
    ]
    return decompose_free_space(boundary_polygon, interior_obstacles)


def decompose_free_space(
    boundary: NavPolygon, obstacles: Iterable[Obstacle]
) -> DecompositionResult:
    """Decompose ``boundary`` minus ``obstacles`` into convex cells.

    Split out from ``decompose()`` so the decomposition math is testable
    without constructing a full ``Environment``.
    """
    shapely_boundary = _navpolygon_to_shapely(boundary)
    shapely_obstacles = [_obstacle_to_shapely_polygon(obs) for obs in obstacles]
    free_space = _free_space(shapely_boundary, shapely_obstacles)
    if free_space.is_empty:
        return DecompositionResult(cells=(), adjacencies=())

    initial_subpolygons: list[ShapelyPolygon] = []
    for component in _iter_polygons(free_space):
        initial_subpolygons.extend(_edge_extension_split(component))

    if not initial_subpolygons:
        return DecompositionResult(cells=(), adjacencies=())

    adjacency = _pairwise_adjacency(initial_subpolygons)
    merge_options = _find_mergeable_polygon_options(initial_subpolygons, adjacency)
    selected = _select_merge_options(len(initial_subpolygons), merge_options)

    cells = tuple(
        DecomposedCell(
            id=idx,
            vertices=tuple(
                Vector2(x, y) for x, y in list(option.polygon.exterior.coords)[:-1]
            ),
            area=float(option.polygon.area),
        )
        for idx, option in enumerate(selected)
    )
    cell_polygons = [option.polygon for option in selected]
    adjacencies = _build_adjacencies(cells, cell_polygons)
    return DecompositionResult(cells=cells, adjacencies=adjacencies)


# -- environment -> boundary polygon --------------------------------------


def _boundary_polygon_from_env(env: Environment) -> NavPolygon:
    """Return the arena's outer boundary as a navcore ``Polygon``.

    Falls back to the axis-aligned ``arena_width``/``arena_height``
    rectangle when no walled ``Boundary`` obstacle is present under
    ``env.obstacles["boundary"]`` -- ``build_boundary()`` is currently
    commented out in ``EnvironmentBuilder``, so this is the common case
    today, not an edge case.
    """
    boundary_obstacle = env.obstacles.get("boundary")
    if boundary_obstacle is not None and isinstance(
        boundary_obstacle.geometry, NavPolygon
    ):
        return boundary_obstacle.geometry

    half_width = float(env.info.arena_width) / 2.0
    half_height = float(env.info.arena_height) / 2.0
    return NavPolygon(
        (
            Vector2(-half_width, -half_height),
            Vector2(half_width, -half_height),
            Vector2(half_width, half_height),
            Vector2(-half_width, half_height),
        )
    )


# -- navcore <-> shapely conversion ---------------------------------------


def _navpolygon_to_shapely(polygon: NavPolygon) -> ShapelyPolygon:
    return ShapelyPolygon([(v.x, v.y) for v in polygon.vertices])


def _obstacle_to_shapely_polygon(obstacle: Obstacle) -> ShapelyPolygon:
    """Convert one navcore ``Obstacle`` into a world-frame shapely polygon.

    Raises:
        TypeError: If ``obstacle.geometry`` is not ``Polygon``,
            ``Rectangle``, or ``Circle``.
    """
    geometry = obstacle.geometry

    if isinstance(geometry, NavPolygon):
        return ShapelyPolygon([(v.x, v.y) for v in geometry.vertices])
    if isinstance(geometry, Rectangle):
        return ShapelyPolygon(
            [(v.x, v.y) for v in _rectangle_to_world_vertices(geometry)]
        )
    if isinstance(geometry, Circle):
        return ShapelyPolygon([(v.x, v.y) for v in _circle_to_world_vertices(geometry)])

    raise TypeError(
        f"decompose() cannot convert obstacle geometry {type(geometry).__name__}."
    )


def _rectangle_to_world_vertices(rectangle: Rectangle) -> tuple[Vector2, ...]:
    """Return ``rectangle``'s corners translated into world coordinates.

    ``Rectangle.vertices()`` is local-frame by design (known upstream
    bug in ``obstacle_to_vertices``); see module docstring in
    ``base_orca_planner``.
    """
    return tuple(v + rectangle.center for v in rectangle.vertices())


def _circle_to_world_vertices(circle: Circle) -> tuple[Vector2, ...]:
    """Tessellate ``circle`` into a many-sided polygon, in world coordinates."""
    return tuple(
        circle.center
        + Vector2(
            circle.radius * cos(2.0 * pi * i / _CIRCLE_SEGMENTS),
            circle.radius * sin(2.0 * pi * i / _CIRCLE_SEGMENTS),
        )
        for i in range(_CIRCLE_SEGMENTS)
    )


# -- free space --------------------------------------------------------------


def _free_space(
    boundary: ShapelyPolygon, obstacles: list[ShapelyPolygon]
) -> ShapelyPolygon:
    boundary = boundary if boundary.is_valid else boundary.buffer(0)
    cleaned_obstacles = [obs if obs.is_valid else obs.buffer(0) for obs in obstacles]

    free = boundary
    if cleaned_obstacles:
        free = free.difference(unary_union(cleaned_obstacles))
    return free if free.is_valid else free.buffer(0)


# -- phase 1: interior extension of edges -----------------------------------


def _signed_area(coords: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _is_reflex(
    prev: tuple[float, float],
    curr: tuple[float, float],
    nxt: tuple[float, float],
    ccw: bool,
) -> bool:
    x1, y1 = curr[0] - prev[0], curr[1] - prev[1]
    x2, y2 = nxt[0] - curr[0], nxt[1] - curr[1]
    cross = x1 * y2 - y1 * x2
    return cross < 0 if ccw else cross > 0


def _edge_extension_split(component: ShapelyPolygon) -> list[ShapelyPolygon]:
    """Split one connected free-space component along every edge-extension cut.

    Reflex vertices are found on ``component``'s own exterior and interior
    (hole) rings -- see module docstring for why this must operate on the
    free-space geometry itself, not on obstacles tested independently.
    """
    rings = [component.exterior, *component.interiors]

    reflex_triples: list[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = []
    for ring in rings:
        coords = [(float(x), float(y)) for x, y in list(ring.coords)[:-1]]
        if len(coords) < 3:
            continue
        ccw = _signed_area(coords) > 0
        n = len(coords)
        for i in range(n):
            prev, curr, nxt = coords[(i - 1) % n], coords[i], coords[(i + 1) % n]
            if _is_reflex(prev, curr, nxt, ccw):
                reflex_triples.append((prev, curr, nxt))

    if not reflex_triples:
        return [component]

    minx, miny, maxx, maxy = component.bounds
    pad = max(maxx - minx, maxy - miny) + 10.0

    cut_segments: list[LineString] = []
    for prev, curr, nxt in reflex_triples:
        # Extend edge (prev, curr) forward past curr, and edge (curr, nxt)
        # backward past curr -- the two edges meeting at this reflex
        # vertex, each continued along its own line into the interior.
        cut_segments.append(_edge_extension_ray(curr, prev, rings, pad))
        cut_segments.append(_edge_extension_ray(curr, nxt, rings, pad))

    geometry = component
    for segment in cut_segments:
        geometry = _split_with_line(geometry, segment)

    return [p for p in _iter_polygons(geometry) if p.area > _EPS]


def _edge_extension_ray(
    vertex: tuple[float, float],
    away_from: tuple[float, float],
    rings: list,
    pad: float,
) -> LineString:
    """Return the ray from ``vertex``, continuing the line through
    ``away_from`` and ``vertex``, trimmed to its nearest hit against ``rings``.
    """
    vx, vy = vertex
    ax, ay = away_from
    dx, dy = vx - ax, vy - ay
    length = math.hypot(dx, dy)
    if length < _EPS:
        return LineString([(vx, vy), (vx, vy)])
    dx, dy = dx / length, dy / length

    far_x, far_y = vx + dx * pad, vy + dy * pad
    ray = LineString([(vx, vy), (far_x, far_y)])

    nearest_distance = pad
    for ring in rings:
        intersection = ray.intersection(ring)
        for hx, hy in _extract_ray_hit_coordinates(intersection):
            dist = math.hypot(hx - vx, hy - vy)
            if _EPS < dist < nearest_distance:
                nearest_distance = dist

    end_x, end_y = vx + dx * nearest_distance, vy + dy * nearest_distance
    return LineString([(vx, vy), (end_x, end_y)])


def _extract_ray_hit_coordinates(geometry) -> list[tuple[float, float]]:
    """Flatten a ray-vs-wall intersection result into ``(x, y)`` points."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [(geometry.x, geometry.y)]
    if geometry.geom_type == "MultiPoint":
        return [(p.x, p.y) for p in geometry.geoms]
    if geometry.geom_type == "LineString":
        return [(x, y) for x, y in geometry.coords]
    if geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        points: list[tuple[float, float]] = []
        for part in geometry.geoms:
            points.extend(_extract_ray_hit_coordinates(part))
        return points
    return []


def _split_with_line(geometry, line: LineString):
    pieces = []
    for poly in _iter_polygons(geometry):
        try:
            result = split(poly, line)
            pieces.extend(p for p in _iter_polygons(result) if p.area > _EPS)
        except Exception:
            pieces.append(poly)

    if not pieces:
        return geometry
    if len(pieces) == 1:
        return pieces[0]
    return unary_union(pieces)


def _iter_polygons(geometry):
    if geometry.is_empty:
        return
    if isinstance(geometry, ShapelyPolygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for g in geometry.geoms:
            if not g.is_empty and g.area > 0:
                yield g
    elif isinstance(geometry, GeometryCollection):
        for g in geometry.geoms:
            if isinstance(g, ShapelyPolygon) and not g.is_empty and g.area > 0:
                yield g
            elif isinstance(g, MultiPolygon):
                yield from _iter_polygons(g)


# -- phase 2: convex merge option search (Algorithm 1) -----------------------


def _pairwise_adjacency(polygons: list[ShapelyPolygon]) -> dict[int, set[int]]:
    """Return, for each initial sub-polygon index, the set of indices it
    shares a boundary edge with.
    """
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(polygons))}
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            shared = polygons[i].boundary.intersection(polygons[j].boundary)
            if shared.is_empty:
                continue
            if getattr(shared, "length", 0.0) > _EPS:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def _is_convex_polygon(polygon: ShapelyPolygon) -> bool:
    if polygon.is_empty or polygon.area <= _EPS:
        return False
    hull = polygon.convex_hull
    return abs(hull.area - polygon.area) <= _CONVEXITY_REL_TOL * max(hull.area, _EPS)


def _polygon_width(polygon: ShapelyPolygon) -> float:
    """Return the minimum width of a convex ``polygon``.

    The minimum width of a convex polygon equals the shorter side of its
    minimum-area enclosing rectangle (a standard rotating-calipers
    result) -- shapely's ``minimum_rotated_rectangle`` gives that
    rectangle directly, avoiding a hand-rolled calipers implementation.
    """
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 4:
        return 0.0
    edge_a = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    edge_b = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    return min(edge_a, edge_b)


def _options_adjacent(
    left: _MergeOption, right: _MergeOption, adjacency: dict[int, set[int]]
) -> bool:
    return any(b in adjacency.get(a, ()) for a in left.indices for b in right.indices)


def _find_mergeable_polygon_options(
    polygons: list[ShapelyPolygon], adjacency: dict[int, set[int]]
) -> list[_MergeOption]:
    """Return every valid convex merge option, per the paper's Algorithm 1.

    ``levels[k]`` holds every option built from exactly ``k`` initial
    sub-polygons. An option of size ``i`` is only assembled from one
    option of size ``floor(i/2)`` and one of size ``ceil(i/2)`` -- the
    paper's approximation, trading completeness for tractability (see
    module docstring's performance caveat).
    """
    n = len(polygons)
    levels: dict[int, list[_MergeOption]] = {1: []}
    all_options: dict[frozenset[int], _MergeOption] = {}

    for i in range(n):
        option = _MergeOption(
            indices=frozenset({i}),
            polygon=polygons[i],
            width=_polygon_width(polygons[i]),
        )
        levels[1].append(option)
        all_options[option.indices] = option

    for size in range(2, n + 1):
        n1, n2 = size // 2, size - size // 2
        level_n1 = levels.get(n1, [])
        level_n2 = levels.get(n2, [])

        pairs: Iterable[tuple[_MergeOption, _MergeOption]] = (
            itertools.combinations(level_n1, 2)
            if n1 == n2
            else itertools.product(level_n1, level_n2)
        )

        found: list[_MergeOption] = []
        for left, right in pairs:
            if left.indices & right.indices:
                continue  # Must be disjoint to form a valid size-`size` option.
            if not _options_adjacent(left, right, adjacency):
                continue

            merged_geometry = unary_union([left.polygon, right.polygon])
            if not isinstance(merged_geometry, ShapelyPolygon):
                continue  # Not spatially contiguous as a single polygon.
            if not _is_convex_polygon(merged_geometry):
                continue

            combined_indices = left.indices | right.indices
            if combined_indices in all_options:
                continue  # Already found via a different (n1, n2) split.

            option = _MergeOption(
                indices=combined_indices,
                polygon=merged_geometry,
                width=_polygon_width(merged_geometry),
            )
            found.append(option)
            all_options[combined_indices] = option

        levels[size] = found

    return list(all_options.values())


# -- phase 3: merge option selection (exact IP) -------------------------------


def _select_merge_options(
    num_initial_subpolygons: int, options: list[_MergeOption]
) -> list[_MergeOption]:
    """Solve the paper's set-partitioning IP (eqs. 1-3) exactly via
    ``scipy.optimize.milp``.

    Minimizes total selected width subject to every initial sub-polygon
    being covered by exactly one selected option. Always feasible: every
    singleton option (one per initial sub-polygon) is present in
    ``options`` by construction, so selecting all of them is a trivial
    feasible (if suboptimal) partition.

    Raises:
        RuntimeError: If the solver fails to find a feasible partition --
            should not happen given the fallback above, but surfaced
            loudly rather than silently returning an incomplete cell set.
    """
    if not options:
        return []

    num_options = len(options)
    coverage = np.zeros((num_initial_subpolygons, num_options), dtype=np.float64)
    for col, option in enumerate(options):
        for row in option.indices:
            coverage[row, col] = 1.0

    costs = np.array([option.width for option in options], dtype=np.float64)
    constraint = LinearConstraint(coverage, lb=1.0, ub=1.0)
    bounds = Bounds(lb=0.0, ub=1.0)
    integrality = np.ones(num_options, dtype=np.int_)

    result = milp(
        c=costs, constraints=constraint, bounds=bounds, integrality=integrality
    )

    if not result.success:
        raise RuntimeError(
            f"Merge-option selection failed to find a feasible partition: {result.message}"
        )

    return [option for option, selected in zip(options, result.x > 0.5) if selected]


# -- final cell adjacency ----------------------------------------------------


def _build_adjacencies(
    cells: tuple[DecomposedCell, ...],
    cell_polygons: list[ShapelyPolygon],
) -> tuple[CellAdjacency, ...]:
    """Return every shared-edge relationship between the final selected cells."""
    adjacencies: list[CellAdjacency] = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            shared = cell_polygons[i].boundary.intersection(cell_polygons[j].boundary)
            if shared.is_empty:
                continue
            if getattr(shared, "length", 0.0) > _EPS:
                adjacencies.append(CellAdjacency(cells[i].id, cells[j].id))
    return tuple(adjacencies)

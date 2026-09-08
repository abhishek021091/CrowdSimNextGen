"""Full-coverage traversal planning over a DecompositionResult's adjacency graph.

Where environment_decomposition.py answers "what are the convex cells and
how do they connect," this module answers "in what order, and via which
entry/exit points, should every cell be visited so all of them get swept
exactly once." It knows nothing about ORCA, SweepingMission, or the Step
loop -- same separation rationale as environment_decomposition.py's "BCD
decomposes; it does not sweep." This module doesn't sweep either; it only
sequences.

Traversal strategy -- DFS with backtracking, not greedy nearest-unvisited:
    A greedy "always move to the nearest unvisited neighbor" strategy can
    strand the robot in a cell whose every neighbor is already visited,
    with unvisited cells still reachable only through cells behind it --
    greedy has no way back without an actual graph search. Depth-first
    search solves this by construction: when a cell's neighbors are all
    visited, the recursion unwinds back through the DFS tree until it
    reaches a cell with an unvisited neighbor, and continues from there.
    This also matches the traversal strategy in the original Boustrophedon
    Cellular Decomposition literature (Choset & Pignon, 1997), for the
    same reason. Every cell is swept exactly once, on its first visit; any
    later re-entry into an already-swept cell is a transit-only step,
    needed purely to reach the unvisited side of the graph.

Multiple connected components:
    Free space can be split into regions with no adjacency path between
    them (obstacles fully partitioning the arena into separate rooms).
    Those aren't a graph-traversal problem -- a single continuous path
    physically cannot connect them -- so ``plan_full_coverage`` returns
    one traversal per connected component rather than raising or silently
    dropping unreachable cells.

Not yet wired to SweepingMission:
    ``SweepingMission.reach_closest_corner``/``update_sweep`` currently
    sweep the *whole arena*, bounded by ``env.info.arena_width``/
    ``arena_height`` -- not an arbitrary convex cell between two given
    points. Turning a ``TraversalStep`` into an actual sweep requires
    generalizing (or replacing) that mission to sweep a caller-supplied
    convex polygon between an entry and exit point; that doesn't exist
    yet and isn't guessed at here.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon as ShapelyPolygon

from navcore.entities.components.geometry.vector2 import Vector2
from navcore.planning.environment_decomposition import (
    DecomposedCell,
    DecompositionResult,
)


@dataclass(frozen=True, slots=True)
class TraversalStep:
    """One instruction in a full-coverage traversal.

    Attributes:
        cell: The cell this step's motion takes place in.
        entry_point: Where the robot enters ``cell`` for this step, in
            world coordinates.
        exit_point: Where the robot exits ``cell`` for this step, or
            ``None`` if this is the traversal's final step (nowhere
            left to go).
        requires_sweep: Whether a coverage pass should run through
            ``cell`` during this step. ``True`` exactly once per cell,
            on its first visit; ``False`` on any later pass-through
            step needed only to reach an unvisited neighbor beyond it.
    """

    cell: DecomposedCell
    entry_point: Vector2
    exit_point: Vector2 | None
    requires_sweep: bool


def plan_full_coverage(
    result: DecompositionResult, start_point: Vector2 | None = None
) -> list[list[TraversalStep]]:
    """Plan a full-coverage traversal of every cell in ``result``.

    Args:
        result: The decomposition to traverse.
        start_point: Where the robot currently is. The cell containing
            it becomes the starting cell for whichever connected
            component it belongs to. If omitted, or if it falls inside
            no cell, each component starts from its lowest-id cell.

    Returns:
        One traversal (an ordered list of ``TraversalStep``) per
        connected component of ``result``'s adjacency graph -- see
        module docstring for why disconnected components aren't merged
        into one path. Most single-room arenas will get back a list of
        length one; concatenate it yourself if the component boundary
        doesn't matter for your use case.
    """
    if not result.cells:
        return []

    cells_by_id = {cell.id: cell for cell in result.cells}
    neighbors = {cell.id: set(result.neighbors_of(cell.id)) for cell in result.cells}
    components = _connected_components(neighbors)

    preferred_start: int | None = None
    if start_point is not None:
        containing = result.cell_containing(start_point)
        if containing is not None:
            preferred_start = containing.id

    traversals: list[list[TraversalStep]] = []
    for component in components:
        if preferred_start in component:
            start_id = preferred_start
            entry = start_point
        else:
            start_id = min(component)
            entry = _cell_centroid(cells_by_id[start_id])

        steps: list[TraversalStep] = []
        visited: set[int] = set()
        _dfs(start_id, entry, cells_by_id, neighbors, visited, steps)
        traversals.append(steps)

    return traversals


def _dfs(
    cell_id: int,
    entry_point: Vector2,
    cells_by_id: dict[int, DecomposedCell],
    neighbors: dict[int, set[int]],
    visited: set[int],
    steps: list[TraversalStep],
) -> None:
    """Depth-first traversal producing one ``TraversalStep`` per edge crossed.

    Re-derives ``unvisited`` on every loop iteration rather than once at
    entry, so cross-edges discovered mid-recursion (a neighbor visited via
    a different branch while this call was still descending) are correctly
    excluded from further consideration -- this is what makes the walk
    correct for graphs with cycles, not just trees.
    """
    visited.add(cell_id)
    current_entry = entry_point
    first_visit = True

    while True:
        unvisited = [n for n in neighbors[cell_id] if n not in visited]
        if not unvisited:
            steps.append(
                TraversalStep(cells_by_id[cell_id], current_entry, None, first_visit)
            )
            return

        next_id = unvisited[0]
        exit_point = _shared_edge_midpoint(cells_by_id[cell_id], cells_by_id[next_id])
        steps.append(
            TraversalStep(cells_by_id[cell_id], current_entry, exit_point, first_visit)
        )
        first_visit = False

        _dfs(next_id, exit_point, cells_by_id, neighbors, visited, steps)
        current_entry = exit_point  # Re-enter this cell where we last exited it.


def _connected_components(neighbors: dict[int, set[int]]) -> list[set[int]]:
    unassigned = set(neighbors.keys())
    components: list[set[int]] = []

    while unassigned:
        root = next(iter(unassigned))
        stack = [root]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(neighbors[node] - component)
        components.append(component)
        unassigned -= component

    return components


def _shared_edge_midpoint(cell_a: DecomposedCell, cell_b: DecomposedCell) -> Vector2:
    """Return the midpoint of the boundary segment shared by two adjacent cells.

    Raises:
        ValueError: If the cells don't actually share a boundary segment
            -- should not happen for a pair returned by
            ``DecompositionResult.neighbors_of``, so this indicates a
            caller passed mismatched cells/adjacency rather than a
            normal runtime condition.
    """
    poly_a = ShapelyPolygon([(v.x, v.y) for v in cell_a.vertices])
    poly_b = ShapelyPolygon([(v.x, v.y) for v in cell_b.vertices])
    shared = poly_a.boundary.intersection(poly_b.boundary)

    if shared.is_empty:
        raise ValueError(
            f"Cells {cell_a.id} and {cell_b.id} do not share a boundary segment."
        )

    if shared.geom_type == "LineString":
        return Vector2(shared.centroid.x, shared.centroid.y)

    if shared.geom_type in ("MultiLineString", "GeometryCollection"):
        line_parts = [g for g in shared.geoms if g.geom_type == "LineString"]
        if line_parts:
            longest = max(line_parts, key=lambda g: g.length)
            return Vector2(longest.centroid.x, longest.centroid.y)

    return Vector2(shared.centroid.x, shared.centroid.y)


def _cell_centroid(cell: DecomposedCell) -> Vector2:
    poly = ShapelyPolygon([(v.x, v.y) for v in cell.vertices])
    centroid = poly.centroid
    return Vector2(centroid.x, centroid.y)

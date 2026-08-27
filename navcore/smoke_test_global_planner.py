from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.planning.boustrophedon_planner import BoustrophedonPlanner
from navcore.planning.coverage_grid import CoverageGrid
from navcore.planning.euclidean_nearest_planner import EuclideanNearestPlanner
from navcore.planning.graph_search_planner import GraphSearchPlanner


def build_sealed_pocket_grid() -> CoverageGrid:
    """A 5x5-cell grid, all free, with one cell fully sealed off (4-connected).

    Cell (2, 2) is surrounded on all four sides by manually-occupied
    cells, so it is geometrically close to the start but unreachable
    without diagonal movement. Cell (0, 0) is far but freely reachable.
    Every other cell is pre-marked swept, so only these two are
    candidates -- isolating the reachability question cleanly.
    """
    grid = CoverageGrid(width=5.0, height=5.0, cell_size=1.0, obstacles=[])

    for row, col in [(1, 2), (3, 2), (2, 1), (2, 3)]:
        grid.occupied[row, col] = True

    grid.swept[:, :] = True
    grid.swept[2, 2] = False
    grid.swept[0, 0] = False

    return grid


def test_euclidean_picks_the_close_but_unreachable_pocket():
    grid = build_sealed_pocket_grid()
    planner = EuclideanNearestPlanner()
    start = grid.cell_to_world(4, 4)

    waypoint = planner.next_waypoint(start, grid)
    assert waypoint is not None
    picked_cell = grid.world_to_cell(waypoint)
    print(f"euclidean (obstacle-blind): picked {picked_cell}")
    assert picked_cell == (2, 2)
    print(f"euclidean (obstacle-blind): picked sealed pocket at {picked_cell}")


def test_graph_search_skips_the_unreachable_pocket():
    grid = build_sealed_pocket_grid()
    planner = GraphSearchPlanner(lookahead_distance=10.0)
    start = grid.cell_to_world(4, 4)

    waypoint = planner.next_waypoint(start, grid)
    assert waypoint is not None
    picked_cell = grid.world_to_cell(waypoint)
    print(f"graph-search (respects reachability): picked {picked_cell}")
    assert picked_cell == (0, 0)
    print(f"graph-search (respects reachability): skipped pocket, picked {picked_cell}")


def test_boustrophedon_sweeps_lines_and_advances():
    # 6 cols x 4 rows: exactly 4 sweep lines available at row_step=1.
    grid = CoverageGrid(width=6.0, height=4.0, cell_size=1.0, obstacles=[])
    planner = BoustrophedonPlanner(row_step=1)

    position = Vector2(-2.5, -1.5)  # bottom-left corner cell
    waypoints = []
    for _ in range(4):
        target = planner.next_waypoint(position, grid)
        assert target is not None
        print(f"boustrophedon: from {position} to {target}")
        waypoints.append(target)
        grid.mark_swept(target)
        position = target

    print("boustrophedon path:", waypoints)
    # each call jumps straight to the far edge of the current line's free
    # run ("extended goal") -- not an incremental step -- so direction
    # should alternate every call: right, left, right, left.
    assert waypoints[0].x > 0 and waypoints[1].x < 0
    assert waypoints[2].x > 0 and waypoints[3].x < 0
    # each successive line should be a different row (y increases)
    assert waypoints[0].y < waypoints[1].y < waypoints[2].y < waypoints[3].y

    # arena is now fully covered -- a 5th call must report completion
    assert planner.next_waypoint(position, grid) is None


if __name__ == "__main__":
    test_euclidean_picks_the_close_but_unreachable_pocket()
    test_graph_search_skips_the_unreachable_pocket()
    test_boustrophedon_sweeps_lines_and_advances()
    print("\nAll GlobalPlanner smoke tests passed.")

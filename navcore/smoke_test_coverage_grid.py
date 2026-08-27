from navcore.entities.components.geometry.circle import Circle
from navcore.entities.components.geometry.rectangle import Rectangle
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.obstacles.obstacle import Obstacle
from navcore.planning.coverage_grid import CoverageGrid


def build_test_grid() -> CoverageGrid:
    pillar = Obstacle(
        id="pillar_1",
        geometry=Circle(center=Vector2(2.0, 2.0), radius=0.5),
    )
    table = Obstacle(
        id="table_1",
        geometry=Rectangle(center=Vector2(-2.0, -2.0), width=1.5, height=1.0),
    )
    return CoverageGrid(width=10.0, height=10.0, cell_size=0.5, obstacles=[pillar, table])


def test_pillar_center_is_blocked():
    grid = build_test_grid()
    row, col = grid.world_to_cell(Vector2(2.0, 2.0))
    assert not grid.is_free(row, col)
    print("pillar center blocked: OK")


def test_far_corner_is_free():
    grid = build_test_grid()
    row, col = grid.world_to_cell(Vector2(4.5, 4.5))
    assert grid.is_free(row, col)
    print("far corner free: OK")


def test_coverage_fraction_updates_as_cells_are_swept():
    grid = build_test_grid()
    assert grid.coverage_fraction() == 0.0

    free_cells = list(grid.unswept_free_cells())
    total_free = grid.free_cell_count()
    assert len(free_cells) == total_free

    # sweep exactly half the free cells
    for row, col in free_cells[: total_free // 2]:
        grid.mark_swept(grid.cell_to_world(row, col))

    fraction = grid.coverage_fraction()
    assert 0.45 < fraction < 0.55
    print(f"coverage fraction after half swept: {fraction:.3f}")


def test_traversable_obstacle_is_not_rasterized():
    traversable_rug = Obstacle(
        id="rug_1",
        geometry=Circle(center=Vector2(0.0, 0.0), radius=1.0),
        traversable=True,
    )
    grid = CoverageGrid(width=10.0, height=10.0, cell_size=0.5, obstacles=[traversable_rug])
    row, col = grid.world_to_cell(Vector2(0.0, 0.0))
    assert grid.is_free(row, col)
    print("traversable obstacle not rasterized: OK")


if __name__ == "__main__":
    test_pillar_center_is_blocked()
    test_far_corner_is_free()
    test_coverage_fraction_updates_as_cells_are_swept()
    test_traversable_obstacle_is_not_rasterized()
    print("\nAll CoverageGrid smoke tests passed.")

"""Smoke test for GlobalPlanner.

Mirrors test_sweep.py's structure (build once, run_simulation() drives
the loop, print status at the end) but exercises the full BCD +
traversal + multi-cell sweep pipeline instead of a single whole-arena
sweep.
"""

from navcore.planner.global_planner import GlobalPlanner


class GlobalPlannerTest:
    def __init__(self) -> None:
        self.planner = GlobalPlanner()

    def run_simulation(self) -> None:
        stats = self.planner.run()
        print(
            f"Cells completed: {stats.cells_completed}/{stats.cells_total} "
            f"across {len(self.planner.traversals)} traversal(s)"
        )
        print(f"Total area swept: {stats.total_area_swept:.2f} m^2")
        print(f"Total collisions: {stats.collisions}")


if __name__ == "__main__":
    GlobalPlannerTest().run_simulation()

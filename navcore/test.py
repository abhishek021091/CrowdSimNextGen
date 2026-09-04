"""Manual end-to-end smoke test: robot sweeps the arena while pedestrians
continually regenerate, with the swept-coverage overlay rendered live.

Design note -- why this is one continuous run, not N reset episodes:
    Coverage is naturally bounded by "has the robot swept the whole
    grid," not by an episode counter, and MatplotlibVisualizer/CoverageGrid
    are currently bound to one env/grid instance for their lifetime (see
    project roadmap: Simulation.reset()/run_episode() is still deferred).
    Swapping to a fresh Environment mid-run would leave the visualizer
    pointed at a stale env. So this script runs a single sweep to
    completion, regenerating the *crowd* (not the environment, robot, or
    grid) whenever every current pedestrian has reached its goal.
"""

from __future__ import annotations

import math

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.missions.sweep_mission import SweepMission
from navcore.planning.coverage_grid import CoverageGrid
from navcore.step.step import Step, StepResult
from navcore.visibility.group_personal_space import GroupPersonalSpaceVisibility
from navcore.visibility.visibility_policy import AsymmetricVisibility
from navcore.visualization.visualizer import MatplotlibVisualizer

#: How close (world units) a pedestrian must be to its goal to count as
#: "arrived." Not wired to VisualizerConfig.goal_reach_threshold on
#: purpose -- that constant is a rendering/UI concern (when the goal
#: marker looks "reached"), this one is a simulation-loop concern (when
#: to regenerate the crowd). They're allowed to diverge.
PEDESTRIAN_ARRIVAL_TOLERANCE = 0.3


class TestNavigationStack:
    def __init__(self) -> None:
        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()

    def _build_grid(self) -> CoverageGrid:
        env_config = self.env_builder.env_config
        return CoverageGrid(
            width=env_config["arenaSize"]["width"],
            height=env_config["arenaSize"]["height"],
            cell_size=0.5,
            obstacles=list(self.env.obstacles.values()),
        )

    def _all_pedestrians_arrived(self) -> bool:
        crowd = self.env.crowd
        if not crowd:
            return False
        return all(
            ped.pose is not None
            and ped.goal is not None
            and math.hypot(ped.goal.gx - ped.pose.px, ped.goal.gy - ped.pose.py)
            <= PEDESTRIAN_ARRIVAL_TOLERANCE
            for ped in crowd.values()
        )

    def _regenerate_crowd(self) -> None:
        """Replace the current crowd with a fresh batch, in place.

        Robot, obstacles, and the coverage grid are untouched -- only
        the pedestrians (and their groups) are regenerated, so the
        robot's sweep progress carries over uninterrupted.
        """
        self.env.crowd = self.env_builder.build_crowd()
        print("All pedestrians reached their goals -- regenerating crowd.")

    def run_sweep_until_complete(self) -> None:
        grid = self._build_grid()
        robot_mission = SweepMission(grid)
        orca = DecentralizedORCAPlanner(
            config_file="orca.toml",
            obstacles=self.env.obstacles,
        )
        visibility = GroupPersonalSpaceVisibility(
            inner=AsymmetricVisibility(),
            personal_space_radius=0.6,  # tune relative to pedestrian radius + safety margin
        )
        step = Step(
            planner=orca,
            env=self.env,
            robot_visible=False,
            robot_mission=robot_mission,
            visibility=visibility,
        )
        viz = MatplotlibVisualizer(self.env, coverage_grid=grid)

        while not robot_mission.finished:
            robot = self.env.robot
            if robot.pose is None or robot.goal is None:
                raise RuntimeError("Robot pose or goal has not been initialized.")

            result: StepResult = step.step()
            print(f"Robot velocity: {result.robot_velocity}")

            if self._all_pedestrians_arrived():
                self._regenerate_crowd()

            viz.refresh()

        print(f"Sweep complete: coverage={robot_mission.coverage_fraction():.2%}")
        viz.show(block=True)


if __name__ == "__main__":
    TestNavigationStack().run_sweep_until_complete()

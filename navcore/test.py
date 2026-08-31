import unittest

import numpy as np

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.missions.sweep_mission import SweepMission
from navcore.planning.coverage_grid import CoverageGrid
from navcore.step.step import Step, StepResult
from navcore.visualization.visualizer import MatplotlibVisualizer


class TestNavigationStack:
    def __init__(self):
        self.episode = 20
        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()

    def run_simulation(self):
        self.viz = MatplotlibVisualizer(self.env)
        for episode in range(self.episode):
            print(f"Episode {episode + 1}/{self.episode}")
            self.test_step_runs_until_goal()
            self.reset(random_seed=episode + 1)

    def _build_grid(self) -> CoverageGrid:
        env_config = self.env_builder.env_config
        return CoverageGrid(
            width=env_config["arenaSize"]["width"],
            height=env_config["arenaSize"]["height"],
            cell_size=0.5,
            obstacles=list(self.env.obstacles.values()),
        )

    def test_step_runs_until_goal(self):
        orca = DecentralizedORCAPlanner(
            config_file="orca.toml",
            obstacles=self.env.obstacles,
        )

        grid = self._build_grid()
        robot_mission = SweepMission(grid)

        step = Step(
            env=self.env,
            robot_visible=False,
            planner=orca,
            robot_mission=robot_mission,
        )
        while True:
            robot = self.env.robot
            if robot.pose is None or robot.goal is None:
                raise RuntimeError("Robot pose or goal has not been initialized.")

            if robot_mission.finished:
                print(
                    f"Sweep complete: coverage={robot_mission.coverage_fraction():.2%}"
                )
                break

            result: StepResult = step.step()
            print(f"Robot velocity: {result.robot_velocity}")
            print(f"Crowd velocities: {result.crowd_velocities}")

            all_pedestrians_reached = all(
                ped.pose is not None
                and ped.goal is not None
                and np.isclose(ped.goal.gx - ped.pose.px, 0.5)
                and np.isclose(ped.goal.gy - ped.pose.py, 0.5)
                for ped in self.env.crowd.values()
            )
            if all_pedestrians_reached:
                print("All pedestrians reached their goals.")
                self.env.crowd = self.env_builder.build_crowd()
            self.viz.refresh()
        self.viz.show(block=False)

    def reset(self, random_seed: int = 43) -> None:
        self.env = self.env_builder.reset(random_seed=random_seed)


if __name__ == "__main__":
    test_nav_stack = TestNavigationStack()
    test_nav_stack.run_simulation()

import unittest

import numpy as np

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.step.step import Step, StepResult
from navcore.visualization.visualizer import MatplotlibVisualizer


class TestNavigationStack:
    def __init__(self):
        self.episode = 20
        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()

    def run_simulation(self):
        # import pdb

        # pdb.set_trace()
        self.viz = MatplotlibVisualizer(self.env)
        for episode in range(self.episode):
            print(f"Episode {episode + 1}/{self.episode}")
            self.test_step_runs_until_goal()
            self.reset(random_seed=episode + 1)

    def test_step_runs_until_goal(self):
        orca = DecentralizedORCAPlanner(
            config_file="orca.toml",
            obstacles=self.env.obstacles,
        )

        step = Step(env=self.env, robot_visible=False, planner=orca)
        while True:
            robot = self.env.robot
            if robot.pose is None or robot.goal is None:
                raise RuntimeError("Robot pose or goal has not been initialized.")

            distance = np.hypot(
                robot.goal.gx - robot.pose.px,
                robot.goal.gy - robot.pose.py,
            )
            print(f"Distance to goal: {distance:.2f}")
            if distance <= 0.5:
                print("Robot reached its goal.")
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

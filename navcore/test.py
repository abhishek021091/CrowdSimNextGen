import unittest

import numpy as np

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.step.step import Step, StepResult
from navcore.visualization.matplotlib_visualizer import visualize_simulation


class TestNavigationStack(unittest.TestCase):
    def setUp(self):
        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()

    def test_environment_built(self):
        self.assertIsNotNone(self.env)
        print(self.env)
        print("Environment built successfully.")

    def test_step_runs_until_goal(self):
        orca = DecentralizedORCAPlanner(
            config_file="orca.toml",
            obstacles=self.env.obstacles,
        )

        step = Step(env=self.env, robot_visible=False, planner=orca)

        while True:
            robot = self.env.robot
            if robot.pose is None or robot.goal is None:
                self.fail("Robot pose or goal is missing.")

            dx = robot.goal.gx - robot.pose.px
            dy = robot.goal.gy - robot.pose.py

            if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
                print("Robot reached its goal.")
                break
            result: StepResult = step.step()
            print(f"Robot velocity: {result.robot_velocity}")
            print(f"Crowd velocities: {result.crowd_velocities}")

            all_pedestrians_reached = all(
                ped.pose is not None
                and ped.goal is not None
                and np.isclose(ped.goal.gx - ped.pose.px, 0.0)
                and np.isclose(ped.goal.gy - ped.pose.py, 0.0)
                for ped in self.env.crowd.values()
            )
            if all_pedestrians_reached:
                print("All pedestrians reached their goals.")
                self.env.crowd = self.env_builder.build_crowd()
            visualize_simulation(
                self.env, step, interval=80, trail_length=40
            )  # Rebuild the crowd for the next iteration


if __name__ == "__main__":
    unittest.main()

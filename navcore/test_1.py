"""Small manual smoke test for the Matplotlib visualizer."""

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.visualization_1.visualizer import Visualizer
from navcore.step.step import Step
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner


class TestVisualizer:
    def __init__(self) -> None:
        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()
        print(self.env.crowd)

    def test_environment_visualization(self) -> None:
        visualizer = Visualizer()
        visualizer.animate(self.env)

    def step(self) -> None:
        """Advance the simulation by one step."""
        step = Step(
            env=self.env,
            robot_visible=False,
            planner=DecentralizedORCAPlanner(
                config_file="orca.toml", obstacles=self.env.obstacles
            ),
        )
        result = step.step()
        return result  # Return the result for potential further analysis

    def run_simulation(self) -> None:
        """Run the simulation for a number of steps."""
        visualizer = Visualizer()
        count = 0
        while True:
            count += 1
            result = self.step()
            visualizer.refresh(self.env)
            if result.robot_reached_goal:
                print("Robot reached its goal!")
                break
            if result.collision_happened:
                print("Collision occurred!")
            for ped_id, reached in result.pedestrian_reached_goals.items():
                if reached:
                    print(f"Pedestrian {ped_id} reached its goal!")
                    self.env_builder.rebuild_pedestrian(
                        ped_id=ped_id,
                        env=self.env,
                        random_seed=self.env.info.random_seed + count + ped_id,
                    )


if __name__ == "__main__":
    test = TestVisualizer()
    test.run_simulation()

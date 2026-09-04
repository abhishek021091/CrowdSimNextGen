from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.missions.sweeping import Sweep, SweepingMission
from navcore.visualization_1.visualizer import Visualizer
from navcore.step.step import Step
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner


class SweepTest:
    def __init__(self) -> None:
        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()
        self.sweeping_mission = SweepingMission(env=self.env)

    def visualize(self) -> None:
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
        return result

    def run_simulation(self) -> None:
        """Run the simulation for a number of steps."""
        visualizer = Visualizer()
        count = 0
        result = self.step()
        while True:
            count += 1
            result = self.step()
            visualizer.refresh(self.env)
            if not Sweep.started:
                self.sweeping_mission.reach_closest_corner()
                Sweep.started = True
            if result.robot_reached_goal and Sweep.started and not Sweep.sweeping:
                self.sweeping_mission.update_sweep()
                Sweep.sweeping = True

            if self.sweeping_mission.sweep_finished:
                print("Sweep mission completed!")
                break
            if result.collision_happened:
                Sweep.collisions += 1
            for ped_id, reached in result.pedestrian_reached_goals.items():
                if reached:
                    print(f"Pedestrian {ped_id} reached its goal!")
                    self.env_builder.rebuild_pedestrian(
                        ped_id=ped_id,
                        env=self.env,
                        random_seed=self.env.info.random_seed + count + ped_id,
                    )
            print(self.env.robot.pose, self.env.robot.goal)


if __name__ == "__main__":
    test = SweepTest()
    test.run_simulation()

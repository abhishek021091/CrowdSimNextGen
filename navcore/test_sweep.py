from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.missions.sweeping import SweepingMission
from navcore.step.step import Step
from navcore.visualization_1.visualizer import Visualizer


class SweepTest:
    def __init__(self) -> None:
        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()
        self.sweeping_mission = SweepingMission(env=self.env)

        # Built once, not per tick: obstacles are static for the episode
        # (see DecentralizedORCAPlanner's own docstring), so rebuilding
        # this -- and its obstacle-vertex cache -- on every step() call
        # was pure wasted work in the hot loop.
        self._planner = DecentralizedORCAPlanner(
            config_file="orca.toml", obstacles=self.env.obstacles
        )
        self._step = Step(
            env=self.env,
            robot_visible=False,
            planner=self._planner,
        )

    def visualize(self) -> None:
        visualizer = Visualizer()
        visualizer.animate(self.env)

    def step(self):
        """Advance the simulation by one step."""
        return self._step.step()

    def run_simulation(self) -> None:
        """Run the simulation for a number of steps."""
        visualizer = Visualizer()
        count = 0

        while True:
            count += 1
            result = self.step()
            visualizer.refresh(self.env)

            mission = self.sweeping_mission

            if not mission.started:
                mission.reach_closest_corner()

            elif result.robot_reached_goal:
                # First arrival (at the corner) transitions into the
                # sweeping phase; every arrival after that advances one
                # more lane step. Both cases need update_sweep() --
                # only the flag flip differs.
                if not mission.sweeping:
                    mission.sweeping = True
                mission.update_sweep()

            if mission.sweep_finished:
                print("Sweep mission completed!")
                break

            if result.collision_happened:
                mission.collisions += 1

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

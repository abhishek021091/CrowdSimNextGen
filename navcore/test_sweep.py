from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.collision_predictor.sat import SAT
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

            self.robot_observation = self.env.robot.sensor.observe(
                self.env, robot_visible=False
            )
            collision_predictor = SAT(
                self.robot_observation, self.env.robot, self.env.obstacles
            ).checkIntrusionSAT()
            if collision_predictor and not mission.avoiding_obstacle:
                mission.avoiding_obstacle = True
                last_pose = self.env.robot.pose
                last_goal = self.env.robot.goal
            elif not collision_predictor and not mission.avoiding_obstacle:
                self.env.robot.set_goal_position(last_pose)
            else:
                distance_to_last_pose = np.linalg.norm(
                    [
                        self.env.robot.pose.px - last_pose.px,
                        self.env.robot.pose.py - last_pose.py,
                    ]
                )
                if distance_to_last_pose < 0.1:
                    mission.avoiding_obstacle = False
                    self.env.robot.set_goal_position(last_goal)

            if mission.sweep_finished:
                print("Sweep mission completed!")
            if self.env.did_collision_happened():
                print(f"Total collisions: {self.env.info.collision_counter}")

            area_swept = mission.total_area_swept()
            print(f"Area swept so far: {area_swept:.2f} m^2")

            for ped_id, reached in result.pedestrian_reached_goals.items():
                if reached:
                    self.env_builder.rebuild_pedestrian(
                        ped_id=ped_id,
                        env=self.env,
                        random_seed=self.env.info.random_seed + count + ped_id,
                    )


if __name__ == "__main__":
    test = SweepTest()
    test.run_simulation()

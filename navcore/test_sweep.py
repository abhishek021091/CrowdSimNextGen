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

        self.mission = SweepingMission(self.env)

        self._planner = DecentralizedORCAPlanner(
            config_file="orca.toml",
        )

        self._step = Step(
            env=self.env,
            robot_visible=False,
            planner=self._planner,
        )

        self.visualizer = Visualizer()

    def step(self):
        return self._step.step()

    def _collision_predictor(self) -> SAT:
        observation = self.env.robot.sensor.observe(
            self.env,
            robot_visible=False,
        )

        return SAT(
            observation,
            self.env.robot,
            self.env.obstacles,
        )

    def _update_mission(self, result) -> None:
        if not self.mission.started:
            self.mission.reach_closest_corner()
            return

        predictor = self._collision_predictor()

        self.mission.avoid_crowd(
            predictor=predictor,
            step=self._step.step,
            safe_point_finder=...,  # your SafePointFinder
        )

        if self.mission.avoiding_obstacle:
            return

        if result.robot_reached_goal:
            if not self.mission.sweeping:
                self.mission.sweeping = True

            self.mission.update_sweep()

    def _respawn_pedestrians(self, result, count: int) -> None:
        for ped_id, reached in result.pedestrian_reached_goals.items():
            if reached:
                self.env_builder.rebuild_pedestrian(
                    ped_id=ped_id,
                    env=self.env,
                    random_seed=self.env.info.random_seed + count + ped_id,
                )

    def _print_status(self) -> None:
        if self.mission.sweep_finished:
            print("Sweep mission completed.")

        if self.env.did_collision_happened():
            print(f"Total collisions: {self.env.info.collision_counter}")

        print(f"Area swept: {self.mission.total_area_swept():.2f} m²")

    def run_simulation(self) -> None:
        step_count = 0

        while not self.mission.sweep_finished:
            step_count += 1

            result = self.step()

            self._update_mission(result)

            self.visualizer.refresh(self.env)

            self._print_status()

            self._respawn_pedestrians(result, step_count)

        print("Simulation finished.")


if __name__ == "__main__":
    SweepTest().run_simulation()

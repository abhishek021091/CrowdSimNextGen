import argparse
from navcore.avoidace_planner.local_avoidace_planner import LocalAvoidancePlanner
from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.collision_predictor.sat import SAT
from navcore.entities.components.pose import Pose
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.environment.environment import Environment
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.missions.sweeping import SweepingMission
from navcore.step.step import Step, StepResult
from navcore.visualization.visualizer import Visualizer
from navcore.planner.global_planner import GlobalPlanner
from navcore.planner.task_planners import GoalPlanner, WaypointPlanner


class SafePointFinder:
    """Temporary bridge: SweepingMission.avoid_crowd expects
    find_safe_point(pose) -> (x, y) | None. LocalAvoidancePlanner exposes
    plan_escape_point(human_states, reference_goal) -> Vector2 | None
    instead. Revisit/delete once avoid_crowd is replaced by the planned
    AVOIDING state (see project roadmap) -- at that point the state
    should probably call LocalAvoidancePlanner directly.
    """

    def __init__(self, planner: LocalAvoidancePlanner, env: Environment) -> None:
        self._planner = planner
        self._env = env

    def find_safe_point(self, pose: Pose) -> tuple[float, float] | None:
        assert self._env.robot.sensor is not None
        observation = self._env.robot.sensor.observe(self._env, robot_visible=False)
        point = self._planner.plan_escape_point(observation)
        return point.to_tuple() if point is not None else None


class SweepTest:
    def __init__(self) -> None:
        self.env_builder = EnvironmentBuilder()
        self.env = self.env_builder.build_environment()

        half_width = float(self.env.info.arena_width) / 2.0
        half_height = float(self.env.info.arena_height) / 2.0
        self.mission = SweepingMission(
            self.env,
            cell_vertices=(
                Vector2(-half_width, -half_height),
                Vector2(half_width, -half_height),
                Vector2(half_width, half_height),
                Vector2(-half_width, half_height),
            ),
        )

        self._planner = DecentralizedORCAPlanner(
            config_file="orca.toml",
            # obstacles=self.env.obstacles,
        )

        self._step = Step(
            env=self.env,
            robot_visible=False,
            planner=self._planner,
            rand=self.env_builder.rand,
        )

        self._local_avoidance = LocalAvoidancePlanner(
            agent=self.env.robot,
            arena_width=float(self.env.info.arena_width),
            arena_height=float(self.env.info.arena_height),
        )
        self._safe_point_finder = SafePointFinder(self._local_avoidance, self.env)

        self.visualizer = Visualizer()

    def step(self):
        return self._step.step()

    def _collision_predictor(self) -> SAT:
        assert self.env.robot.sensor is not None
        observation = self.env.robot.sensor.observe(self.env, robot_visible=False)
        return SAT(observation, self.env.robot, self.env.obstacles)

    def _update_mission(self, result: StepResult) -> None:
        if not self.mission.started:
            self.mission.reach_closest_corner()
            return

        predictor = self._collision_predictor()

        # Only enter avoidance if the robot's *current* planned path is
        # actually unsafe -- previously this ran unconditionally every tick.
        if predictor.checkIntrusionSAT():
            self.mission.avoid_crowd(
                predictor=predictor,
                safe_point_finder=self._safe_point_finder,
            )
            # Avoidance only selects the next temporary goal. This outer
            # runner performs the following simulation tick.
            return

        if result.robot_reached_goal:
            if not self.mission.sweeping:
                self.mission.sweeping = True
            self.mission.update_sweep()

    def _respawn_pedestrians(self, result: StepResult, count: int) -> None:
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
            self.visualizer.refresh(self.env, mission=self.mission)
            self._print_status()
            self._respawn_pedestrians(result, step_count)

        print("Simulation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Navcore task switchboard")
    parser.add_argument("mode", choices=("coverage", "goal", "waypoint"), nargs="?", default="coverage")
    parser.add_argument("--waypoint", nargs=2, type=float, metavar=("X", "Y"), default=(0.0, 0.0))
    args = parser.parse_args()
    if args.mode == "coverage":
        print(GlobalPlanner().run())
    elif args.mode == "goal":
        print(GoalPlanner().run().report())
    else:
        print(WaypointPlanner(Vector2(*args.waypoint)).run().report())

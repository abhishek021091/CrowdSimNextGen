from __future__ import annotations

import math
import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from unittest import expectedFailure
from unittest.mock import patch

from numpy.random import default_rng

from navcore.entities.agents.pedestrians import Pedestrian
from navcore.entities.agents.robot import Robot
from navcore.entities.components.goal import Goal
from navcore.entities.components.pose import Pose
from navcore.entities.components.state import FullState, ObservableState
from navcore.entities.components.velocity import Velocity
from navcore.environment.environment import Environment
from navcore.middleware.orca_middleware import DecentralizedORCAPlanner
from navcore.step.step import Step


class StrictPlanner:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def compute_velocities(
        self,
        self_id: int,
        self_state: FullState,
        observations: dict[int, ObservableState],
    ) -> tuple[Velocity, dict[int, Velocity]]:
        self.calls.append((self_id, self_state, dict(observations)))
        assert isinstance(self_id, int)
        assert isinstance(self_state, FullState)
        assert isinstance(observations, dict)
        return Velocity(0.25, 0.5), {
            k: Velocity(float(k) + 0.1, float(k) + 0.2)
            for k in observations
            if k != self_id
        }


class FakeBaseORCAPlanner:
    instances: list["FakeBaseORCAPlanner"] = []

    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.initialized_population: dict[object, FullState] | None = None
        self.initialized_obstacle_vertices: list[list[tuple[float, float]]] | None = (
            None
        )
        self.computed_population: dict[object, FullState] | None = None
        FakeBaseORCAPlanner.instances.append(self)


class BaseFixture(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls) -> None:
    #     cls._robot_sensors = deepcopy(Robot.config["sensors"])
    #     cls._ped_sensors = deepcopy(Pedestrian.config["sensors"])

    #     Robot.config["sensors"].setdefault(
    #         "range", Robot.config["sensors"].get("sensor_range", 10.0)
    #     )
    #     Pedestrian.config["sensors"].setdefault(
    #         "range", Pedestrian.config["sensors"].get("sensor_range", 10.0)
    #     )

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     # Robot.config["sensors"] = cls._robot_sensors
    #     # Pedestrian.config["sensors"] = cls._ped_sensors

    def _make_robot(
        self,
        px: float = 0.0,
        py: float = 0.0,
        theta: float = 0.0,
        gx: float = 3.0,
        gy: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> Robot:
        robot = Robot()
        robot.set_state(
            Pose(px, py, theta),
            Goal(gx, gy),
            robot.v_pref,
            robot.radius,
            Velocity(vx, vy),
        )
        return robot

    def _make_pedestrian(
        self,
        pid: int = 1,
        px: float = 1.0,
        py: float = 0.0,
        theta: float = 0.0,
        gx: float = 5.0,
        gy: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        seed: int = 7,
    ) -> Pedestrian:
        ped = Pedestrian(default_rng(seed))
        ped.set_id(pid)
        ped.set_state(
            Pose(px, py, theta),
            Goal(gx, gy),
            ped.v_pref,
            ped.radius,
            Velocity(vx, vy),
        )
        return ped

    def _make_env(
        self, robot: Robot, crowd: list[Pedestrian] | None = None
    ) -> Environment:
        crowd_map = {ped.id: ped for ped in (crowd or [])}
        return Environment(obstacles={}, crowd=crowd_map, groups={}, robot=robot)


class ORCATests(unittest.TestCase):
    def test_as_full_state_builds_synthetic_goal_from_current_velocity(self) -> None:
        observed = ObservableState(
            pose=Pose(2.0, 3.0, 0.5),
            velocity=Velocity(0.4, -0.2),
            radius=0.6,
        )

        full_state = DecentralizedORCAPlanner._as_full_state(observed)

        self.assertEqual(full_state.pose.px, 2.0)
        self.assertEqual(full_state.pose.py, 3.0)
        self.assertEqual(full_state.pose.theta, 0.5)
        self.assertEqual(full_state.goal.gx, 2.4)
        self.assertEqual(full_state.goal.gy, 2.8)
        self.assertEqual(full_state.velocity.vx, 0.4)
        self.assertEqual(full_state.velocity.vy, -0.2)
        self.assertEqual(full_state.radius, 0.6)
        self.assertAlmostEqual(full_state.preferred_speed, math.hypot(0.4, -0.2))

    def test_as_full_state_uses_stationary_self_goal(self) -> None:
        observed = ObservableState(
            pose=Pose(-1.0, 4.5, 1.25),
            velocity=Velocity(0.0, 0.0),
            radius=0.3,
        )

        full_state = DecentralizedORCAPlanner._as_full_state(observed)

        self.assertEqual(full_state.goal.gx, -1.0)
        self.assertEqual(full_state.goal.gy, 4.5)
        self.assertEqual(full_state.preferred_speed, 0.0)

    def test_compute_velocities_wraps_base_planner_and_splits_outputs(self) -> None:
        FakeBaseORCAPlanner.instances.clear()
        planner = DecentralizedORCAPlanner("orca.toml", obstacles=[])

        self_state = FullState(
            pose=Pose(0.0, 0.0, 0.0),
            goal=Goal(4.0, 0.0),
            velocity=Velocity(0.2, 0.3),
            radius=0.5,
            preferred_speed=1.0,
        )
        observations = {
            7: ObservableState(Pose(1.0, 2.0, 0.0), Velocity(1.2, 1.7), 0.4),
            9: ObservableState(Pose(-2.0, 1.0, 0.0), Velocity(1.0, 1.7), 0.4),
        }

        with patch(
            "navcore.middleware.orca_middleware.BaseORCAPlanner", FakeBaseORCAPlanner
        ):
            self_velocity, other_velocities = planner.compute_velocities(
                self_id=7,
                self_state=self_state,
                observations=observations,
            )

        self.assertEqual((self_velocity.vx, self_velocity.vy), (0.8, 0.9))
        self.assertIn(9, other_velocities)
        self.assertNotIn(7, other_velocities)
        self.assertEqual((other_velocities[9].vx, other_velocities[9].vy), (1.0, 1.7))

        self.assertEqual(len(FakeBaseORCAPlanner.instances), 1)
        fake = FakeBaseORCAPlanner.instances[0]
        assert fake.initialized_population is not None
        assert fake.computed_population is not None

        self.assertEqual(set(fake.initialized_population.keys()), {"__self__", 9})
        self.assertEqual(set(fake.computed_population.keys()), {"__self__", 9})

        neighbor_state = fake.initialized_population[9]
        self.assertEqual((neighbor_state.goal.gx, neighbor_state.goal.gy), (-1.0, 2.7))
        # self.assertAlmostEqual(
        #    neighbor_state.preferred_speed,
        #    math.hypot(1.2, 1.7),
        # )

        self.assertEqual(fake.initialized_population["__self__"], self_state)


class StepTests(BaseFixture):
    def test_validate_rejects_missing_state(self) -> None:
        robot = self._make_robot()
        crowd = [self._make_pedestrian(pid=1)]
        env = self._make_env(robot, crowd)
        env.robot.pose = None

        step = Step(robot, StrictPlanner(), env, robot_visible=True)

        with self.assertRaises(RuntimeError):
            step._validate()

    def test_advance_agent_updates_position_and_heading(self) -> None:
        robot = self._make_robot(px=0.0, py=0.0, theta=0.0, vx=2.0, vy=1.0)
        env = self._make_env(robot)
        step = Step(robot, StrictPlanner(), env, robot_visible=True)

        start_x, start_y = robot.pose.px, robot.pose.py
        step._advance_agent(robot)

        self.assertAlmostEqual(robot.pose.px, start_x + 2.0 * Step.dt)
        self.assertAlmostEqual(robot.pose.py, start_y + 1.0 * Step.dt)
        self.assertAlmostEqual(robot.pose.theta, math.atan2(1.0, 2.0))

    def test_set_velocities_applies_robot_and_crowd(self) -> None:
        robot = self._make_robot(vx=0.0, vy=0.0)
        ped1 = self._make_pedestrian(pid=1, vx=0.0, vy=0.0)
        ped2 = self._make_pedestrian(pid=2, vx=0.0, vy=0.0)
        env = self._make_env(robot, [ped1, ped2])
        step = Step(robot, StrictPlanner(), env, robot_visible=True)

        step.set_velocities(
            Velocity(0.6, -0.2),
            {1: Velocity(0.4, 0.5), 2: Velocity(-0.1, 0.0), -1: Velocity(9.9, 9.9)},
        )

        self.assertEqual((env.robot.velocity.vx, env.robot.velocity.vy), (0.6, -0.2))
        self.assertEqual(
            (env.crowd[1].velocity.vx, env.crowd[1].velocity.vy), (0.4, 0.5)
        )
        self.assertEqual(
            (env.crowd[2].velocity.vx, env.crowd[2].velocity.vy), (-0.1, 0.0)
        )

    def test_set_velocities_rejects_unknown_pedestrian(self) -> None:
        robot = self._make_robot()
        ped1 = self._make_pedestrian(pid=1)
        env = self._make_env(robot, [ped1])
        step = Step(robot, StrictPlanner(), env, robot_visible=True)

        with self.assertRaises(KeyError):
            step.set_velocities(Velocity(0.0, 0.0), {99: Velocity(1.0, 1.0)})

    def test_apply_robot_velocity_returns_and_sets_robot_velocity(self) -> None:
        robot = self._make_robot(vx=0.0, vy=0.0)
        env = self._make_env(robot)
        step = Step(robot, StrictPlanner(), env, robot_visible=True)

        result = step._apply_robot_velocity(robot, {-1: Velocity(0.3, 0.7)})

        self.assertEqual((result.vx, result.vy), (0.3, 0.7))
        self.assertEqual((robot.velocity.vx, robot.velocity.vy), (0.3, 0.7))

    def test_apply_crowd_velocities_updates_all_agents(self) -> None:
        robot = self._make_robot()
        ped1 = self._make_pedestrian(pid=1)
        ped2 = self._make_pedestrian(pid=2)
        env = self._make_env(robot, [ped1, ped2])
        step = Step(robot, StrictPlanner(), env, robot_visible=True)

        result = step._apply_crowd_velocities(
            [ped1, ped2],
            {1: Velocity(0.1, 0.2), 2: Velocity(-0.4, 0.9)},
        )

        self.assertEqual((ped1.velocity.vx, ped1.velocity.vy), (0.1, 0.2))
        self.assertEqual((ped2.velocity.vx, ped2.velocity.vy), (-0.4, 0.9))
        self.assertEqual(set(result.keys()), {1, 2})

    def test_apply_crowd_velocities_rejects_missing_key(self) -> None:
        robot = self._make_robot()
        ped1 = self._make_pedestrian(pid=1)
        env = self._make_env(robot, [ped1])
        step = Step(robot, StrictPlanner(), env, robot_visible=True)

        with self.assertRaises(KeyError):
            step._apply_crowd_velocities([ped1], {})

    @expectedFailure
    def test_compute_velocities_should_pass_self_id_to_planner(self) -> None:
        robot = self._make_robot()
        ped1 = self._make_pedestrian(pid=1)
        env = self._make_env(robot, [ped1])

        step = Step(robot, StrictPlanner(), env, robot_visible=True)
        step._compute_velocities()

    @expectedFailure
    def test_step_should_apply_returned_velocities_before_advancing(self) -> None:
        robot = self._make_robot(px=0.0, py=0.0, vx=0.0, vy=0.0)
        ped1 = self._make_pedestrian(pid=1, px=1.0, py=0.0, vx=0.0, vy=0.0)
        env = self._make_env(robot, [ped1])

        planner = StrictPlanner()
        step = Step(robot, planner, env, robot_visible=True)

        def fake_compute_velocities() -> tuple[Velocity, dict[int, Velocity]]:
            return Velocity(1.0, 0.0), {1: Velocity(0.5, 0.0)}

        with patch.object(step, "_compute_velocities", fake_compute_velocities):
            step.step()

        self.assertAlmostEqual(robot.pose.px, Step.dt)
        self.assertAlmostEqual(ped1.pose.px, 1.0 + 0.5 * Step.dt)


if __name__ == "__main__":
    unittest.main(verbosity=2)

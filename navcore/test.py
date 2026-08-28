from __future__ import annotations

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.entities.components.velocity import Velocity
from navcore.policies.robot_orca_local_planner import RobotORCAPlanner
from navcore.step.step import Step


class DummyPlanner:
    """Deterministic planner used only for testing."""

    def compute_velocities(
        self,
        robot_observable_state,
        crowd_observable_states,
    ) -> tuple[Velocity, dict[int, Velocity]]:
        robot_velocity = Velocity(1.0, 0.0)

        crowd_velocities = {
            ped_id: Velocity(0.5, 0.0) for ped_id in crowd_observable_states
        }

        return robot_velocity, crowd_velocities


def test_step_pipeline() -> None:
    """Smoke test for the complete Step pipeline."""

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    env = EnvironmentBuilder().build_environment()

    # ------------------------------------------------------------------
    # Environment sanity checks
    # ------------------------------------------------------------------
    assert env.robot is not None
    assert len(env.crowd) > 0
    assert len(env.obstacles) > 0

    assert env.robot.pose is not None
    assert env.robot.goal is not None

    for ped in env.crowd.values():
        assert ped.pose is not None
        assert ped.goal is not None

    # ------------------------------------------------------------------
    # Construct Step
    # ------------------------------------------------------------------
    planner = RobotORCAPlanner()  # Use the actual planner for testing
    planner.initialize(env)
    print("✓ Environment and planner initialized successfully.")
    print(planner.robot_id)
    print(planner.pedestrian_ids)
    step = Step(
        planner=RobotORCAPlanner(),  # Use the actual planner for testing
        robot_visible=True,
    )
    planner.initialize(env)

    # ------------------------------------------------------------------
    # Save initial positions
    # ------------------------------------------------------------------
    robot_before = (
        env.robot.pose.px,
        env.robot.pose.py,
    )

    crowd_before = {
        ped.id: (
            ped.pose.px,
            ped.pose.py,
        )
        for ped in env.crowd.values()
    }
    print("Initial positions saved.")
    print(f"Robot initial position: {robot_before}")
    print(f"Crowd initial positions: {crowd_before}")
    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------
    robot_velocity, crowd_velocities = step.compute_velocities(env)

    assert isinstance(robot_velocity, Velocity)
    assert len(crowd_velocities) == len(env.crowd) + 1

    # ------------------------------------------------------------------
    # Apply velocities
    # ------------------------------------------------------------------
    step.set_velocities(
        env,
        robot_velocity,
        crowd_velocities,
    )

    assert env.robot.velocity is not None

    for ped in env.crowd.values():
        assert ped.velocity is not None

    # ------------------------------------------------------------------
    # Advance simulation
    # ------------------------------------------------------------------
    step.step(env)

    # ------------------------------------------------------------------
    # Robot moved
    # ------------------------------------------------------------------
    robot_after = (
        env.robot.pose.px,
        env.robot.pose.py,
    )

    assert robot_after != robot_before

    # ------------------------------------------------------------------
    # Crowd moved
    # ------------------------------------------------------------------
    for ped in env.crowd.values():
        before = crowd_before[ped.id]
        after = (
            ped.pose.px,
            ped.pose.py,
        )

        assert after != before
    print("✓ Robot and crowd moved after step.")
    print(f"Robot position after step: {robot_after}")
    print(
        f"Crowd positions after step: { {ped.id: (ped.pose.px, ped.pose.py) for ped in env.crowd.values()} }"
    )

    print("✓ Step pipeline test passed.")


if __name__ == "__main__":
    test_step_pipeline()

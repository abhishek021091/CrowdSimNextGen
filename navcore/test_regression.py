"""Regression / wiring smoke tests for CrowdSimNextGen.

Drop this into the project root as ``navcore/test_regression_checks.py`` and
run:

    python -m navcore.test_regression_checks

Why this exists (separate from test_sweep.py / test_global_planner.py):
    Those two are live, visual, full-pipeline runs against real ORCA +
    rvo2 -- great for "does the whole thing work end to end," bad for
    isolating *which* piece is wrong when something doesn't. This file
    checks specific, previously-flagged wiring points in isolation, using
    a stub VelocityPlanner wherever the real ORCA/rvo2 pipeline isn't the
    thing under test. That means most checks here run even in
    environments where the rvo2 C extension isn't built.

Design of the harness:
    - Each check is independent: one check's exception is caught and
      recorded as FAIL without stopping the rest, so one broken subsystem
      doesn't hide unrelated results.
    - A check that specifically needs the real ORCA/rvo2 pipeline raises
      `Skipped` if rvo2 isn't importable, and is reported as SKIP, not
      FAIL -- an environment without the compiled extension isn't a
      regression.
    - Checks assert on observable *behavior* (what target/goal actually
      reached the planner, how many times a collaborator was called),
      not on internal call graphs, so they stay valid across refactors
      that preserve behavior.

Paste the full printed output back for triage. A FAIL's message or
traceback is the part that matters; PASS/SKIP lines are just bookkeeping.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from navcore.builder.environment_builder import EnvironmentBuilder
from navcore.entities.components.geometry.vector2 import Vector2
from navcore.entities.components.state import FullState, ObservableState
from navcore.entities.components.velocity import Velocity
from navcore.gym_wrapper.goal_reaching_task import GoalReachingTask
from navcore.gym_wrapper.rl_missions import RLWaypointMission
from navcore.missions.sweeping import SweepingMission
from navcore.step.step import Step

# ---------------------------------------------------------------------------
# Test harness (no pytest dependency, so this runs anywhere the project
# itself runs).
# ---------------------------------------------------------------------------


class Skipped(Exception):
    """Raise inside a check to mark it SKIP rather than FAIL."""


@dataclass
class _Result:
    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    detail: str = ""


_CHECKS: list[tuple[str, Callable[[], None]]] = []


def check(name: str):
    """Register `fn` as a named check, run later by `main()`."""

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        _CHECKS.append((name, fn))
        return fn

    return decorator


def _rvo2_available() -> bool:
    try:
        import rvo2  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stub planner: satisfies the VelocityPlanner protocol Step depends on
# without touching rvo2. Every call is recorded so checks can assert on
# exactly what target/goal Step handed to the planner for a given tick.
# ---------------------------------------------------------------------------


@dataclass
class _RecordedCall:
    self_id: int
    self_state: FullState
    observations: Mapping[int, ObservableState]


class RecordingStubPlanner:
    """Returns zero velocity for everyone; records every call it receives."""

    def __init__(self) -> None:
        self.calls: list[_RecordedCall] = []

    def compute_velocities(self, self_id, self_state, observations):
        self.calls.append(_RecordedCall(self_id, self_state, observations))
        return Velocity(0.0, 0.0), {}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@check("1. Step routes robot_mission's target, not robot.goal, to the planner")
def _test_robot_mission_wiring():
    env = EnvironmentBuilder(include_static_obstacles=False).build_environment()
    assert env.robot.pose is not None and env.robot.goal is not None

    # Deliberately far from the robot's real goal so a wiring bug (Step
    # silently falling back to agent.goal) is unmistakable in the diff.
    waypoint = Vector2(env.robot.goal.gx + 1000.0, env.robot.goal.gy + 1000.0)
    mission = RLWaypointMission(initial_target=waypoint)

    planner = RecordingStubPlanner()
    step_driver = Step(
        planner=planner, env=env, robot_visible=False, robot_mission=mission
    )
    step_driver.step()

    robot_calls = [c for c in planner.calls if c.self_id == Step.ROBOT_KEY]
    assert robot_calls, "planner.compute_velocities was never called for the robot"

    used_goal = robot_calls[0].self_state.goal
    assert (used_goal.gx, used_goal.gy) == (waypoint.x, waypoint.y), (
        f"Step planned toward {(used_goal.gx, used_goal.gy)}, expected the "
        f"mission's waypoint {(waypoint.x, waypoint.y)} -- robot_mission is "
        f"not being consulted."
    )
    assert (env.robot.goal.gx, env.robot.goal.gy) != (waypoint.x, waypoint.y), (
        "robot.goal was overwritten by the mission target -- Step must keep "
        "agent.goal untouched (see Step's own module docstring)."
    )


@check("2. Step routes a per-pedestrian crowd_mission's target, not ped.goal")
def _test_crowd_mission_wiring():
    env = EnvironmentBuilder(include_static_obstacles=False).build_environment()
    assert env.crowd, "env.toml's [pedestrians].num_pedestrians must be > 0."

    # Avoid a pedestrian id divisible by 10: Step._compute_crowd_velocities
    # has a ~20% chance of skipping the planner entirely for those ids
    # (simulated "frozen" pedestrians), which would make this check flaky
    # for an unrelated reason.
    target_ped_id = next(pid for pid in env.crowd if pid % 10 != 0)
    ped = env.crowd[target_ped_id]
    assert ped.goal is not None

    fixed_target = Vector2(ped.goal.gx + 500.0, ped.goal.gy + 500.0)

    class _FixedTargetMission:
        def get_target(self, agent, neighbors):
            return fixed_target

    planner = RecordingStubPlanner()
    step_driver = Step(
        planner=planner,
        env=env,
        robot_visible=False,
        crowd_missions={target_ped_id: _FixedTargetMission()},
    )
    step_driver.step()

    ped_calls = [c for c in planner.calls if c.self_id == target_ped_id]
    assert ped_calls, (
        f"planner.compute_velocities was never called for pedestrian {target_ped_id}"
    )

    used_goal = ped_calls[0].self_state.goal
    assert (used_goal.gx, used_goal.gy) == (fixed_target.x, fixed_target.y), (
        f"Step planned pedestrian {target_ped_id} toward "
        f"{(used_goal.gx, used_goal.gy)}, expected "
        f"{(fixed_target.x, fixed_target.y)} -- crowd_missions is not being "
        f"consulted."
    )


@check("3. EnvironmentBuilder.reset() gives each episode a fresh EnvironmentInfo")
def _test_environment_info_reset():
    builder = EnvironmentBuilder(include_static_obstacles=False)
    env1 = builder.build_environment()
    env1.info.collision_counter = 7  # simulate a prior episode with collisions

    env2 = builder.reset(random_seed=12345)

    assert env2.info.collision_counter == 0, (
        f"New episode started with collision_counter="
        f"{env2.info.collision_counter}, expected 0 -- EnvironmentInfo is "
        f"leaking state across episodes."
    )
    assert env2.info is not env1.info, (
        "reset() reused the previous episode's EnvironmentInfo instance."
    )


@check("4. Robot goal-reach tolerance is consistent between Step and env.toml")
def _test_goal_reach_tolerance_consistency():
    env = EnvironmentBuilder(include_static_obstacles=False).build_environment()
    assert env.robot.pose is not None and env.robot.goal is not None

    configured_tolerance = env.info.goal_reach_tolerance

    # Place the robot at a distance strictly between the configured
    # tolerance (env.toml's [tolerance].goal_reach) and 0.5 -- the value
    # seen hardcoded in Step._compute_velocities. If the two disagree on
    # what "reached" means, this distance exposes it without needing to
    # read Step's source for the literal.
    probe_distance = (configured_tolerance + 0.5) / 2.0
    env.robot.pose.px = env.robot.goal.gx + probe_distance
    env.robot.pose.py = env.robot.goal.gy

    planner = RecordingStubPlanner()
    step_driver = Step(planner=planner, env=env, robot_visible=False)
    result = step_driver.step()

    tolerance_says_reached = probe_distance <= configured_tolerance
    assert result.robot_reached_goal == tolerance_says_reached, (
        f"At distance {probe_distance:.3f} from goal: "
        f"Step.robot_reached_goal={result.robot_reached_goal}, but "
        f"env.info.goal_reach_tolerance={configured_tolerance} implies "
        f"{tolerance_says_reached}. Step and the configured tolerance "
        f"disagree on what 'reached' means -- find Step's hardcoded "
        f"threshold(s) and replace them with env.info.goal_reach_tolerance."
    )


@check(
    "5. SweepingMission.avoid_crowd() is tick-resumable, not an internal blocking loop"
)
def _test_avoid_crowd_is_tick_resumable():
    env = EnvironmentBuilder(include_static_obstacles=False).build_environment()
    half_w = float(env.info.arena_width) / 2.0
    half_h = float(env.info.arena_height) / 2.0
    mission = SweepingMission(
        env,
        cell_vertices=(
            Vector2(-half_w, -half_h),
            Vector2(half_w, -half_h),
            Vector2(half_w, half_h),
            Vector2(-half_w, half_h),
        ),
    )
    mission.reach_closest_corner()

    call_counts = {"intrusion": 0, "safe_point": 0}

    class _AlwaysBlockedPredictor:
        def checkIntrusionSAT(self, *args, **kwargs):
            call_counts["intrusion"] += 1
            return True

    class _FixedSafePointFinder:
        def find_safe_point(self, pose):
            call_counts["safe_point"] += 1
            return (pose.px + 0.1, pose.py + 0.1)

    predictor = _AlwaysBlockedPredictor()
    finder = _FixedSafePointFinder()

    start = time.perf_counter()
    mission.avoid_crowd(predictor=predictor, safe_point_finder=finder)
    elapsed = time.perf_counter() - start

    assert mission.avoiding_obstacle is True, (
        "avoid_crowd() did not set avoiding_obstacle."
    )
    assert call_counts["safe_point"] == 1, (
        f"find_safe_point was called {call_counts['safe_point']} times "
        f"during a single avoid_crowd() call -- expected exactly 1. A "
        f"count > 1 means an internal loop is still driving multiple "
        f"ticks per call, i.e. the pre-refactor blocking behavior."
    )
    assert elapsed < 0.05, (
        f"avoid_crowd() took {elapsed * 1000:.1f} ms for one call against "
        f"an always-blocked predictor -- suspiciously slow for O(1) work; "
        f"check for a hidden retry loop."
    )


@check("6. No leftover duplicate GoalReachingTask in rl_missions.py")
def _test_no_duplicate_goal_reaching_task():
    import navcore.gym_wrapper.rl_missions as rl_missions

    assert not hasattr(rl_missions, "GoalReachingTask"), (
        "rl_missions.py defines its own GoalReachingTask, duplicating "
        "navcore.gym_wrapper.goal_reaching_task.GoalReachingTask (the one "
        "actually imported by CrowdSimEnv/task_planners). Delete the copy "
        "in rl_missions.py."
    )


@check("7. No leftover blocking-loop avoid_crowd variant on SweepingMission")
def _test_no_dead_blocking_avoid_crowd():
    suspicious = [
        name
        for name in dir(SweepingMission)
        if "avoid" in name.lower() and name != "avoid_crowd"
    ]
    assert not suspicious, (
        f"SweepingMission still defines {suspicious} alongside avoid_crowd "
        f"-- looks like a leftover pre-refactor method that should be "
        f"deleted."
    )


@check(
    "8. CrowdSimEnv WAYPOINT action mode actually changes robot velocity (requires rvo2)"
)
def _test_crowd_sim_env_waypoint_mode():
    if not _rvo2_available():
        raise Skipped("rvo2 is not importable in this environment.")

    import numpy as np

    from navcore.gym_wrapper.crowd_sim_env import (
        ActionMode,
        CrowdSimEnv,
        CrowdSimEnvConfig,
    )

    env = CrowdSimEnv(
        GoalReachingTask(),
        CrowdSimEnvConfig(
            action_mode=ActionMode.WAYPOINT, include_static_obstacles=False
        ),
    )
    obs, _ = env.reset(seed=0)
    assert "robot" in obs and "neighbors" in obs

    # Two clearly different waypoints (both valid within the [-1, 1]
    # normalized action space) should produce clearly different robot
    # velocities if WAYPOINT wiring actually reaches the planner.
    action_a = np.array([0.9, 0.9], dtype=np.float32)
    action_b = np.array([-0.9, -0.9], dtype=np.float32)

    env.step(action_a)
    velocity_a = Vector2(env.env.robot.velocity.vx, env.env.robot.velocity.vy)

    env.reset(seed=0)
    env.step(action_b)
    velocity_b = Vector2(env.env.robot.velocity.vx, env.env.robot.velocity.vy)

    assert velocity_a.distance_to(velocity_b) > 1e-6, (
        "Opposite waypoint actions produced (near-)identical robot "
        "velocities -- WAYPOINT mode's target does not appear to be "
        "reaching the planner."
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    results: list[_Result] = []

    for name, fn in _CHECKS:
        header = f"--- {name} "
        print(header + "-" * max(1, 78 - len(header)))
        try:
            fn()
        except Skipped as exc:
            results.append(_Result(name, "SKIP", str(exc)))
            print(f"SKIP: {exc}")
        except AssertionError as exc:
            results.append(_Result(name, "FAIL", str(exc)))
            print(f"FAIL: {exc}")
        except Exception:
            tb = traceback.format_exc()
            results.append(_Result(name, "FAIL", tb))
            print(f"FAIL (unexpected exception):\n{tb}")
        else:
            results.append(_Result(name, "PASS"))
            print("PASS")
        print()

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")

    print("=" * 78)
    print(f"SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 78)
    for r in results:
        print(f"  [{r.status}] {r.name}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

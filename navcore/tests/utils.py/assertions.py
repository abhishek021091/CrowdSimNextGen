# tests/utils/assertions.py
"""Reusable assertions over an ``EpisodeResult``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.utils.metrics import Thresholds
from tests.utils.runner import EpisodeResult


def assert_navigation_ok(
    result: EpisodeResult, thresholds: Thresholds | None = None
) -> None:
    """Checks every test must pass, regardless of scenario."""
    th = thresholds or result.thresholds
    m = result.metrics
    ctx = result.describe()

    assert not m.collision, f"Robot collided.\n{ctx}"
    assert not m.left_map, f"Robot left the map.\n{ctx}"
    assert not m.deadlock, f"Robot deadlocked.\n{ctx}"
    assert not m.timed_out, f"Episode timed out.\n{ctx}"
    assert m.goal_reached, f"Robot did not reach the goal.\n{ctx}"

    assert m.oscillation_count <= th.max_oscillations, (
        f"Excessive oscillation ({m.oscillation_count} > {th.max_oscillations}).\n{ctx}"
    )
    assert m.max_speed <= result.v_max + th.speed_tolerance, (
        f"Velocity limit exceeded ({m.max_speed:.3f} > {result.v_max}).\n{ctx}"
    )
    assert m.min_human_distance >= th.safe_human_distance, (
        f"Human clearance {m.min_human_distance:.3f} < {th.safe_human_distance}.\n{ctx}"
    )
    assert m.min_obstacle_distance >= th.safe_obstacle_distance, (
        f"Obstacle clearance {m.min_obstacle_distance:.3f} < "
        f"{th.safe_obstacle_distance}.\n{ctx}"
    )
    budget = result.frame.length / result.v_max * th.time_budget_factor
    budget += th.time_budget_offset_s
    assert m.time_to_goal is not None and m.time_to_goal <= budget, (
        f"Too slow: {m.time_to_goal}s > budget {budget:.1f}s.\n{ctx}"
    )
    assert m.path_efficiency >= th.min_path_efficiency, (
        f"Path efficiency {m.path_efficiency:.3f} < {th.min_path_efficiency}.\n{ctx}"
    )


def assert_sidestepped(result: EpisodeResult, min_deviation: float = 0.6) -> None:
    """The robot left the straight line by at least ``min_deviation`` metres.

    Pedestrians do not react to the robot, so a head-on human is only avoided
    by a lateral manoeuvre of at least the sum of both radii.
    """
    deviation = result.metrics.max_lateral_deviation
    assert deviation >= min_deviation, (
        f"Robot never sidestepped (max lateral deviation {deviation:.2f} m < "
        f"{min_deviation} m).\n{result.describe()}"
    )


def assert_detour_bounded(result: EpisodeResult, max_deviation: float) -> None:
    deviation = result.metrics.max_lateral_deviation
    assert deviation <= max_deviation, (
        f"Detour too large ({deviation:.2f} m > {max_deviation} m).\n{result.describe()}"
    )


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    step: int
    order: str  # "robot_first" | "human_first"
    separation: float


def crossing_event(result: EpisodeResult, human_index: int) -> CrossingEvent | None:
    """First tick at which the human crosses the nominal robot path line."""
    frame = result.frame
    robot = result.trajectory.robot_array()
    humans = result.trajectory.human_array()
    human = humans[:, human_index]
    lateral = frame.lateral(human)
    side = np.sign(lateral[0])
    if side == 0:
        return None
    crossed = np.nonzero(np.sign(lateral) == -side)[0]
    if crossed.size == 0:
        return None
    t = int(crossed[0])
    robot_lon = float(frame.longitudinal(robot[t]))
    human_lon = float(frame.longitudinal(human[t]))
    separation = (
        float(np.hypot(*(robot[t] - human[t])))
        - result.trajectory.robot_radius
        - float(result.trajectory.human_radii[human_index])
    )
    return CrossingEvent(
        step=t,
        order="robot_first" if robot_lon > human_lon else "human_first",
        separation=separation,
    )


def assert_crosses_safely(result: EpisodeResult, human_index: int = 0) -> None:
    """Robot either passes before the human (``robot_first``) or yields and
    passes behind it (``human_first``); either way with safe separation."""
    th = result.thresholds
    event = crossing_event(result, human_index)
    assert event is not None, (
        f"Human {human_index} never crossed the robot path.\n{result.describe()}"
    )
    assert event.separation >= th.safe_human_distance, (
        f"Unsafe crossing ({event.order}), separation {event.separation:.3f} m.\n"
        f"{result.describe()}"
    )


def overtake_status(result: EpisodeResult, human_index: int = 0) -> str:
    """Relative longitudinal ordering over the episode."""
    frame = result.frame
    robot_lon = frame.longitudinal(result.trajectory.robot_array())
    human_lon = frame.longitudinal(result.trajectory.human_array()[:, human_index])
    relative = robot_lon - human_lon
    if relative[0] > 0:
        return "human_overtook" if np.any(relative < 0) else "robot_stayed_ahead"
    return "robot_overtook" if np.any(relative > 0) else "followed"


def assert_follows_or_overtakes(result: EpisodeResult, human_index: int = 0) -> None:
    status = overtake_status(result, human_index)
    assert status in ("robot_overtook", "followed"), (
        f"Unexpected ordering '{status}' for a human ahead of the robot.\n"
        f"{result.describe()}"
    )


def assert_humans_overtake(
    result: EpisodeResult, fast_index: int, slow_index: int
) -> None:
    """Scenario premise: the fast human actually passed the slow one."""
    frame = result.frame
    humans = result.trajectory.human_array()
    fast = frame.longitudinal(humans[:, fast_index])
    slow = frame.longitudinal(humans[:, slow_index])
    assert np.any(fast > slow), (
        f"Humans {fast_index} never overtook {slow_index}.\n{result.describe()}"
    )

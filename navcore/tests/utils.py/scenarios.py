# tests/utils/scenarios.py
"""Catalog of scenario factories (pure data; no simulation is run here)."""

from __future__ import annotations

import math

import numpy as np

from tests.utils.scenario_builder import (
    ROBOT_GOAL,
    ROBOT_START,
    GroupSpec,
    HumanSpec,
    Scenario,
)


def _human(start, goal, speed=1.0) -> HumanSpec:
    return HumanSpec(start=start, goal=goal, speed=speed)


# -- single human -------------------------------------------------------------


def same_direction(human_speed: float = 0.5) -> Scenario:
    return Scenario(
        name=f"same_direction_v{human_speed}",
        humans=(_human((0.0, -4.0), (0.0, 12.0), human_speed),),
    )


def head_on(speed: float = 1.0, lateral_offset: float = 0.0) -> Scenario:
    return Scenario(
        name=f"head_on_v{speed}_dx{lateral_offset}",
        humans=(_human((lateral_offset, 7.0), (lateral_offset, -12.0), speed),),
    )


def cross_left(speed: float = 1.0) -> Scenario:
    """Human walks across the robot path from the robot's left (-x)."""
    return Scenario(
        name=f"cross_left_v{speed}",
        humans=(_human((-3.0, -5.0), (6.0, -5.0), speed),),
    )


def cross_right(speed: float = 1.0) -> Scenario:
    return Scenario(
        name=f"cross_right_v{speed}",
        humans=(_human((3.0, -5.0), (-6.0, -5.0), speed),),
    )


def from_behind(speed: float = 1.3) -> Scenario:
    return Scenario(
        name=f"from_behind_v{speed}",
        humans=(_human((0.0, -11.0), (0.0, 12.0), speed),),
    )


# -- multiple humans ------------------------------------------------------------


def two_crossing() -> Scenario:
    return Scenario(
        name="two_crossing",
        humans=(
            _human((-3.0, -5.0), (6.0, -5.0), 1.0),
            _human((3.0, -5.6), (-6.0, -5.6), 1.0),
        ),
    )


def two_together() -> Scenario:
    return Scenario(
        name="two_together",
        humans=(
            _human((-0.45, 7.0), (-0.45, -12.0), 1.0),
            _human((0.45, 7.0), (0.45, -12.0), 1.0),
        ),
    )


def group_walk() -> Scenario:
    return Scenario(
        name="group_walk",
        humans=(
            _human((-3.0, -5.0), (6.0, -5.0), 1.0),  # leader
            _human((-3.9, -4.4), (6.0, -5.0), 1.0),
            _human((-3.9, -5.6), (6.0, -5.0), 1.0),
        ),
        groups=(GroupSpec(member_indices=(0, 1, 2), leader_index=0),),
    )


def random_crowd(
    num_humans: int,
    seed: int,
    *,
    x_range=(-4.5, 4.5),
    y_range=(-6.0, 6.0),
    min_spacing: float = 0.9,
    keepout: float = 2.0,
    speed_range=(0.8, 1.2),
) -> tuple[HumanSpec, ...]:
    """Seeded crowd flowing through the middle of the arena.

    Humans keep ``keepout`` metres from the robot start/goal; every human
    heads for the far side (y beyond +-10) so the flow crosses the robot's path.
    """
    rng = np.random.default_rng(1000 + seed)
    anchors = (ROBOT_START, ROBOT_GOAL)
    starts: list[tuple[float, float]] = []
    attempts = 0
    while len(starts) < num_humans:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("Could not place the requested crowd.")
        p = (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))
        if any(math.hypot(p[0] - a[0], p[1] - a[1]) < keepout for a in anchors):
            continue
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) < min_spacing for q in starts):
            continue
        starts.append(p)

    humans = []
    for start in starts:
        gx = float(rng.uniform(-4.0, 4.0))
        if abs(gx) < 1.5:
            gx = math.copysign(1.5, gx if gx != 0.0 else 1.0)
        gy = float(rng.uniform(10.0, 13.0)) * (-1.0 if start[1] > 0 else 1.0)
        humans.append(
            HumanSpec(
                start=start, goal=(gx, gy), speed=float(rng.uniform(*speed_range))
            )
        )
    return tuple(humans)


def sparse_crowd(seed: int = 0, num_humans: int = 4) -> Scenario:
    return Scenario(
        name=f"sparse_crowd_seed{seed}",
        humans=random_crowd(num_humans, seed),
        seed=seed,
    )


def dense_crowd(seed: int = 0, num_humans: int = 14) -> Scenario:
    return Scenario(
        name=f"dense_crowd_seed{seed}",
        humans=random_crowd(num_humans, seed),
        seed=seed,
    )


def multi_direction() -> Scenario:
    return Scenario(
        name="multi_direction",
        humans=(
            _human((-3.5, -4.0), (6.0, -4.0), 1.0),  # from the left
            _human((3.5, -1.0), (-6.0, -1.0), 1.0),  # from the right
            _human((0.5, 6.0), (0.5, -12.0), 0.9),  # head-on
            _human((-0.5, -11.5), (-0.5, 12.0), 1.2),  # from behind
        ),
    )


def parallel_pedestrians() -> Scenario:
    """Two lanes of slow humans flanking the robot's lane, walking its way."""
    return Scenario(
        name="parallel_pedestrians",
        humans=(
            _human((-1.4, -4.0), (-1.4, 12.0), 0.7),
            _human((1.4, -5.0), (1.4, 12.0), 0.7),
            _human((-1.4, -1.0), (-1.4, 12.0), 0.7),
            _human((1.4, -2.0), (1.4, 12.0), 0.7),
        ),
    )


def overtaking_humans() -> Scenario:
    """A fast human (index 1) overtakes a slow one (index 0) in the robot's way."""
    return Scenario(
        name="overtaking_humans",
        humans=(
            _human((-0.6, -3.0), (-0.6, 12.0), 0.6),
            _human((-0.6, -6.0), (-1.5, 13.0), 1.2),
        ),
    )


ALL_SCENARIOS: tuple[Scenario, ...] = (
    same_direction(),
    head_on(),
    cross_left(),
    cross_right(),
    from_behind(),
    two_crossing(),
    two_together(),
    group_walk(),
    sparse_crowd(0),
    sparse_crowd(1),
    sparse_crowd(2),
    dense_crowd(0),
    dense_crowd(1),
    dense_crowd(2),
    multi_direction(),
    parallel_pedestrians(),
    overtaking_humans(),
)

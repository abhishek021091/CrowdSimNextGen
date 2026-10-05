# tests/test_scenario_validity.py
"""Policy-independent checks on the scenario catalog itself."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString

from navcore.entities.obstacles.geometry_conversion import obstacle_to_shapely_polygon

from tests.utils.scenario_builder import (
    STRAIGHT_PATH_CLEARANCE,
    build_environment,
    validate_environment,
)
from tests.utils.scenarios import ALL_SCENARIOS


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.name)
def test_scenario_is_valid(scenario):
    env = build_environment(scenario)
    validate_environment(env, scenario)

    assert len(env.crowd) == len(scenario.humans)
    assert env.obstacles, "Scenarios must contain static obstacles."

    corridor = LineString([scenario.robot_start, scenario.robot_goal]).buffer(
        env.robot.radius + STRAIGHT_PATH_CLEARANCE
    )
    for obstacle in env.obstacles.values():
        assert not corridor.intersects(obstacle_to_shapely_polygon(obstacle))

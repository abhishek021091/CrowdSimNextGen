# tests/test_head_on.py
from __future__ import annotations

import pytest

from tests.utils.assertions import assert_navigation_ok, assert_sidestepped
from tests.utils.scenarios import head_on

# robot radius + human radius
SUM_OF_RADII = 0.6


@pytest.mark.parametrize("speed", [0.8, 1.0, 1.3])
@pytest.mark.parametrize("lateral_offset", [0.0, 0.3, -0.3])
def test_head_on_robot_avoids_pedestrian(run_scenario, speed, lateral_offset):
    result = run_scenario(head_on(speed, lateral_offset))
    assert_navigation_ok(result)
    assert_sidestepped(result, min_deviation=SUM_OF_RADII)

# tests/test_two_together.py
from __future__ import annotations

from tests.utils.assertions import assert_navigation_ok, assert_sidestepped
from tests.utils.scenarios import two_together


def test_two_humans_walking_together_head_on(run_scenario):
    result = run_scenario(two_together())
    assert_navigation_ok(result)
    # The pair leaves a 0.3 m gap, narrower than the robot: it must go around.
    assert_sidestepped(result, min_deviation=0.9)

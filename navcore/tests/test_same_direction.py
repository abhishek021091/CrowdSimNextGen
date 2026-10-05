# tests/test_same_direction.py
from __future__ import annotations

import pytest

from tests.utils.assertions import assert_follows_or_overtakes, assert_navigation_ok
from tests.utils.scenarios import same_direction


@pytest.mark.parametrize("human_speed", [0.4, 0.5, 0.8])
def test_same_direction_follow_or_overtake(run_scenario, human_speed):
    result = run_scenario(same_direction(human_speed))
    assert_navigation_ok(result)
    assert_follows_or_overtakes(result, human_index=0)

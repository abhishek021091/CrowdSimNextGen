# tests/test_from_behind.py
from __future__ import annotations

import pytest

from tests.utils.assertions import assert_navigation_ok, overtake_status
from tests.utils.scenarios import from_behind


@pytest.mark.parametrize("speed", [1.1, 1.3])
def test_human_approaching_from_behind(run_scenario, speed):
    result = run_scenario(from_behind(speed))
    assert_navigation_ok(result)
    status = overtake_status(result, human_index=0)
    assert status in ("human_overtook", "robot_stayed_ahead"), (
        f"Unexpected ordering '{status}'.\n{result.describe()}"
    )

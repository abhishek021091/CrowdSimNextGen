# tests/test_cross_right.py
from __future__ import annotations

import pytest

from tests.utils.assertions import assert_crosses_safely, assert_navigation_ok
from tests.utils.scenarios import cross_right


@pytest.mark.parametrize("speed", [0.8, 1.0, 1.3])
def test_human_crossing_from_right(run_scenario, speed):
    result = run_scenario(cross_right(speed))
    assert_navigation_ok(result)
    assert_crosses_safely(result, human_index=0)

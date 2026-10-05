# tests/test_two_crossing.py
from __future__ import annotations

from tests.utils.assertions import assert_crosses_safely, assert_navigation_ok
from tests.utils.scenarios import two_crossing


def test_two_humans_crossing_simultaneously(run_scenario):
    result = run_scenario(two_crossing())
    assert_navigation_ok(result)
    for human_index in (0, 1):
        assert_crosses_safely(result, human_index)

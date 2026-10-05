# tests/test_parallel_pedestrians.py
from __future__ import annotations

from tests.utils.assertions import assert_detour_bounded, assert_navigation_ok
from tests.utils.scenarios import parallel_pedestrians


def test_parallel_pedestrians(run_scenario):
    result = run_scenario(parallel_pedestrians())
    assert_navigation_ok(result)
    assert_detour_bounded(result, max_deviation=3.5)

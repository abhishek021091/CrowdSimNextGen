# tests/test_overtaking_humans.py
from __future__ import annotations

from tests.utils.assertions import assert_humans_overtake, assert_navigation_ok
from tests.utils.scenarios import overtaking_humans


def test_humans_overtaking_each_other(run_scenario):
    result = run_scenario(overtaking_humans())
    assert_humans_overtake(result, fast_index=1, slow_index=0)
    assert_navigation_ok(result)

# tests/test_group.py
from __future__ import annotations

from tests.utils.assertions import assert_crosses_safely, assert_navigation_ok
from tests.utils.scenarios import group_walk


def test_group_walking_together(run_scenario):
    scenario = group_walk()
    result = run_scenario(scenario)
    assert_navigation_ok(result)
    assert len(result.trajectory.human_ids) == 3
    assert_crosses_safely(result, human_index=0)  # leader

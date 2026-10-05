# tests/test_multi_direction.py
from __future__ import annotations

from tests.utils.assertions import assert_crosses_safely, assert_navigation_ok
from tests.utils.metrics import Thresholds
from tests.utils.scenarios import multi_direction

THRESHOLDS = Thresholds(min_path_efficiency=0.4, max_oscillations=12)


def test_humans_approaching_from_different_directions(run_scenario):
    result = run_scenario(multi_direction(), thresholds=THRESHOLDS)
    assert_navigation_ok(result)
    assert_crosses_safely(result, human_index=0)  # from the left
    assert_crosses_safely(result, human_index=1)  # from the right

# tests/test_dense_crowd.py
from __future__ import annotations

import pytest

from tests.utils.assertions import assert_detour_bounded, assert_navigation_ok
from tests.utils.metrics import Thresholds
from tests.utils.scenarios import dense_crowd

DENSE_THRESHOLDS = Thresholds(
    min_path_efficiency=0.35,
    max_oscillations=14,
    deadlock_window_s=15.0,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dense_crowd(run_scenario, seed):
    result = run_scenario(dense_crowd(seed), thresholds=DENSE_THRESHOLDS)
    assert len(result.trajectory.human_ids) == 14
    assert_navigation_ok(result)
    assert_detour_bounded(result, max_deviation=5.0)

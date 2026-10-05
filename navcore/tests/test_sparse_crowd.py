# tests/test_sparse_crowd.py
from __future__ import annotations

import pytest

from tests.utils.assertions import assert_detour_bounded, assert_navigation_ok
from tests.utils.scenarios import sparse_crowd


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_crowd(run_scenario, seed):
    result = run_scenario(sparse_crowd(seed))
    assert_navigation_ok(result)
    assert_detour_bounded(result, max_deviation=4.5)

# tests/conftest.py
"""Shared fixtures for the navigation regression suite."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

pytest.register_assert_rewrite("tests.utils.assertions")

from tests.utils.controllers import make_controller_from_env  # noqa: E402
from tests.utils.metrics import Thresholds  # noqa: E402
from tests.utils.runner import run_episode  # noqa: E402


@pytest.fixture(scope="session")
def controller():
    try:
        return make_controller_from_env()
    except FileNotFoundError as exc:
        pytest.skip(f"No robot policy checkpoint available: {exc}")


@pytest.fixture
def run_scenario(controller):
    def _run(scenario, *, thresholds: Thresholds | None = None, robot_visible=False):
        return run_episode(
            scenario,
            controller,
            thresholds or Thresholds(),
            robot_visible=robot_visible,
        )

    return _run

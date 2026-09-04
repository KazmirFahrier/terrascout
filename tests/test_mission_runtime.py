"""Measure runtime without coverage instrumentation: pytest -m performance --no-cov."""

import pytest

from terrascout.runner.mission import run_mission


@pytest.mark.performance
@pytest.mark.parametrize("planner_kind", ["grid", "hybrid"])
def test_mission_completes_within_runtime_budget(planner_kind: str) -> None:
    metrics = run_mission(seed=7, planner_kind=planner_kind)
    assert metrics.inspected_rows == metrics.total_rows
    assert metrics.collisions == 0
    assert metrics.wall_time_s < 5.0

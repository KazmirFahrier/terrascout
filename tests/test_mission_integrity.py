"""Independent regression cases for motion safety and honest mission accounting."""

from dataclasses import asdict
import json
from math import nan
from unittest.mock import patch

import numpy as np
import pytest

from terrascout.plan.astar import GridAStarPlanner
from terrascout.plan.hybrid_astar import HybridAStarPlanner, HybridPlannerConfig
from terrascout.runner.mission import _global_detections, run_mission
from terrascout.safety.collision_guard import SafetySupervisor
from terrascout.sim.collision import segment_clearance
from terrascout.sim.geometry import Point2D, Pose2D
from terrascout.sim.world import LocalLidarDetection, MovingAgent, OrchardWorld, ScenarioConfig


def test_unreachable_wall_does_not_become_direct_goal():
    world = OrchardWorld(ScenarioConfig(rows=4, trees_per_row=4, worker_count=0))
    world.trees = [Point2D(4.0, float(y) / 4) for y in range(48)]
    start, goal = Pose2D(2, 3, 0), Point2D(7, 3)
    for planner in (
        GridAStarPlanner(world),
        HybridAStarPlanner(world, HybridPlannerConfig(max_expansions=100)),
    ):
        assert planner.plan(start, goal) == []


def test_astar_cannot_cut_a_blocked_corner():
    world = OrchardWorld(ScenarioConfig(worker_count=0))
    planner = GridAStarPlanner(world)
    blocked = np.ones((planner.width_cells, planner.height_cells), dtype=bool)
    blocked[2, 2] = blocked[3, 3] = False
    assert planner._astar((2, 2), (3, 3), blocked) == []
    assert planner._astar((1, 1), (1, 1), blocked) == []


def test_planner_checks_whole_segments_and_body():
    world = OrchardWorld(ScenarioConfig(worker_count=0))
    world.trees = [Point2D(4, 4)]
    planner = GridAStarPlanner(world)
    assert not planner.segment_is_safe(Point2D(2, 4), Point2D(6, 4))
    assert not planner.segment_is_safe(Point2D(2, 4.6), Point2D(6, 4.6))
    path = planner.plan(Pose2D(2, 4, 0), Point2D(6, 4))
    assert len(path) > 2
    assert all(segment_clearance(a, b, world.trees[0]) > 0.63 for a, b in zip(path, path[1:]))
    assert planner.plan(Pose2D(2, 4, 0), Point2D(4, 4)) == []


def test_swept_evaluator_detects_crossing_without_endpoint_overlap():
    world = OrchardWorld(ScenarioConfig(worker_count=0))
    world.trees = [Point2D(4, 4)]
    assert world.swept_contacts(Pose2D(2, 4, 0), Pose2D(6, 4, 0), []) == {"tree:0"}
    world.trees = []
    world.workers = [MovingAgent(Point2D(4, 2), np.zeros(2))]
    contacts = world.swept_contacts(Pose2D(2, 4, 0), Pose2D(6, 4, 0), [Point2D(4, 6)])
    assert contacts == {"worker:0"}


def test_worker_motion_is_independent_of_observation_count_and_rover():
    a, b = [OrchardWorld(ScenarioConfig(worker_count=1)) for _ in range(2)]
    for _ in range(10):
        a.local_lidar_detections(Pose2D(3, 4, 0))
        a.step_workers(0.05)
        b.step_workers(0.05)
    assert a.workers[0].position == b.workers[0].position
    a.workers = [MovingAgent(Point2D(2.1, 2), np.zeros(2))]
    a.step_workers(0.05)
    assert a.collision_with_worker(Pose2D(2, 2, 0))


@pytest.mark.parametrize("age", [None, -0.01, 0.26, nan])
def test_safety_stops_for_missing_or_stale_observations(age):
    decision = SafetySupervisor().supervise(Pose2D(2, 2, 0), 1, 1, [], [], observation_age_s=age)
    assert decision.stopped and decision.left_mps == decision.right_mps == 0


def test_valid_empty_observation_is_distinct_from_missing_frame():
    decision = SafetySupervisor().supervise(Pose2D(2, 2, 0), 1, 1, [], [], observation_age_s=0)
    assert not decision.stopped


def test_perception_transform_uses_selected_pose():
    detections = _global_detections(Pose2D(10, 20, 0), [LocalLidarDetection(2, 0, "worker")])
    assert (detections[0].x, detections[0].y) == (12, 20)


def test_no_path_keeps_rover_stationary_and_reports_failure():
    with patch.object(GridAStarPlanner, "plan", return_value=[]):
        metrics = run_mission(max_steps=25, worker_count=0)
    assert metrics.path_length_m == 0
    assert metrics.plan_failures == 2
    assert metrics.success_rate == 0
    assert metrics.status == "step_limit"


@pytest.mark.parametrize("budgets", [(0, 180), (140, 0), (0, 0)])
def test_empty_schedule_never_reports_success(budgets):
    metrics = run_mission(battery_budget_m=budgets[0], daylight_budget_s=budgets[1])
    assert metrics.total_rows == 7
    assert metrics.scheduled_goals == 0
    assert metrics.success_rate == 0
    assert metrics.status == "no_feasible_goals"
    assert metrics.mission_time_s == metrics.path_length_m == 0


def test_zero_steps_and_strict_json(tmp_path):
    output = tmp_path / "trace.json"
    metrics = run_mission(max_steps=0, worker_count=0, trace_path=output)
    assert metrics.mission_time_s == 0
    assert metrics.status == "step_limit"
    assert metrics.min_worker_clearance_m is None
    json.dumps(asdict(metrics), allow_nan=False)
    assert json.loads(output.read_text())["scenario"]["headland_m"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0},
        {"dt": nan},
        {"dt": 0.2},
        {"max_steps": -1},
        {"max_steps": 2.2},
        {"planner_kind": "typo"},
        {"battery_budget_m": nan},
        {"daylight_budget_s": -1},
        {"max_goals": True},
    ],
)
def test_invalid_mission_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        run_mission(**kwargs)


def test_resource_limits_use_executed_distance_and_time():
    metrics = run_mission(worker_count=0, battery_budget_m=50, daylight_budget_s=70)
    assert metrics.path_length_m <= 50
    assert metrics.mission_time_s <= 70
    assert metrics.battery_remaining_m == pytest.approx(50 - metrics.path_length_m)
    assert metrics.daylight_remaining_s == pytest.approx(70 - metrics.mission_time_s)
    assert metrics.total_rows == 7
    assert metrics.success_rate == metrics.inspected_rows / 7
    assert metrics.success_rate < 1


def test_tree_contact_is_reported_and_terminates_mission():
    original = OrchardWorld.__init__

    def put_tree_at_start(self, config):
        original(self, config)
        self.trees.append(Point2D(1, 0.8))

    with patch.object(OrchardWorld, "__init__", put_tree_at_start):
        metrics = run_mission(max_steps=10, worker_count=0)
    assert metrics.collisions == metrics.tree_collisions == metrics.collision_frames == 1
    assert metrics.status == "collision"


def test_mission_does_not_use_global_truth_detections():
    with patch.object(OrchardWorld, "lidar_detections", side_effect=AssertionError("truth leak")):
        metrics = run_mission(max_steps=5, pose_source="truth")
    assert metrics.mean_localization_error_m == 0
    assert metrics.particle_localization_error_m > 0


def test_large_grid_route_passes_previously_stalled_corner():
    metrics = run_mission(
        seed=7,
        rows=30,
        max_goals=10,
        max_steps=2200,
        battery_budget_m=700,
        daylight_budget_s=900,
    )
    assert metrics.inspected_rows >= 3
    assert metrics.path_length_m > 40
    assert metrics.footprint_stops == metrics.collisions == 0


def test_mission_records_worker_contact_without_displacing_worker():
    original = OrchardWorld.__init__

    def put_worker_at_start(self, config):
        original(self, config)
        self.workers = [MovingAgent(Point2D(1.1, 0.8), np.zeros(2))]

    with patch.object(OrchardWorld, "__init__", put_worker_at_start):
        metrics = run_mission(max_steps=10, worker_count=0)
    assert metrics.status == "collision"
    assert metrics.collisions == metrics.worker_collisions == 1
    assert metrics.min_worker_clearance_m < 0

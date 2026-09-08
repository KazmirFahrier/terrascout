"""Keep README checks sensitive to results, without requiring identical hardware."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path

build_kpi_block = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "docs" / "update_readme_kpis.py")
)["build_kpi_block"]


class ReadmeKpiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = {
            "mission": {"success_rate": 1.0, "collisions": 0},
            "benchmark_summary": {
                "tracking_mean_prediction_error_m": 0.037,
                "tracking_mean_association_accuracy": 1.0,
                "localization_p95_pose_error_m": 0.029,
                "slam_mean_pose_error_m": 0.032,
                "slam_mean_landmark_error_m": 0.070,
                "slam_mean_landmarks": 160,
                "planner_mean_wall_time_ms": {"hybrid_astar": 28.5},
                "planner_mean_steering_reduction_percent": 86.6,
                "scheduler_max_optimality_gap_percent": 0.0,
                "scheduler_max_wall_time_ms": 20.0,
                "resource_scheduler_max_wall_time_ms": 30.0,
                "resource_scheduler_max_optimality_gap_percent": 0.0,
                "end_to_end_priority_goals": 10,
                "end_to_end_total_collisions": 0,
                "end_to_end_mean_pose_error_m": 0.201,
                "end_to_end_max_wall_time_s": 12.0,
            },
        }

    def block(self) -> str:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(self.summary))
            return build_kpi_block(path)

    def test_machine_timing_variation_within_budget_keeps_block_current(self) -> None:
        original = self.block()
        self.summary["benchmark_summary"]["planner_mean_wall_time_ms"]["hybrid_astar"] = 75.0
        self.assertEqual(self.block(), original)

    def test_timing_budget_regression_changes_block(self) -> None:
        original = self.block()
        self.summary["benchmark_summary"]["planner_mean_wall_time_ms"]["hybrid_astar"] = 251.0
        updated = self.block()
        self.assertNotEqual(updated, original)
        self.assertIn("budget missed solve time", updated)

    def test_requested_goals_are_not_reported_as_completed_goals(self) -> None:
        self.summary["benchmark_summary"]["end_to_end_min_completed_goals"] = 6
        self.assertIn("6/10 minimum completed goals", self.block())

    def test_collision_regression_changes_block(self) -> None:
        original = self.block()
        self.summary["mission"]["collisions"] = 1
        self.assertNotEqual(self.block(), original)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import unittest

from terrascout.eval.benchmarks import (
    ControlBenchmarkRow,
    LocalizationBenchmarkRow,
    PlannerBenchmarkRow,
    SchedulerBenchmarkRow,
    SlamBenchmarkRow,
    StressSummaryRow,
    TrackingBenchmarkRow,
)
from terrascout.runner.mission import MissionMetrics
from terrascout.runner.reproduce import build_reproduce_summary


class ReproduceSummaryTest(unittest.TestCase):
    def test_build_reproduce_summary_reports_outputs_and_benchmarks(self) -> None:
        mission = MissionMetrics(
            seed=7,
            inspected_rows=7,
            total_rows=7,
            success_rate=1.0,
            collisions=0,
            path_length_m=42.0,
            mission_time_s=12.0,
            wall_time_s=0.5,
            tracker_count=1,
            mapped_landmarks=80,
            slam_landmarks=75,
            slam_covariance_trace=1.2,
            mean_localization_error_m=0.2,
            planner="grid",
            pose_source="truth",
            scheduler_value=5.0,
            scheduler_dropped_goals=0,
            battery_remaining_m=90.0,
            daylight_remaining_s=60.0,
            battery_soc_final=0.9,
            battery_soc_min=0.85,
            recharge_events=1,
            safety_interventions=3,
            safety_stops=1,
            min_worker_clearance_m=1.1,
            replans=8,
        )

        summary = build_reproduce_summary(
            artifacts_dir=Path("artifacts"),
            mission=mission,
            mission_rows=[mission],
            control_rows=[ControlBenchmarkRow(7, 0.02, 0.02, 1.0, 0.04, 1.0)],
            tracking_rows=[TrackingBenchmarkRow(7, 10, 10, 0.04, 1.0, 2.0)],
            localization_rows=[LocalizationBenchmarkRow(7, 0.5, 5.0, 0.12, 1350, 4.0)],
            scheduler_rows=[SchedulerBenchmarkRow(7, 7, 10.0, 10.0, 0.0, 8, 5.0)],
            planner_rows=[
                PlannerBenchmarkRow(7, "grid_astar", 12, 18.5, 10.0, 8.0),
                PlannerBenchmarkRow(7, "hybrid_astar", 8, 19.0, 4.0, 55.0),
            ],
            slam_rows=[SlamBenchmarkRow(7, 50, 1.5, 0.04, 0.08, 0.15, 2.0, 3.0)],
            stress_rows=[StressSummaryRow("grid_truth", "grid", "truth", 3, 1.0, 0, 0.2, 0.4, 0)],
            outputs={"mission_trace_json": Path("artifacts/mission_trace.json")},
        )

        self.assertEqual(summary["artifacts_dir"], "artifacts")
        self.assertEqual(summary["mission"]["collisions"], 0)
        self.assertEqual(summary["benchmark_summary"]["mission_total_collisions"], 0)
        self.assertEqual(summary["benchmark_summary"]["control_max_cross_track_error_m"], 0.02)
        self.assertEqual(summary["benchmark_summary"]["tracking_mean_association_accuracy"], 1.0)
        self.assertEqual(summary["benchmark_summary"]["localization_p95_pose_error_m"], 0.12)
        self.assertEqual(summary["benchmark_summary"]["scheduler_max_optimality_gap_percent"], 0.0)
        self.assertEqual(
            summary["benchmark_summary"]["planner_mean_wall_time_ms"]["hybrid_astar"],
            55.0,
        )
        self.assertEqual(
            summary["benchmark_summary"]["planner_mean_steering_reduction_percent"],
            60.0,
        )
        self.assertEqual(summary["benchmark_summary"]["slam_mean_pose_error_m"], 0.04)
        self.assertEqual(summary["outputs"]["mission_trace_json"], "artifacts/mission_trace.json")


if __name__ == "__main__":
    unittest.main()

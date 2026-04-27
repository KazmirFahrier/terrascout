from __future__ import annotations

import unittest

from terrascout.eval.benchmarks import run_end_to_end_benchmark
from terrascout.runner.mission import run_mission
from terrascout.sim.world import ScenarioConfig


class MissionTest(unittest.TestCase):
    def test_default_mission_completes_without_collision(self) -> None:
        metrics = run_mission(seed=7)

        self.assertEqual(metrics.inspected_rows, metrics.total_rows)
        self.assertEqual(metrics.collisions, 0)
        self.assertGreater(metrics.mapped_landmarks, 50)
        self.assertGreater(metrics.slam_landmarks, 50)
        self.assertGreater(metrics.slam_covariance_trace, 0.0)
        self.assertLess(metrics.mean_localization_error_m, 0.5)
        self.assertEqual(metrics.planner, "grid")
        self.assertEqual(metrics.scheduler_dropped_goals, 0)
        self.assertGreater(metrics.battery_remaining_m, 0.0)
        self.assertGreater(metrics.daylight_remaining_s, 0.0)
        self.assertGreater(metrics.battery_soc_final, 0.8)
        self.assertLessEqual(metrics.battery_soc_min, metrics.battery_soc_final)
        self.assertGreaterEqual(metrics.recharge_events, 1)
        self.assertGreater(metrics.min_worker_clearance_m, 0.0)
        self.assertTrue(
            metrics.safety_interventions > 0 or metrics.min_worker_clearance_m >= 2.4
        )
        self.assertLess(metrics.wall_time_s, 5.0)

    def test_hybrid_planner_mission_completes(self) -> None:
        metrics = run_mission(seed=7, planner_kind="hybrid")

        self.assertEqual(metrics.inspected_rows, metrics.total_rows)
        self.assertEqual(metrics.collisions, 0)
        self.assertEqual(metrics.planner, "hybrid")
        self.assertLess(metrics.wall_time_s, 5.0)

    def test_particle_pose_source_mission_completes(self) -> None:
        metrics = run_mission(seed=7, pose_source="particle")

        self.assertEqual(metrics.inspected_rows, metrics.total_rows)
        self.assertEqual(metrics.collisions, 0)
        self.assertEqual(metrics.pose_source, "particle")
        self.assertLess(metrics.mean_localization_error_m, 0.5)

    def test_slam_pose_source_mission_completes(self) -> None:
        metrics = run_mission(seed=7, pose_source="slam")

        self.assertEqual(metrics.inspected_rows, metrics.total_rows)
        self.assertEqual(metrics.collisions, 0)
        self.assertEqual(metrics.pose_source, "slam")
        self.assertGreater(metrics.slam_landmarks, 50)

    def test_mission_accepts_scenario_config(self) -> None:
        config = ScenarioConfig(rows=5, trees_per_row=8, worker_count=0, random_seed=13)

        metrics = run_mission(scenario_config=config)

        self.assertEqual(metrics.seed, 13)
        self.assertGreater(metrics.total_rows, 0)
        self.assertEqual(metrics.collisions, 0)

    def test_mission_accepts_resource_budget_and_goal_limit(self) -> None:
        metrics = run_mission(
            seed=7,
            rows=12,
            worker_count=0,
            battery_budget_m=300.0,
            daylight_budget_s=500.0,
            max_goals=4,
        )

        self.assertEqual(metrics.total_rows, 4)
        self.assertEqual(metrics.inspected_rows, 4)
        self.assertEqual(metrics.scheduler_dropped_goals, 0)
        self.assertGreater(metrics.battery_remaining_m, 0.0)

    def test_end_to_end_benchmark_reports_acceptance_rows(self) -> None:
        rows = run_end_to_end_benchmark(
            seeds=[7],
            rows=8,
            trees_per_row=10,
            worker_count=0,
            priority_goals=3,
        )

        self.assertEqual(rows[0].priority_goals, 3)
        self.assertEqual(rows[0].inspected_goals, 3)
        self.assertEqual(rows[0].collisions, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

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
        self.assertGreater(metrics.safety_interventions, 0)
        self.assertGreater(metrics.min_worker_clearance_m, 0.0)
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


if __name__ == "__main__":
    unittest.main()

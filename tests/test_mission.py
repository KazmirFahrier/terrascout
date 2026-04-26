from __future__ import annotations

import unittest

from terrascout.runner.mission import run_mission


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
        self.assertLess(metrics.wall_time_s, 5.0)

    def test_hybrid_planner_mission_completes(self) -> None:
        metrics = run_mission(seed=7, planner_kind="hybrid")

        self.assertEqual(metrics.inspected_rows, metrics.total_rows)
        self.assertEqual(metrics.collisions, 0)
        self.assertEqual(metrics.planner, "hybrid")
        self.assertLess(metrics.wall_time_s, 5.0)


if __name__ == "__main__":
    unittest.main()

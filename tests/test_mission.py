from __future__ import annotations

import unittest

from terrascout.runner.mission import run_mission


class MissionTest(unittest.TestCase):
    def test_default_mission_completes_without_collision(self) -> None:
        metrics = run_mission(seed=7)

        self.assertEqual(metrics.inspected_rows, metrics.total_rows)
        self.assertEqual(metrics.collisions, 0)
        self.assertLess(metrics.wall_time_s, 5.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from terrascout.scheduler.value_iteration import InspectionScheduler
from terrascout.sim.geometry import Point2D, Pose2D


class InspectionSchedulerTest(unittest.TestCase):
    def test_scheduler_uses_priority_not_only_nearest_distance(self) -> None:
        scheduler = InspectionScheduler()
        goals = [Point2D(1.0, 0.0), Point2D(8.0, 0.0)]

        order = scheduler.plan_order(Pose2D(0.0, 0.0, 0.0), goals, priorities=[1.0, 8.0])

        self.assertEqual(order[0], 1)
        self.assertEqual(sorted(order), [0, 1])


if __name__ == "__main__":
    unittest.main()

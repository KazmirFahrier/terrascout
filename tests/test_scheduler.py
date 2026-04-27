from __future__ import annotations

import unittest

from terrascout.eval.benchmarks import run_resource_scheduler_benchmark, run_scheduler_benchmark
from terrascout.scheduler.value_iteration import InspectionScheduler
from terrascout.sim.geometry import Point2D, Pose2D


class InspectionSchedulerTest(unittest.TestCase):
    def test_scheduler_uses_priority_not_only_nearest_distance(self) -> None:
        scheduler = InspectionScheduler()
        goals = [Point2D(1.0, 0.0), Point2D(8.0, 0.0)]

        order = scheduler.plan_order(Pose2D(0.0, 0.0, 0.0), goals, priorities=[1.0, 8.0])

        self.assertEqual(order[0], 1)
        self.assertEqual(sorted(order), [0, 1])

    def test_resource_plan_drops_unreachable_goals(self) -> None:
        scheduler = InspectionScheduler()
        goals = [Point2D(3.0, 0.0), Point2D(30.0, 0.0), Point2D(35.0, 0.0)]

        plan = scheduler.plan_with_resources(
            Pose2D(0.0, 0.0, 0.0),
            goals,
            priorities=[1.0, 1.0, 10.0],
            battery_budget_m=10.0,
            daylight_budget_s=20.0,
        )

        self.assertEqual(plan.order, [0])
        self.assertEqual(plan.dropped_goals, 2)
        self.assertGreater(plan.battery_remaining_m, 0.0)

    def test_resource_plan_prefers_high_priority_when_budget_allows(self) -> None:
        scheduler = InspectionScheduler()
        goals = [Point2D(2.0, 0.0), Point2D(8.0, 0.0)]

        plan = scheduler.plan_with_resources(
            Pose2D(0.0, 0.0, 0.0),
            goals,
            priorities=[1.0, 6.0],
            battery_budget_m=20.0,
            daylight_budget_s=40.0,
        )

        self.assertEqual(plan.order[0], 1)
        self.assertEqual(plan.dropped_goals, 0)

    def test_scheduler_benchmark_matches_oracle(self) -> None:
        rows = run_scheduler_benchmark(seeds=[2, 3, 5], goal_count=7)

        self.assertLessEqual(max(row.optimality_gap_percent for row in rows), 0.01)
        self.assertLess(max(row.wall_time_ms for row in rows), 800.0)
        self.assertLessEqual(max(row.iterations for row in rows), 20)

    def test_resource_scheduler_benchmark_matches_oracle(self) -> None:
        rows = run_resource_scheduler_benchmark(seeds=range(50), goal_count=8)

        self.assertEqual(len(rows), 50)
        self.assertLessEqual(max(row.optimality_gap_percent for row in rows), 0.01)
        self.assertLess(max(row.wall_time_ms for row in rows), 800.0)
        self.assertGreater(min(row.inspected_goals for row in rows), 0)


if __name__ == "__main__":
    unittest.main()

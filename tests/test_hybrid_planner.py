from __future__ import annotations

import unittest
from math import atan2, hypot

from terrascout.plan.hybrid_astar import HybridAStarPlanner, HybridPlannerConfig
from terrascout.sim.geometry import Pose2D, distance, wrap_angle
from terrascout.sim.world import OrchardWorld, ScenarioConfig


class HybridAStarPlannerTest(unittest.TestCase):
    def test_hybrid_planner_returns_heading_aware_path(self) -> None:
        world = OrchardWorld(ScenarioConfig(rows=5, trees_per_row=8, worker_count=0, random_seed=2))
        planner = HybridAStarPlanner(
            world,
            HybridPlannerConfig(max_expansions=12_000, goal_tolerance_m=0.9),
        )
        goal = Pose2D(world.row_goals[0].x, world.row_goals[0].y, 1.57)

        path = planner.plan(Pose2D(1.0, 0.8, 1.2), goal)

        self.assertGreater(len(path), 2)
        self.assertLess(distance(path[-1], goal), 0.01)
        self.assertTrue(any(abs(wrap_angle(pose.theta - 1.2)) > 0.2 for pose in path[1:]))

    def test_hybrid_path_has_consistent_segment_headings(self) -> None:
        world = OrchardWorld(ScenarioConfig(rows=5, trees_per_row=8, worker_count=0, random_seed=5))
        planner = HybridAStarPlanner(world)
        goal = Pose2D(world.row_goals[-1].x, world.row_goals[-1].y, 0.0)

        path = planner.plan(Pose2D(1.0, 0.8, 0.2), goal)

        for current, nxt in zip(path, path[1:]):
            segment = atan2(nxt.y - current.y, nxt.x - current.x)
            if hypot(nxt.x - current.x, nxt.y - current.y) > 0.2:
                self.assertLess(abs(wrap_angle(segment - current.theta)), 1.8)


if __name__ == "__main__":
    unittest.main()


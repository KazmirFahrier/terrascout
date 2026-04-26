from __future__ import annotations

import unittest

from terrascout.plan.astar import GridAStarPlanner
from terrascout.sim.geometry import Pose2D, distance
from terrascout.sim.world import OrchardWorld, ScenarioConfig


class PlannerTest(unittest.TestCase):
    def test_planner_returns_waypoints_for_orchard_lane(self) -> None:
        world = OrchardWorld(ScenarioConfig(rows=5, trees_per_row=8, worker_count=0, random_seed=2))
        planner = GridAStarPlanner(world)
        path = planner.plan(Pose2D(1.0, 0.8, 1.2), world.row_goals[0])

        self.assertGreater(len(path), 1)
        self.assertLess(distance(path[-1], world.row_goals[0]), 1.0)


if __name__ == "__main__":
    unittest.main()


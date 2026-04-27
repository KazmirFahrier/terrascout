from __future__ import annotations

import unittest

from terrascout.control.pid import DriveController
from terrascout.eval.benchmarks import run_control_benchmark
from terrascout.sim.geometry import Point2D, Pose2D, distance
from terrascout.sim.rover import DifferentialDriveRover


class DriveControllerTest(unittest.TestCase):
    def test_rover_reaches_nearby_waypoint(self) -> None:
        rover = DifferentialDriveRover(Pose2D(0.0, 0.0, 0.0), slip_fraction=0.02)
        controller = DriveController.default()
        goal = Point2D(5.0, 0.0)
        dt = 0.05

        for _ in range(180):
            left, right = controller.wheel_commands(rover.pose, goal, dt)
            rover.command(left, right)
            rover.step(dt)

        self.assertLess(distance(rover.pose, goal), 0.7)

    def test_control_benchmark_meets_l0_acceptance_metrics(self) -> None:
        rows = run_control_benchmark(seeds=range(10))

        self.assertLessEqual(max(row.max_cross_track_error_m for row in rows), 0.03)
        self.assertLess(max(row.heading_settle_time_s for row in rows), 1.5)
        self.assertLessEqual(max(row.heading_overshoot_fraction for row in rows), 0.08)


if __name__ == "__main__":
    unittest.main()

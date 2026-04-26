from __future__ import annotations

import unittest

from terrascout.control.pid import DriveController
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


if __name__ == "__main__":
    unittest.main()


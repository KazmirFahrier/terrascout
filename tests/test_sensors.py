from __future__ import annotations

import math
import unittest

from terrascout.sim.geometry import Pose2D
from terrascout.sim.rover import DifferentialDriveRover
from terrascout.sim.sensors import SensorConfig
from terrascout.sim.world import OrchardWorld, ScenarioConfig


class SensorSuiteTest(unittest.TestCase):
    def test_lidar_scan_uses_plan_resolution_and_range(self) -> None:
        world = OrchardWorld(ScenarioConfig(rows=3, trees_per_row=4, worker_count=0, random_seed=1))
        scan = world.lidar_scan(Pose2D(2.0, 1.0, math.pi / 2.0), SensorConfig())

        self.assertEqual(len(scan.angles_rad), 541)
        self.assertEqual(len(scan.ranges_m), 541)
        self.assertAlmostEqual(scan.angles_rad[0], -math.radians(135.0))
        self.assertAlmostEqual(scan.angles_rad[-1], math.radians(135.0))
        self.assertTrue(all(0.0 <= value <= scan.max_range_m for value in scan.ranges_m))
        self.assertLess(min(scan.ranges_m), scan.max_range_m)

    def test_sensor_frame_contains_imu_and_encoder_samples(self) -> None:
        world = OrchardWorld(ScenarioConfig(worker_count=0, random_seed=2))
        rover = DifferentialDriveRover(pose=Pose2D(1.0, 0.8, 0.0), slip_fraction=0.02)
        rover.command(0.8, 1.0)

        frame = world.sensor_frame(rover, dt=0.05, sensor_config=SensorConfig())

        self.assertEqual(len(frame.lidar.ranges_m), 541)
        self.assertGreater(frame.imu.yaw_rate_rps, 0.0)
        self.assertGreater(frame.encoders.left_delta_m, 0.0)
        self.assertGreater(frame.encoders.right_delta_m, frame.encoders.left_delta_m * 0.8)


if __name__ == "__main__":
    unittest.main()
